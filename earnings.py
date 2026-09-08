"""
Point-in-time earnings-surprise (time-series SUE) from SEC EDGAR XBRL — free and
AS-FILED, so it carries no look-ahead.

SUE_q = (EPS_q - EPS_{q-4}) / rolling_std(EPS - EPS_{q-4}, 8q)   [year-over-year
seasonal difference, standardized]. Each quarter's SUE is stamped at its FILING
date + 1 trading day (never the period-end — that is the #1 leak) and forward-filled
onto the daily price index, so on any date the value is exactly what was public then.

Deterministic feature, computed offline and cached — NOT a live LLM scoring text
(which would inherit training-cutoff look-ahead). Data hygiene handled: 3-month
periods only (drops YTD cumulatives), originally-filed value (drops restatements),
same-fiscal-quarter YoY difference.
"""
from __future__ import annotations
import os
import json
import urllib.request
import numpy as np
import pandas as pd

from data import CACHE_DIR, _safe

_UA = {"User-Agent": "latss-research v@m2te.ch"}          # SEC requires a UA with contact


def _get(url: str):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=30))


_TICKER_CIK = None


def ticker_to_cik(ticker: str) -> int | None:
    """Map a ticker to its SEC CIK via the public company_tickers.json (cached)."""
    global _TICKER_CIK
    if _TICKER_CIK is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, "company_tickers.json")
        m = json.load(open(path)) if os.path.exists(path) else _get(
            "https://www.sec.gov/files/company_tickers.json")
        if not os.path.exists(path):
            json.dump(m, open(path, "w"))
        _TICKER_CIK = {v["ticker"].upper(): int(v["cik_str"]) for v in m.values()}
    return _TICKER_CIK.get(ticker.upper())


def quarterly_eps(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """As-filed quarterly diluted EPS: columns [period_end, filed, eps], one row per
    fiscal quarter (3-month period), keeping the ORIGINALLY-filed value."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"eps_{_safe(ticker)}.parquet")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    cik = ticker_to_cik(ticker)
    if cik is None:
        raise RuntimeError(f"no SEC CIK for {ticker}")
    d = _get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/"
             f"EarningsPerShareDiluted.json")
    rows = []
    for u in d["units"].get("USD/shares", []):
        s, e, f, v = u.get("start"), u.get("end"), u.get("filed"), u.get("val")
        if not (s and e and f and v is not None):
            continue
        days = (pd.Timestamp(e) - pd.Timestamp(s)).days
        if 70 <= days <= 100:                                 # 3-month period only (drop YTD)
            rows.append((pd.Timestamp(e), pd.Timestamp(f), float(v)))
    df = pd.DataFrame(rows, columns=["period_end", "filed", "eps"])
    # originally filed: earliest filing per fiscal quarter (drop later restatements)
    df = (df.sort_values(["period_end", "filed"])
            .groupby("period_end", as_index=False).first())
    df.to_parquet(path)
    return df


def sue_series(ticker: str, index: pd.DatetimeIndex, refresh: bool = False) -> pd.Series:
    """Point-in-time SUE forward-filled onto `index`. NaN before the first usable
    quarter (needs >=5 quarters). Standing value between earnings (no decay)."""
    df = quarterly_eps(ticker, refresh=refresh).sort_values("period_end")
    if df.empty:
        return pd.Series(np.nan, index=index, name="sue")
    # collapse to one row per FISCAL QUARTER and reindex onto a CONTINUOUS quarterly grid,
    # so shift(4) is exactly the same quarter one year earlier. A missing quarter becomes
    # NaN rather than silently making the "YoY" difference span 9 or 15 months.
    df = df.assign(q=pd.PeriodIndex(df["period_end"], freq="Q")).drop_duplicates("q", keep="last")
    q = df.set_index("q").sort_index()
    grid = pd.period_range(q.index.min(), q.index.max(), freq="Q")
    eps = q["eps"].reindex(grid)
    filed = q["filed"].reindex(grid)
    seas = eps - eps.shift(4)                                 # YoY same-quarter difference (gap-aware)
    sd = seas.rolling(8, min_periods=4).std()
    sue = (seas / sd.replace(0.0, np.nan)).rename("sue")
    stamp = pd.DataFrame({"sue": sue, "filed": filed}).dropna()   # only real quarters that have a filing
    if stamp.empty:
        return pd.Series(np.nan, index=index, name="sue")
    # index the SUE by (filing date + 1 trading day), then as-of forward-fill onto `index`
    filed_next = pd.to_datetime(stamp["filed"].values) + pd.Timedelta(days=1)
    s = pd.Series(stamp["sue"].values, index=filed_next).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(index)).ffill().reindex(index).rename("sue")


if __name__ == "__main__":
    import data_mag7
    px = data_mag7.get_panel()
    print("Most-recent point-in-time SUE per Mag-7 name (as of last bar):")
    for t in data_mag7.MAG7:
        s = sue_series(t, px.index)
        eps = quarterly_eps(t)
        print(f"  {t:6s} SUE {s.iloc[-1]:+5.2f}   ({len(eps)} quarters, "
              f"latest filed {eps['filed'].max().date()})")
