"""
Cross-sectional data loader: a PANEL of several tickers on a common date index,
for the "pick 1 of N" selection experiment (default universe = the Magnificent 7).

Unlike data.py (one traded asset), this returns a wide close matrix — one column
per ticker, inner-joined on dates so every row has a price for every name. Prices
are split/dividend adjusted (auto_adjust). Cached to parquet like data.py.
"""
from __future__ import annotations
import os
from datetime import date
import pandas as pd

from data import CACHE_DIR, _safe, get_data

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]


def get_panel(tickers=MAG7, start: str = "2015-01-01", end: str | None = None,
              refresh: bool = False) -> pd.DataFrame:
    """Wide adjusted-close matrix indexed by date, columns = `tickers` (in order).

    Every column is fetched via data.get_data (so it reuses the same per-ticker
    parquet cache), then inner-joined on the common trading days — a row survives
    only if every name has a close that day. Cached whole to
    .cache/panel_<tickers>_<start>_<end>.parquet.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    # date-stamp the cache when end is open ('latest'), so a panel built on an earlier day
    # is not silently reused (missing the newest bars) without --refresh-data. The same
    # resolved end is passed to get_data so the per-ticker caches are date-keyed too.
    end_r = end or date.today().isoformat()
    tag = f"panel_{'-'.join(_safe(t) for t in tickers)}_{start}_{end_r}"
    path = os.path.join(CACHE_DIR, f"{tag}.parquet")
    if os.path.exists(path) and not refresh:
        cached = pd.read_parquet(path)
        if list(cached.columns) == list(tickers):
            return cached

    closes = {}
    for t in tickers:
        df = get_data(ticker=t, start=start, end=end_r, refresh=refresh)
        closes[t] = df["close"]
    panel = pd.DataFrame(closes)[list(tickers)].dropna(how="any")
    if len(panel) < 300:
        raise RuntimeError(f"panel too short after aligning {tickers}: {len(panel)} rows "
                           f"(a late-IPO name like META/TSLA may be trimming the common window)")
    panel.to_parquet(path)
    return panel


if __name__ == "__main__":
    p = get_panel()
    print(p.tail())
    print(f"\n{list(p.columns)}: {len(p)} rows, {p.index[0].date()} -> {p.index[-1].date()}")
