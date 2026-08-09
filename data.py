"""
Data loader: any Yahoo Finance ticker (the asset we trade) plus ^VIX (a market
regime input), via yfinance, cached to a local parquet so repeated runs don't
re-hit the network.

The traded asset is whatever `ticker` you pass — SPY, AAPL, TSLA, an ETF, etc.
^VIX is always fetched as an auxiliary regime column; it stays useful as a
market-fear gauge even for single names. If a ticker has no VIX overlap (or the
VIX fetch fails), the `vix` column is left as NaN and regime tools simply return
neutral for that stock.
"""
from __future__ import annotations
import os
import re
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def _safe(name: str) -> str:
    """Filesystem-safe token for a ticker (handles ^, =, /, etc.)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def get_data(ticker: str = "SPY", start: str = "2005-01-01", end: str | None = None,
             refresh: bool = False, include_vix: bool = True) -> pd.DataFrame:
    """
    Return a DataFrame indexed by date with columns:
        close : `ticker` adjusted close (the asset traded)
        vix   : ^VIX close, forward-filled onto the asset's calendar
                (all-NaN if include_vix is False or the VIX fetch is empty)

    Cached to .cache/<ticker>_<start>_<end>.parquet.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"{_safe(ticker)}_{start}_{end or 'latest'}"
    path = os.path.join(CACHE_DIR, f"{tag}.parquet")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)

    import yfinance as yf  # imported lazily so the module loads without it

    asset = yf.download(ticker, start=start, end=end, auto_adjust=True,
                        progress=False)
    if asset.empty:
        raise RuntimeError(
            f"yfinance returned no data for '{ticker}' — check the symbol / dates / network.")

    def _col(df, name):
        if isinstance(df.columns, pd.MultiIndex):
            return df[name].iloc[:, 0]
        return df[name]

    df = pd.DataFrame({"close": _col(asset, "Close")})

    df["vix"] = pd.NA
    if include_vix:
        try:
            vix = yf.download("^VIX", start=start, end=end, auto_adjust=False,
                              progress=False)
            if not vix.empty:
                df["vix"] = _col(vix, "Close").reindex(df.index).ffill()
        except Exception:
            pass  # leave vix as NaN; regime tools return neutral

    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    df = df.dropna(subset=["close"]).copy()
    df.to_parquet(path)
    return df


if __name__ == "__main__":
    import sys
    tkr = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    d = get_data(tkr)
    print(d.tail())
    print(f"\n{tkr}: {len(d)} rows, {d.index[0].date()} -> {d.index[-1].date()}")
