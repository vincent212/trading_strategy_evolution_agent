"""
Data loader: any Yahoo Finance ticker (the asset we trade), via yfinance, cached
to a local parquet so repeated runs don't re-hit the network.

The traded asset is whatever `ticker` you pass — SPY, AAPL, TSLA, an ETF, etc.
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
             refresh: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame indexed by date with a single column:
        close : `ticker` split/dividend-adjusted close (the asset traded)

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
    df = df.dropna(subset=["close"]).copy()
    df.to_parquet(path)
    return df


if __name__ == "__main__":
    import sys
    tkr = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    d = get_data(tkr)
    print(d.tail())
    print(f"\n{tkr}: {len(d)} rows, {d.index[0].date()} -> {d.index[-1].date()}")
