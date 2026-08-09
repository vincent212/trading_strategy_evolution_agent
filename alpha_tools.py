"""
Alpha tools: the fixed building-block library the evolved strategies may call.

Every function is CAUSAL (uses only past/current data) by construction — rolling
windows look backward, and where a value would otherwise peek at the current bar
we .shift(1). The backtester applies an *additional* one-bar execution lag on the
final signal, so a strategy decides at the close of bar t and trades bar t+1.
This is the single most important defense against look-ahead bias; do not remove
the shifts here or in backtest.run_backtest.

All functions take/return pandas Series aligned to the price index.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ---- trend / momentum -------------------------------------------------------

def sma(x: pd.Series, n: int) -> pd.Series:
    """Simple moving average over n bars."""
    return x.rolling(int(n), min_periods=int(n)).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    """Exponential moving average (span = n)."""
    return x.ewm(span=int(n), min_periods=int(n), adjust=False).mean()


def roc(x: pd.Series, n: int) -> pd.Series:
    """Rate of change (momentum) over n bars: x_t / x_{t-n} - 1."""
    return x.pct_change(int(n))


def zscore(x: pd.Series, n: int) -> pd.Series:
    """Rolling z-score of x over an n-bar window."""
    m = x.rolling(int(n), min_periods=int(n)).mean()
    s = x.rolling(int(n), min_periods=int(n)).std()
    return (x - m) / s.replace(0.0, np.nan)


# ---- oscillators ------------------------------------------------------------

def rsi(x: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]."""
    d = x.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / int(n), min_periods=int(n), adjust=False).mean()
    roll_dn = dn.ewm(alpha=1.0 / int(n), min_periods=int(n), adjust=False).mean()
    rs = roll_up / roll_dn.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ---- volatility -------------------------------------------------------------

def realized_vol(x: pd.Series, n: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualized rolling realized volatility of returns of x."""
    r = x.pct_change()
    return r.rolling(int(n), min_periods=int(n)).std() * np.sqrt(periods_per_year)


def vol_target_scale(x: pd.Series, target_vol: float = 0.15, n: int = 20,
                     max_leverage: float = 1.0) -> pd.Series:
    """
    Position-sizing multiplier in [0, max_leverage] that scales exposure inversely
    to recent realized vol, aiming for `target_vol` annualized. Multiply a raw
    signal by this to get a vol-targeted position.
    """
    rv = realized_vol(x, n=n).replace(0.0, np.nan)
    return (target_vol / rv).clip(upper=max_leverage).fillna(0.0)


# ---- breakout / range -------------------------------------------------------

def rolling_high(x: pd.Series, n: int) -> pd.Series:
    """Highest value over the PRIOR n bars (excludes the current bar)."""
    return x.rolling(int(n), min_periods=int(n)).max().shift(1)


def rolling_low(x: pd.Series, n: int) -> pd.Series:
    """Lowest value over the PRIOR n bars (excludes the current bar)."""
    return x.rolling(int(n), min_periods=int(n)).min().shift(1)


def breakout(x: pd.Series, n: int) -> pd.Series:
    """+1 on a new n-bar high, -1 on a new n-bar low, 0 otherwise."""
    hi = rolling_high(x, n)
    lo = rolling_low(x, n)
    sig = pd.Series(0.0, index=x.index)
    sig[x > hi] = 1.0
    sig[x < lo] = -1.0
    return sig


# ---- regime -----------------------------------------------------------------

def vix_regime(vix: pd.Series, low: float = 15.0, high: float = 25.0) -> pd.Series:
    """
    Coarse volatility regime from the VIX level:
      +1 = calm  (vix < low), 0 = normal, -1 = stressed (vix > high).
    """
    reg = pd.Series(0.0, index=vix.index)
    reg[vix < low] = 1.0
    reg[vix > high] = -1.0
    return reg


def pctile_rank(x: pd.Series, n: int) -> pd.Series:
    """Rolling percentile rank of x in [0, 1] over an n-bar window."""
    return x.rolling(int(n), min_periods=int(n)).apply(
        lambda w: (w.argsort().argsort()[-1] + 1) / len(w), raw=False)


# ---- helpers the evolved code may find handy --------------------------------

def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 where fast crosses above slow, -1 where it crosses below, else 0."""
    diff = (fast - slow)
    sign = np.sign(diff)
    return pd.Series(np.where(sign.diff() > 0, 1.0,
                     np.where(sign.diff() < 0, -1.0, 0.0)), index=fast.index)


def clip_signal(sig: pd.Series) -> pd.Series:
    """Clean a raw signal into a valid position in [-1, 1]."""
    return sig.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)


# Names exposed to the LLM in the prompt (see prompt.py).
TOOL_NAMES = [
    "sma", "ema", "roc", "zscore", "rsi", "realized_vol", "vol_target_scale",
    "rolling_high", "rolling_low", "breakout", "vix_regime", "pctile_rank",
    "crossover", "clip_signal",
]
