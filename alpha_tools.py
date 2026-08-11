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

def vol_regime(x: pd.Series, n: int = 20, low: float = -0.5, high: float = 0.5,
               lookback: int = 126) -> pd.Series:
    """
    Realized-volatility regime by rolling Z-SCORE of the asset's own vol — scale-free (the
    z-score normalizes each ticker's vol) and fast:
      +1 = low vol   (vol z-score < low),
      -1 = high vol  (vol z-score > high),
       0 = in between.
    z = (realized_vol - rolling_mean) / rolling_std over `lookback` bars. `low`/`high` are
    z-score cutoffs (e.g. -0.5 / +0.5) — the SAME on any ticker, so no per-ticker tuning.
    """
    rv = realized_vol(x, n=n)
    m = rv.rolling(int(lookback), min_periods=int(n)).mean()
    s = rv.rolling(int(lookback), min_periods=int(n)).std().replace(0.0, np.nan)
    z = (rv - m) / s
    reg = pd.Series(0.0, index=x.index)
    reg[z < low] = 1.0
    reg[z > high] = -1.0
    return reg


def pctile_rank(x: pd.Series, n: int) -> pd.Series:
    """Rolling percentile rank of the current bar within its n-bar window, in [0, 1]."""
    return x.rolling(int(n), min_periods=int(n)).apply(
        lambda w: float((w <= w[-1]).mean()), raw=True)   # raw=True -> w is ndarray, w[-1] is current


# ---- helpers the evolved code may find handy --------------------------------

def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 where fast crosses above slow, -1 where it crosses below, else 0."""
    diff = (fast - slow)
    sign = np.sign(diff)
    return pd.Series(np.where(sign.diff() > 0, 1.0,
                     np.where(sign.diff() < 0, -1.0, 0.0)), index=fast.index)


# Position cap (leverage). 1.0 = fully long/short only (default). run_search sets this per run;
# clip_signal and the backtest both respect it, so a strategy can lever UP to MAX_LEVERAGE in
# favorable regimes (the only way to beat buy-and-hold on total return, since B&H is 1.0 long).
MAX_LEVERAGE = 1.0


def clip_signal(sig) -> pd.Series:
    """Clean a raw signal into a valid position in [-MAX_LEVERAGE, MAX_LEVERAGE].
    Accepts a pandas Series OR a numpy array (e.g. the result of np.where, which drops the index);
    an array is cleaned here and the backtest re-aligns it positionally to the price index."""
    lev = float(MAX_LEVERAGE)
    if isinstance(sig, pd.Series):
        return sig.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-lev, lev)
    arr = np.nan_to_num(np.asarray(sig, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -lev, lev)


# Names exposed to the LLM in the prompt (see prompt.py).
TOOL_NAMES = [
    "sma", "ema", "roc", "zscore", "rsi", "realized_vol", "vol_target_scale",
    "rolling_high", "rolling_low", "breakout", "vol_regime",
    "pctile_rank", "crossover", "clip_signal",
]
