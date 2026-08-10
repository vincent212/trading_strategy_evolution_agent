"""
The evolvable unit — now a TRAINABLE strategy. Every candidate the LLM emits
declares two things:

  param_space() -> dict of tunable knobs and their ranges
  strategy(data, tools, p) -> a position Series in [-1, 1], using p's values

The harness fits `p` on the train quarters (SciPy differential evolution,
maximizing Sharpe) and evaluates the fitted strategy on the held-out test
quarters. The LLM never runs the fit; it only designs the parametric form.

Contract for every accepted candidate:
  - param_space() returns {name: ("float"|"int", lo, hi) | ("cat", [choices])}
  - strategy(data, tools, p): data has a single column 'close' (the traded asset);
    read parameters via p["name"]; return a pd.Series aligned to data.index in
    [-1, 1]; end with tools.clip_signal(...).
  - Causal only (tools look backward; never index the future; the backtester adds
    the 1-bar execution lag). Pure function: no imports, no I/O, no randomness.
"""


def param_space():
    """Declare the tunable parameters and the RANGE the optimizer may search.
    These are search bounds, not chosen values — differential_evolution picks the
    value inside each range that maximizes train Sharpe (see strategy())."""
    return {
        "fast":   ("int", 5, 60),        # short SMA window (bars) - the fast trend line
        "slow":   ("int", 60, 250),      # long  SMA window (bars) - the slow trend line
        "rsi_n":  ("int", 5, 30),        # RSI window (bars) for the oversold dip
        "tilt":   ("float", -1.0, 1.0),  # weight on the RSI-dip term (+ adds, - trims)
        "vol_n":  ("int", 10, 60),       # realized-vol window (bars)
        "vol_lo": ("float", -2.0, 0.0),  # low-vol z-score cutoff (scale-free)
        "vol_hi": ("float", 0.0, 2.0),   # high-vol z-score cutoff (scale-free)
        "vol_w":  ("float", -1.0, 1.0),  # weight on the vol regime (+1 low vol / -1 high vol)
    }


def strategy(data, tools, p):
    """A trainable trend-follower with an RSI-oversold tilt and a realized-vol regime.

    Args:
        data:  DataFrame with a single column 'close' (the traded asset).
        tools: causal indicator library (tools.sma, tools.rsi, tools.vol_regime, ...).
        p:     dict of parameter VALUES the optimizer chose for THIS call, e.g.
               {"fast": 41, "slow": 118, "rsi_n": 9, "tilt": 0.3, "vol_n": 20,
                "vol_lo": 0.35, "vol_hi": 0.80, "vol_w": -0.5}. The strategy only
               reads p; differential_evolution sets it to maximize train Sharpe.

    Parameters (each read from p, each declared in param_space):
        p["fast"]   short SMA window   - fast trend line
        p["slow"]   long  SMA window   - slow trend line
        p["rsi_n"]  RSI window         - lookback for the oversold check (RSI < 30)
        p["tilt"]   weight             - how much an oversold dip adds to (+) / trims (-)
        p["vol_n"]  realized-vol window
        p["vol_lo"] low-vol cutoff  - vol below this Z-SCORE of recent vol = low vol
        p["vol_hi"] high-vol cutoff - vol above this Z-SCORE of recent vol = high vol
        p["vol_w"]  weight             - how the vol regime shifts exposure (+1 low, -1 high)

    Returns:
        pd.Series position in [-1, 1]: long when fast SMA > slow SMA, nudged by the
        RSI dip and the realized-vol regime, clipped to [-1, 1].
    """
    close = data["close"]
    trend = (tools.sma(close, p["fast"]) > tools.sma(close, p["slow"])).astype(float)
    dip = (tools.rsi(close, p["rsi_n"]) < 30).astype(float)
    vol = tools.vol_regime(close, p["vol_n"], p["vol_lo"], p["vol_hi"])   # +1 low vol, -1 high vol
    signal = trend + p["tilt"] * dip + p["vol_w"] * vol
    return tools.clip_signal(signal)


SEED_CODE = '''\
def param_space():
    """Declare the tunable parameters and the RANGE the optimizer may search.
    These are search bounds, not chosen values — differential_evolution picks the
    value inside each range that maximizes train Sharpe (see strategy())."""
    return {
        "fast":   ("int", 5, 60),        # short SMA window (bars) - the fast trend line
        "slow":   ("int", 60, 250),      # long  SMA window (bars) - the slow trend line
        "rsi_n":  ("int", 5, 30),        # RSI window (bars) for the oversold dip
        "tilt":   ("float", -1.0, 1.0),  # weight on the RSI-dip term (+ adds, - trims)
        "vol_n":  ("int", 10, 60),       # realized-vol window (bars)
        "vol_lo": ("float", -2.0, 0.0),  # low-vol z-score cutoff (scale-free)
        "vol_hi": ("float", 0.0, 2.0),   # high-vol z-score cutoff (scale-free)
        "vol_w":  ("float", -1.0, 1.0),  # weight on the vol regime (+1 low vol / -1 high vol)
    }


def strategy(data, tools, p):
    """A trainable trend-follower with an RSI-oversold tilt and a realized-vol regime.

    Args:
        data:  DataFrame with a single column 'close' (the traded asset).
        tools: causal indicator library (tools.sma, tools.rsi, tools.vol_regime, ...).
        p:     dict of parameter VALUES the optimizer chose for THIS call, e.g.
               {"fast": 41, "slow": 118, "rsi_n": 9, "tilt": 0.3, "vol_n": 20,
                "vol_lo": 0.35, "vol_hi": 0.80, "vol_w": -0.5}. The strategy only
               reads p; differential_evolution sets it to maximize train Sharpe.

    Parameters (each read from p, each declared in param_space):
        p["fast"]   short SMA window   - fast trend line
        p["slow"]   long  SMA window   - slow trend line
        p["rsi_n"]  RSI window         - lookback for the oversold check (RSI < 30)
        p["tilt"]   weight             - how much an oversold dip adds to (+) / trims (-)
        p["vol_n"]  realized-vol window
        p["vol_lo"] low-vol cutoff  - vol below this Z-SCORE of recent vol = low vol
        p["vol_hi"] high-vol cutoff - vol above this Z-SCORE of recent vol = high vol
        p["vol_w"]  weight             - how the vol regime shifts exposure (+1 low, -1 high)

    Returns:
        pd.Series position in [-1, 1]: long when fast SMA > slow SMA, nudged by the
        RSI dip and the realized-vol regime, clipped to [-1, 1].
    """
    close = data["close"]
    trend = (tools.sma(close, p["fast"]) > tools.sma(close, p["slow"])).astype(float)
    dip = (tools.rsi(close, p["rsi_n"]) < 30).astype(float)
    vol = tools.vol_regime(close, p["vol_n"], p["vol_lo"], p["vol_hi"])   # +1 low vol, -1 high vol
    signal = trend + p["tilt"] * dip + p["vol_w"] * vol
    return tools.clip_signal(signal)
'''


# --- other starting families, one per island (so the search does not all start as trend) ---

MEANREV_CODE = '''\
def param_space():
    return {"z_n": ("int", 5, 60), "z_k": ("float", 0.5, 3.0), "w": ("float", 0.0, 1.0)}


def strategy(data, tools, p):
    close = data["close"]
    z = tools.zscore(close, p["z_n"])
    # mean reversion: long when oversold (z below -k), short when overbought (z above +k)
    sig = ((z < -p["z_k"]).astype(float) - (z > p["z_k"]).astype(float)) * p["w"]
    return tools.clip_signal(sig)
'''

BREAKOUT_CODE = '''\
def param_space():
    return {"bo_n": ("int", 10, 120), "vol_n": ("int", 10, 60),
            "vol_hi": ("float", 0.0, 2.0), "w": ("float", 0.0, 1.0)}


def strategy(data, tools, p):
    close = data["close"]
    bo = tools.breakout(close, p["bo_n"])                 # +1 new high, -1 new low
    calm = (tools.vol_regime(close, p["vol_n"], -0.5, p["vol_hi"]) >= 0).astype(float)
    sig = p["w"] * bo * calm
    return tools.clip_signal(sig)
'''

MOMENTUM_CODE = '''\
def param_space():
    return {"roc_n": ("int", 10, 120), "vol_n": ("int", 10, 60),
            "target": ("float", 0.1, 0.4), "w": ("float", -1.0, 1.0)}


def strategy(data, tools, p):
    close = data["close"]
    mom = tools.roc(close, p["roc_n"])
    sig = p["w"] * ((mom > 0).astype(float) - (mom < 0).astype(float))
    scale = tools.vol_target_scale(close, p["target"], p["vol_n"], 1.0)
    return tools.clip_signal(sig * scale)
'''

# Trend, mean-reversion, breakout, momentum — one family seeded per island.
SEEDS = [SEED_CODE, MEANREV_CODE, BREAKOUT_CODE, MOMENTUM_CODE]
