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
  - strategy(data, tools, p): data has columns 'close' and 'vix' (vix may be NaN);
    read parameters via p["name"]; return a pd.Series aligned to data.index in
    [-1, 1]; end with tools.clip_signal(...).
  - Causal only (tools look backward; never index the future; the backtester adds
    the 1-bar execution lag). Pure function: no imports, no I/O, no randomness.
"""


def param_space():
    return {
        "fast":  ("int", 5, 60),
        "slow":  ("int", 60, 250),
        "rsi_n": ("int", 5, 30),
        "tilt":  ("float", -1.0, 1.0),
    }


def strategy(data, tools, p):
    close = data["close"]
    trend = (tools.sma(close, p["fast"]) > tools.sma(close, p["slow"])).astype(float)
    dip = (tools.rsi(close, p["rsi_n"]) < 30).astype(float)
    signal = trend + p["tilt"] * dip
    return tools.clip_signal(signal)


SEED_CODE = '''\
def param_space():
    return {
        "fast":  ("int", 5, 60),
        "slow":  ("int", 60, 250),
        "rsi_n": ("int", 5, 30),
        "tilt":  ("float", -1.0, 1.0),
    }


def strategy(data, tools, p):
    close = data["close"]
    trend = (tools.sma(close, p["fast"]) > tools.sma(close, p["slow"])).astype(float)
    dip = (tools.rsi(close, p["rsi_n"]) < 30).astype(float)
    signal = trend + p["tilt"] * dip
    return tools.clip_signal(signal)
'''
