"""
The evolvable unit. This is the ONLY code the LLM rewrites — everything else
(data, tools, backtest, gate) is fixed. The loop seeds its first island with
this trivial trend-follower, then mutates from here.

Contract for every strategy the loop accepts:
  - signature: def strategy(data, tools) -> pd.Series
  - `data`  : DataFrame with columns 'close' and 'vix', indexed by date
  - `tools` : the alpha_tools module (call tools.rsi(...), tools.sma(...), etc.)
  - returns : a pd.Series aligned to data.index, values in [-1, 1]
              (+1 = full long, -1 = full short, 0 = flat)
  - MUST be causal: use only the provided tools and pandas; never index into the
    future. The backtester adds a 1-bar execution lag on top, so do not shift the
    final signal yourself.
"""


def strategy(data, tools):
    close = data["close"]
    fast = tools.sma(close, 50)
    slow = tools.sma(close, 200)
    signal = (fast > slow).astype(float)      # long in an uptrend, flat otherwise
    return tools.clip_signal(signal)


SEED_CODE = '''\
def strategy(data, tools):
    close = data["close"]
    fast = tools.sma(close, 50)
    slow = tools.sma(close, 200)
    signal = (fast > slow).astype(float)
    return tools.clip_signal(signal)
'''
