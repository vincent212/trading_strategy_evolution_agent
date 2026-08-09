"""
Builds the mutation prompt: show the LLM two high-scoring parent strategies (each a
trainable param_space() + strategy()) and ask for an improved, distinct child.
"""
from __future__ import annotations
import alpha_tools  # noqa: F401 (kept for parity / future tool listing)

_TOOL_DOC = """\
sma(x, n) ema(x, n) roc(x, n) zscore(x, n)          -- trend / momentum
rsi(x, n=14)                                         -- oscillator, 0..100
realized_vol(x, n=20) vol_target_scale(x, target_vol, n, max_leverage)
rolling_high(x, n) rolling_low(x, n) breakout(x, n)  -- breakout / range
vix_regime(vix, low=15, high=25) pctile_rank(x, n)   -- regime
crossover(fast, slow) clip_signal(sig)               -- helpers"""

SYSTEM = f"""You are a quantitative researcher evolving TRAINABLE trading strategies for a \
single stock or ETF. The traded asset is data['close']; data['vix'] is the market VIX \
regime (may be NaN for some names -- handle gracefully).

Each strategy is EXACTLY two functions, with these exact signatures:

  def param_space():
      # tunable knobs a numerical optimizer will fit; return a dict of:
      #   name: ("int", lo, hi) | ("float", lo, hi) | ("cat", [choice, ...])
      return {{"fast": ("int", 5, 50), "slow": ("int", 60, 200), "w": ("float", -1.0, 1.0)}}

  def strategy(data, tools, p):
      close = data["close"]
      # build the signal ONLY from tools.* calls plus simple arithmetic
      return tools.clip_signal(sig)

RULES — every one is mandatory; breaking any makes the code fail and be discarded:
1. Build ALL indicators by calling tools.* ONLY, reading parameters from p
   (e.g. tools.rsi(close, p["rsi_n"])).
2. NEVER call a pandas/numpy method to compute an indicator. FORBIDDEN and WILL crash:
   .rolling(...), .ewm(...), .expanding(...), .apply(...), .resample(...), win_type=,
   np.anything, importing anything.
3. The ONLY operations allowed on the Series that tools return: + - * / , comparisons
   (>, <, >=, <=), .astype(float), and multiplying by a number from p. Combine tool
   outputs with these to form `sig`.
4. Declare 2-6 knobs in param_space(). EVERY p["..."] read in strategy() MUST be declared.
5. The LAST line of strategy() must be exactly:  return tools.clip_signal(sig)
6. Pure function: no imports, no I/O, no randomness, no loops over rows/dates, no .iloc.

tools you may call (as tools.NAME(...)):
{_TOOL_DOC}

VALID EXAMPLE — copy this structure, then vary the ideas and knobs:

```python
def param_space():
    return {{"fast": ("int", 5, 50), "slow": ("int", 60, 200),
             "rsi_n": ("int", 7, 30), "w": ("float", -1.0, 1.0)}}

def strategy(data, tools, p):
    close = data["close"]
    trend = (tools.sma(close, p["fast"]) > tools.sma(close, p["slow"])).astype(float)
    dip = (tools.rsi(close, p["rsi_n"]) < 35).astype(float)
    sig = trend + p["w"] * dip
    return tools.clip_signal(sig)
```

Write a NEW pair that is DIFFERENT from the parents below — a new mechanism, different
tools, or better knobs. Output ONLY one ```python code block with BOTH functions. No prose."""


def build_user_prompt(parents: list[dict]) -> str:
    """parents: list of {code, score, diagnostics} dicts, best first."""
    blocks = []
    for i, p in enumerate(parents):
        d = p.get("diagnostics", {})
        blocks.append(
            f"# Parent v{i}  (median_oos_sharpe={d.get('median_oos', float('nan')):.3f}, "
            f"frac_positive={d.get('frac_positive', float('nan')):.2f})\n"
            f"{p['code'].strip()}"
        )
    joined = "\n\n".join(blocks)
    return (f"{joined}\n\n"
            f"Write strategy v{len(parents)}: a better, distinct TRAINABLE strategy "
            f"(both param_space and strategy). Return only the ```python code block.")
