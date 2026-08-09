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

Each strategy has TWO functions:

  def param_space():
      # the tunable knobs an optimizer will fit; return a dict of:
      #   name: ("float", lo, hi)  |  ("int", lo, hi)  |  ("cat", [choice, ...])
      return {{"fast": ("int", 5, 60), "slow": ("int", 60, 250), "tilt": ("float", -1.0, 1.0)}}

  def strategy(data, tools, p):
      # read parameters via p["name"]; build a position Series in [-1, 1]
      ...
      return tools.clip_signal(sig)

Write a NEW pair that is meaningfully different from the parents -- combine their ideas, \
introduce a new mechanism, or expose better-chosen tunable parameters. The parameters are \
fit by a numerical optimizer (differential evolution) to maximize Sharpe on the training \
quarters; you only design the form, so DECLARE ENOUGH MEANINGFUL PARAMETERS for it to tune \
(roughly 2-6). More out-of-sample-robust strategies score higher.

Hard requirements (violating any makes the candidate invalid and discarded):
- Define BOTH param_space() and strategy(data, tools, p) with those exact signatures.
- param_space() returns a dict in the format above; every knob strategy() reads from p must
  be declared there.
- Call indicators only via the `tools` module (e.g. tools.rsi(close, p["rsi_n"])).
- Return a pandas Series aligned to data.index in [-1, 1]; end with tools.clip_signal(...).
- Causal only: no .iloc[future], no shifting the final signal (the backtester lags it).
- Pure functions: no imports, no I/O, no randomness, no global state.

Available tools:
{_TOOL_DOC}

Respond with ONLY a single ```python code block containing both functions. No prose."""


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
