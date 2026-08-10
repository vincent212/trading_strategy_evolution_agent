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
vol_regime(x, n=20, low=-0.5, high=0.5)              -- vol regime by z-score (low/high are z cutoffs): +1 low / -1 high
pctile_rank(x, n)                                    -- rolling percentile rank, 0..1
crossover(fast, slow) clip_signal(sig)               -- helpers"""

SYSTEM = f"""You are a quantitative researcher evolving TRAINABLE trading strategies for a \
single stock or ETF. The traded asset is data['close'].

YOUR GOAL is to EVOLVE strategies that score higher. Each strategy is scored by its \
median out-of-sample Sharpe across many cross-validation splits (higher is better); its \
parameters are fit automatically before scoring, so you design only the FORM. You are shown \
the strategies tried so far with their scores; produce a CHILD that BEATS the best of them -- \
keep the mechanisms that scored well, drop the ones that scored poorly, and recombine them in \
a NEW nonlinear way. The child must be a genuine variation (not a copy of one already tried) \
and should generalize -- score well on data it was NOT fit to -- not merely fit the past. The \
edge comes from HOW you combine the tools (see rule 2), not from any single indicator.

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
1. Build ALL indicators by calling tools.* ONLY (reading windows from p, e.g.
   tools.rsi(close, p["rsi_n"])). Do NOT compute your own with .rolling/.ewm/.apply/np.* or
   import anything. For COMBINING you MAY use comparisons, .astype(float), arithmetic, np.where.
2. Do NOT just add signals together. A flat sum like `trend + w*dip + w2*vol` has PLATEAUED —
   it will not beat the parents. The edge is combining NONLINEARLY, with THRESHOLDS and REGIME
   SWITCHES. Concretely:
     - GATE / SWITCH on the trailing-vol REGIME (scale-free, works on any ticker):
         calm = (tools.vol_regime(close, p["vol_n"], p["vol_lo"], p["vol_hi"]) > 0).astype(float)
         sig  = calm * trend + (1.0 - calm) * meanrev      # trend when calm, mean-revert when wild
     - per-BAR switch with np.where:
         sig = np.where(tools.zscore(close, p["z_n"]) < 0, longs, shorts)
     - threshold an oscillator to 0/1:   (tools.rsi(close, p["n"]) < p["lvl"]).astype(float)
     - product to GATE one signal by another:   sig = trend * tools.vol_regime(close, p["vol_n"])
     - if/else on a PARAMETER the optimizer picks:
         if p["mode"] == "trend":  sig = a
         else:                     sig = b
   NOTE: `if series > x:` is NOT valid on a Series — use (series > x).astype(float) or
   np.where(series > x, a, b).
   Every threshold, cutoff, weight, and switch is a PARAMETER you declare in param_space();
   the OPTIMIZER fits its value. You choose only that it exists and its range.
3. Declare 2-6 knobs in param_space(). EVERY p["..."] read in strategy() MUST be declared —
   an undeclared p["..."] (or a bare tool name without the tools. prefix) crashes and is discarded.
4. The LAST line of strategy() must be exactly:  return tools.clip_signal(sig)
5. Pure function: no imports, no I/O, no randomness, no loops over rows/dates, no .iloc.

tools you may call (as tools.NAME(...)):
{_TOOL_DOC}

VALID EXAMPLE — note the REGIME SWITCH (not a flat sum). Copy this shape, then vary the ideas/knobs:

```python
def param_space():
    return {{"fast": ("int", 5, 50), "slow": ("int", 60, 200), "z_n": ("int", 5, 40),
             "vol_n": ("int", 10, 60), "vol_lo": ("float", -2.0, 0.0), "vol_hi": ("float", 0.0, 2.0)}}

def strategy(data, tools, p):
    close = data["close"]
    trend   = (tools.sma(close, p["fast"]) > tools.sma(close, p["slow"])).astype(float)
    z       = tools.zscore(close, p["z_n"])
    meanrev = (z < 0).astype(float) - (z > 0).astype(float)     # buy dips, sell rips
    calm    = (tools.vol_regime(close, p["vol_n"], p["vol_lo"], p["vol_hi"]) > 0).astype(float)
    sig     = calm * trend + (1.0 - calm) * meanrev             # SWITCH: trend when calm, mean-revert when wild
    return tools.clip_signal(sig)
```

Now write the CHILD — a new param_space() + strategy() that OUT-SCORES the best strategy in the
history below (higher median out-of-sample Sharpe). Study which STRUCTURES scored high vs low,
keep what worked, and make a real structural change (a regime switch or threshold gate) — do not
repeat a structure already listed. Output ONLY one ```python code block with BOTH functions. No prose."""


def _norm(code: str) -> str:
    return "\n".join(ln.rstrip() for ln in code.strip().splitlines() if ln.strip())


def build_user_prompt(history: list[dict], explore: bool = False) -> str:
    """history: EVERY strategy tried so far, each {code, score, diagnostics}. We show the whole
    landscape (best first, deduplicated) with each one's median OOS Sharpe, so the model can see
    which STRUCTURES worked and which did not — and avoid re-proposing ones already tried.

    explore=True turns this into an EXPLORATION turn: ignore the scores, take a random jump to a
    structure unlike anything tried (to escape local optima)."""
    seen, rows = set(), []
    for p in sorted(history, key=lambda x: x.get("score", float("-inf")), reverse=True):
        k = _norm(p.get("code", ""))
        if not k or k in seen:
            continue
        seen.add(k)
        rows.append(p)
    MAX = 24                                       # bound context: top 16 + worst 8
    shown = rows if len(rows) <= MAX else rows[:16] + rows[-8:]
    parts = ["HISTORY — every distinct strategy tried so far and its median OOS Sharpe "
             "(higher is better), best first. Learn which STRUCTURES win:"]
    for p in shown:
        d = p.get("diagnostics", {})
        parts.append(f"# median_oos={d.get('median_oos', float('nan')):+.3f}  "
                     f"positive_splits={d.get('frac_positive', float('nan')):.0%}  "
                     f"consistency_std={d.get('std_oos', float('nan')):.2f}\n{p['code'].strip()}")
    if len(rows) > MAX:
        parts.append(f"# (+{len(rows) - MAX} more tried, scoring in between — not shown)")
    best = rows[0].get("score", float("nan")) if rows else float("nan")
    joined = "\n\n".join(parts)
    if explore:
        return (f"{joined}\n\n"
                f"EXPLORATION TURN — ignore the scores above. Take a RANDOM JUMP: propose a "
                f"strategy whose STRUCTURE looks like NOTHING in the history — different tools and "
                f"a different combination than any listed (e.g. breakout gated by momentum, an "
                f"oscillator cross switched by vol regime, a rank-based signal). A wild, untried "
                f"idea is the whole point, even if it scores worse. Do NOT copy any structure "
                f"above. Return only the ```python code block.")
    return (f"{joined}\n\n"
            f"The best so far is {best:+.3f}. The top entries are mostly flat linear blends that "
            f"have plateaued — to beat them, change the STRUCTURE (add a nonlinear regime switch "
            f"or threshold gate per rule 2), and do not repeat one already listed. Return only the "
            f"```python code block.")
