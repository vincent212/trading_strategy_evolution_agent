"""
Builds the mutation prompt: show the LLM the whole scored history of strategies tried (each a
trainable param_space() + strategy()) and ask for an improved, distinct child that OUTPERFORMS
buy-and-hold — merely holding the asset scores 0.
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

YOUR GOAL is to EVOLVE strategies that OUTPERFORM BUY-AND-HOLD. Each strategy is scored by how \
much it beats buy-and-hold, risk-adjusted, out-of-sample: the median across cross-validation \
splits of the Sharpe of its ACTIVE return (strategy minus a constant fully-long position). \
CRUCIAL: simply being long the stock scores ZERO — you are NOT rewarded for exposure or for the \
stock going up. You earn ONLY by TIMING your exposure better than passively holding: be more \
invested before good stretches and less before bad ones (step aside or short in drawdowns, lean \
in during favorable regimes). A strategy that is always long, or long whenever a trend is up, \
scores ~0. Parameters are fit automatically, so you design only the FORM. You are shown the \
strategies tried so far with their scores; produce a CHILD that BEATS the best of them -- keep \
the mechanisms that scored well, drop the ones that scored poorly, and recombine them in a NEW \
nonlinear way. The child must generalize (score well on data it was NOT fit to), not just fit the \
past. The edge comes from HOW you TIME exposure by combining the tools (see rule 2).

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


DEFAULT_THEME = """INVESTMENT THEME — design your strategies in this spirit (this is HOW good \
single-stock strategies look; build realistic, tradeable structures like these, not exotic ones):
- The base case for a single stock is being LONG — it drifts up over time, so most of the time you \
should be invested. Your edge is TIMING that exposure around the base case, not avoiding it.
- ADD / lean in on DIPS, especially when volatility is LOW — calm pullbacks inside an uptrend tend \
to resolve upward, so a low-vol dip is a good place to be MORE exposed.
- REDUCE or hedge exposure when volatility is HIGH or a downtrend / drawdown regime sets in. \
Cutting risk before bad stretches is the main way a long-biased strategy BEATS buy-and-hold.
- Only go fully FLAT or SHORT on strong evidence (a confirmed high-vol / bearish regime) — time out \
of the market costs you the drift, so be out only when it really matters.
- If LEVERAGE is available (positions may exceed 1.0), concentrate it: lever UP toward the max only \
in your highest-conviction calm uptrends, and cut hard in stress. Beating buy-and-hold on total \
RETURN means amplifying the good regimes, not being uniformly levered.
Because the score is OUTPERFORMANCE of buy-and-hold, you are paid for the risk you CUT in bad \
regimes and the extra (or levered) exposure you ADD at good entries — never for merely being long."""


def load_theme(path: str | None = None) -> str:
    """Load the investment-theme paragraph injected into the system prompt. Reads `path`, else
    $STRATEGY_THEME_FILE, else a `theme.txt` next to this module, else the built-in DEFAULT_THEME.
    Editing that file swaps the theme with no code change."""
    import os
    p = path or os.environ.get("STRATEGY_THEME_FILE") \
        or os.path.join(os.path.dirname(__file__), "theme.txt")
    try:
        with open(p) as f:
            txt = f.read().strip()
            return txt or DEFAULT_THEME
    except OSError:
        return DEFAULT_THEME


def system_prompt(theme: str | None = None, max_leverage: float = 1.0) -> str:
    """The SYSTEM message with the investment theme (and a leverage note when max_leverage>1)
    injected before the RULES. theme=None -> the default/loaded theme; theme='' -> no theme."""
    theme = DEFAULT_THEME if theme is None else theme
    block = (theme.strip() + "\n\n") if theme and theme.strip() else ""
    if max_leverage and max_leverage > 1.0:
        block += (
            f"LEVERAGE IS AVAILABLE — positions may exceed fully-long: clip_signal allows up to "
            f"±{max_leverage:g}. Buy-and-hold is exactly 1.0, so the ONLY way to beat it on RETURN "
            f"is to be MORE than 1.0 long (up to {max_leverage:g}) in the best regimes and LESS "
            f"(trim, cash, or short) in the worst. Return a sig >1.0 to lever up — e.g. "
            f"`sig = 1.0 + boost*calm_uptrend - cut*high_vol` so exposure rises toward {max_leverage:g} "
            f"in calm uptrends and falls in stress. Leverage adds volatility (lower Sharpe) but that "
            f"is the accepted trade for beating buy-and-hold on total return.\n\n")
    return SYSTEM.replace("RULES — every one is mandatory",
                          block + "RULES — every one is mandatory", 1)


def _norm(code: str) -> str:
    return "\n".join(ln.rstrip() for ln in code.strip().splitlines() if ln.strip())


def build_user_prompt(history: list[dict], explore: bool = False, rejects=None) -> str:
    """history: EVERY surviving strategy, each {code, score, diagnostics}, best first. rejects: the
    last few THROWN-OUT candidates, each {reason, code} — shown as negative examples so the model
    learns what fails (invalid code, or beat buy-and-hold but no real timing skill) instead of
    re-proposing it and getting stuck.

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
    parts = ["HISTORY — every distinct strategy tried so far, best first. `score` is OUTPERFORMANCE "
             "of buy-and-hold (0 = no better than just holding; higher is better) — that is what you "
             "maximize. The other numbers show the RISK behind the score so you don't chase return "
             "with reckless leverage: Sharpe = risk-adjusted excess (info ratio), MAR = return per "
             "unit of worst drawdown (HIGHER is safer; leverage inflates return AND drawdown so a "
             "high score with low MAR is fragile), maxDD = worst peak-to-trough, avg_lev = average "
             "exposure (1.0 = fully long). Prefer structures with high score AND high MAR:"]
    for p in shown:
        d = p.get("diagnostics", {})
        parts.append(
            f"# score={d.get('median_oos', float('nan')):+.3f}  "
            f"excess_ret={d.get('median_return', float('nan')):+.0%}  "
            f"Sharpe={d.get('median_sharpe', float('nan')):+.2f}  "
            f"MAR={d.get('mar', float('nan')):.2f}  "
            f"maxDD=-{d.get('maxdd', float('nan')):.0%}  "
            f"avg_lev={d.get('avg_exposure', float('nan')):.2f}x  "
            f"pos_splits={d.get('frac_positive', float('nan')):.0%}\n{p['code'].strip()}")
    if len(rows) > MAX:
        parts.append(f"# (+{len(rows) - MAX} more tried, scoring in between — not shown)")
    best = rows[0].get("score", float("nan")) if rows else float("nan")
    joined = "\n\n".join(parts)
    if rejects:
        rparts = ["REJECTED — these were tried and THROWN OUT for the stated reason; they are NOT in "
                  "the history above and CANNOT be used. Study them to learn what a BAD strategy "
                  "looks like, and do NOT propose these or minor variants of them:"]
        for r in rejects[-6:]:
            rparts.append(f"# REJECTED: {r.get('reason', '').strip()}\n{r.get('code', '').strip()}")
        joined = joined + "\n\n" + "\n\n".join(rparts)
    if explore:
        return (f"{joined}\n\n"
                f"EXPLORATION TURN — ignore the scores above. Take a RANDOM JUMP: propose a "
                f"strategy whose STRUCTURE looks like NOTHING in the history — different tools and "
                f"a different combination than any listed (e.g. breakout gated by momentum, an "
                f"oscillator cross switched by vol regime, a rank-based signal). A wild, untried "
                f"idea is the whole point, even if it scores worse. Do NOT copy any structure "
                f"above. Return only the ```python code block.")
    return (f"{joined}\n\n"
            f"The best so far is {best:+.3f} (outperformance of buy-and-hold; 0 = no better than "
            f"holding). Beat it by TIMING exposure better — change the STRUCTURE (a nonlinear regime "
            f"switch or threshold gate per rule 2) so you are OUT or SHORT during bad stretches and "
            f"invested during good ones. Remember: always-long, or long-whenever-a-trend-is-up, "
            f"scores ~0 — the score only rewards being right about WHEN. And prefer a HIGH MAR: a big "
            f"score bought with reckless leverage (huge maxDD, low MAR) is fragile — concentrate any "
            f"leverage in the calmest, highest-conviction regimes and cut it in stress. Do not repeat "
            f"a structure already listed. Return only the ```python code block.")
