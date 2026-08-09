"""
Builds the mutation prompt: show the LLM two high-scoring parent strategies and
ask it to write an improved child. This is the FunSearch mutation operator.
"""
from __future__ import annotations
import alpha_tools

_TOOL_DOC = """\
sma(x, n) ema(x, n) roc(x, n) zscore(x, n)         -- trend / momentum
rsi(x, n=14)                                        -- oscillator, 0..100
realized_vol(x, n=20) vol_target_scale(x, target_vol, n, max_leverage)
rolling_high(x, n) rolling_low(x, n) breakout(x, n) -- breakout / range
vix_regime(vix, low=15, high=25) pctile_rank(x, n)  -- regime
crossover(fast, slow) clip_signal(sig)              -- helpers"""

SYSTEM = f"""You are a quantitative researcher evolving trading strategies for a single \
stock or ETF (the traded asset is data['close']; data['vix'] is the market VIX regime, \
which may be NaN for some names — handle that gracefully).

You will be shown one or more parent `strategy` functions with their scores. Write a \
NEW, improved `strategy` function that is meaningfully different from the parents — \
combine their ideas, adjust parameters, or introduce a new mechanism from the tools.

Hard requirements (violating any makes the strategy invalid and discarded):
- Exact signature: def strategy(data, tools):
- `data` is a DataFrame with columns 'close' (the traded asset) and 'vix' (may be NaN).
- Call indicators only via the `tools` module (e.g. tools.rsi(close, 14)).
- Return a pandas Series aligned to data.index with values in [-1, 1].
- End with `return tools.clip_signal(...)`.
- Causal only: no .iloc[future], no shifting the final signal (the backtester lags it).
- Pure function: no imports, no I/O, no randomness, no global state.

Available tools:
{_TOOL_DOC}

The score is the median Sharpe across time blocks minus a turnover penalty; higher is \
better. Strategies that never trade, or overfit one window, score poorly.

Respond with ONLY a single ```python code block containing the function. No prose."""


def build_user_prompt(parents: list[dict]) -> str:
    """parents: list of {code, score, diagnostics} dicts, best first."""
    blocks = []
    for i, p in enumerate(parents):
        diag = p.get("diagnostics", {})
        blocks.append(
            f"# Parent v{i}  (score={p['score']:.3f}, "
            f"full_sharpe={diag.get('full_sharpe', float('nan')):.2f}, "
            f"block_sharpes={diag.get('block_sharpes', [])})\n"
            f"{p['code'].strip()}"
        )
    joined = "\n\n".join(blocks)
    return (f"{joined}\n\n"
            f"Write strategy v{len(parents)}: a better, distinct strategy. "
            f"Return only the ```python code block.")
