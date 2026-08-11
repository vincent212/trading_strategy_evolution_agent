# Trading Strategy Evolution Agent

An LLM proposes and mutates trading-strategy **code**; a numerical optimizer fits the parameters; a cross-validation + null-test framework decides whether any of it survives out of sample. It's a FunSearch-style evolutionary search adapted to noisy financial data — the model does structural search, not the trading.

**📄 Full write-up:** [Why LLMs Can't Trade — and How to Use Them in Trading](https://vincentmayeski.substack.com/p/why-llms-cant-trade-and-how-to-use)

![search loop](assets/01_loop.png)

## What it does

- An LLM acts as the **mutation operator**, rewriting strategy structure. It never sees market data and never picks a numeric value.
- Each candidate declares free parameters; SciPy `differential_evolution` **fits** them on the training quarters of each split.
- Fitness is the **median out-of-sample** score over random quarter cross-validation splits. Each candidate also gets a **report-only skill p-value** — a selection-aware *shift-the-signal* permutation test (does its timing beat a random re-timing of its own positions?). Selection stays fitness-driven; nothing is hard-gated. The holdout years are sealed and measured once against buy-and-hold.
- The population evolves across **islands** with periodic resets. The objective is switchable: Sharpe, or return under a Sharpe floor.
- The mutation model is pluggable: a local open model (Ollama), any OpenAI-compatible endpoint, the Anthropic API, or a Claude Code subagent.

The [article](https://vincentmayeski.substack.com/p/why-llms-cant-trade-and-how-to-use) covers the method in full, a worked NVDA example (including where it fails out of sample), and how it relates to FunSearch, AlgoEvolve, and MadEvolve.

## Quick start

Local model, no API key:

```bash
brew install ollama
ollama serve                       # leave running
ollama pull qwen2.5-coder:7b

pip install -r requirements.txt
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwen2.5-coder:7b
python run.py --ticker NVDA --start 2017-01-01 --iterations 300
```

- **Hosted model:** unset `LLM_*` and set `ANTHROPIC_API_KEY`, or point `LLM_BASE_URL` / `LLM_MODEL` at any OpenAI-compatible provider (e.g. Groq).
- **Claude Code subagent:** `LLM_PROVIDER=subagent`.
- **Maximize return under a Sharpe floor:** add `--objective return --min-sharpe 0.8`.

Progress streams to `runs/<TICKER>_progress.log`. The champion code, its median OOS fitness, its shift-the-signal skill p-value, and the sealed-holdout numbers vs buy-and-hold land in `runs/run_<TICKER>_*.json`.

## Layout

| file | role |
|---|---|
| `alpha_tools.py` | fixed, causal indicator library available to strategies |
| `strategy_seed.py` | the trainable `param_space()` + `strategy(data, tools, p)` contract and seeds |
| `prompt.py` | the mutation prompt (whole scored history → one child) |
| `backtest.py` | backtest, differential-evolution fit, quarter-CV median-OOS fitness, shift-the-signal skill test |
| `evolve.py` | islands, sampling, mutation call, evaluation, repair/self-correct |
| `run.py` | orchestration: `run_search(...)` and CLI |
| `llm.py` | provider shim: Ollama / OpenAI-compatible / Anthropic / Claude Code subagent |
| `data.py` | Yahoo Finance price data (any ticker), cached to disk |
| `params.py` | encodes `param_space()` into the optimizer's search box |
| `make_diagrams.py`, `make_tables.py` | regenerate the figures in `assets/` |

## Notes

Not investment advice. This is a research prototype: transaction costs are a flat per-trade fee, the evaluation is a single held-out window, and the worked example does **not** beat buy-and-hold out of sample. See the article's limitations and future-work sections.

## License

MIT — see [`LICENSE`](LICENSE).
