# Teach Your LLM to Trade Any Stock

### An LLM proposes the strategies; a framework enforces the discipline of a quant.

*The system searches for a trading strategy on a given Yahoo Finance ticker. Producing a strategy that fits past data is straightforward; establishing that the result is not an artifact of overfitting is the difficult part, and is the focus of the design.*

Asked directly for a profitable trading strategy, a language model returns something plausible that does not hold up out of sample. The cause is not limited capability: the model has extensive exposure to text about markets but no grounded model of how markets behave. Without structure, it produces plausible strategies with no reliable edge.

## The approach: build the quant, not the tip

The remedy is not a larger model but a **framework** — constraints on what the model may generate, together with the working method of a quantitative researcher expressed as code. That method is explicit: state a hypothesis, fit it to historical data, evaluate it on data withheld from fitting, test it against a null model of pure chance, and reserve a final period of data that is not used until the end.

In this system the model performs one function — proposing and recombining candidate strategies — and the framework performs the rest: fitting, cross-validation, and the statistical checks that decide whether an apparent edge is real. The model supplies search; the framework supplies rigor.

The sections below describe what the system does.

## Search loop

![The evolutionary loop: a population of strategies, the language model producing a new candidate from two high-scoring parents, and a fit-and-score step that retains or discards it.](assets/01_loop.png)

The system maintains a population of candidate strategies and evolves it. Higher-scoring strategies are selected as parents; the model is asked to combine and modify them into a new candidate; the candidate is fitted and scored; if its score is competitive it enters the population. Repeated over many rounds, this drives the population toward strategies that score well under the evaluation described below.

## Data

Price data comes from Yahoo Finance via the `yfinance` library; the named ticker is the traded instrument. Prices are split- and dividend-adjusted and cached after the first download. The VIX is retrieved alongside and exposed to strategies as an optional regime input.

Strategies are **trainable**: each declares tunable parameters and a parametric rule, and the parameters are fit to data before the strategy is scored. The seed strategy:

```python
def param_space():
    # the tunable parameters and their ranges
    return {"fast": ("int", 5, 60), "slow": ("int", 60, 250),
            "rsi_n": ("int", 5, 30), "tilt": ("float", -1.0, 1.0)}

def strategy(data, tools, p):
    close = data["close"]
    trend = tools.sma(close, p["fast"]) > tools.sma(close, p["slow"])
    dip   = tools.rsi(close, p["rsi_n"]) < 30
    signal = trend.astype(float) + p["tilt"] * dip.astype(float)
    return tools.clip_signal(signal)
```

A mutation rewrites both the parametric rule and its parameter declarations, so every candidate is itself a trainable model. Strategies may use only a fixed library of causal indicators (moving averages, momentum, RSI, realized volatility, breakout levels, VIX regime); they cannot import code or reference future data. Parameter fitting is performed by a numerical optimizer (SciPy differential evolution), not by the model.

## Islands

A single population converges prematurely: an early front-runner is selected repeatedly and its variants dominate within tens of rounds. To preserve diversity, the population is divided into independent sub-populations — **islands** — that evolve separately and do not exchange members. This is the standard island model from evolutionary computation, also used by DeepMind's FunSearch; the number of islands is a parameter, set to four by default.

The lifecycle:

- **Initialization.** Every island begins with the same seed strategy and diverges through independent mutation.
- **Evolution.** Each round selects one island, samples two high-scoring parents from it, and returns the resulting child to that island.
- **Reset.** Every 50 rounds the islands are ranked by their best member. The lower-scoring half — two of the four — are cleared, and each is reinitialized with a single copy of the best strategy found across all islands. Evolution then continues from that strategy.

Half of the search thus exploits the current best strategy while the other half continues to explore from a strong initialization. Reset islands are reseeded from the search's own best result, not from any external source.

## Evaluation: controlling for overfitting

A strategy that fits the past well may have no predictive value. The evaluation is designed to separate the two.

![The evaluation pipeline: fit parameters on 75% of quarters, test on the withheld 25%, repeat 100 times and take the median out-of-sample Sharpe, compare against the null-max noise bar, then measure once on the held-out 2026.](assets/04_gauntlet.png)

**Quarter-level cross-validation.** The pre-2026 history is partitioned into calendar quarters. For each evaluation, 75% of the quarters are selected at random and used to fit the strategy's parameters (maximizing Sharpe on those quarters); the remaining 25% are used to measure out-of-sample Sharpe. This is repeated over 100 random quarter assignments, and the strategy's score is the **median out-of-sample Sharpe** across them. Quarters are kept intact and are not shuffled; only their assignment to the training or test set changes between splits. A strategy that generalizes to quarters withheld from fitting scores well; one that does not is penalized.

**Null-max bar.** A high median out-of-sample Sharpe may still arise by chance given the number of candidate forms searched. To bound this, the same fit-and-cross-validate procedure is applied to sign-flipped returns — a series with the same marginal distribution but no genuine temporal structure. The best median out-of-sample Sharpe obtainable on that noise defines the null-max bar. A candidate is retained only if its median out-of-sample Sharpe exceeds the bar by a set margin. The bar scales with the flexibility of the search rather than with a raw count of trials.

**Held-out year.** All of the above uses pre-2026 data only. The current year to date (2026) is excluded from both the search and the null-max bar; the selected strategy is evaluated on it once, as a final out-of-sample measurement.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

python run.py --ticker NVDA --start 2017-01-01 --iterations 300
```

`--ticker` accepts any Yahoo Finance symbol. A 300-mutation run issues roughly 300 model
calls; the dominant cost is local CPU, since each candidate is fit and cross-validated
across 100 quarter splits. The run prints a report and writes the full result to
`runs/run_<TICKER>_*.json`. Data begins in 2017; the current year (2026 to date) is held
out as the final measurement.

The search is also exposed as a Claude Code subagent (`.claude/agents/funsearch-alpha.md`),
which launches a run, reads the result file, and reports the outcome.

## Repository

| file | role |
|---|---|
| `alpha_tools.py` | fixed, causal indicator library available to strategies |
| `data.py` | Yahoo Finance data (any ticker) plus VIX, cached to disk |
| `params.py` | encodes a strategy's `param_space()` into the optimizer's search box |
| `strategy_seed.py` | the trainable `param_space()` + `strategy(data, tools, p)` contract and seed |
| `prompt.py` | the mutation prompt: two parents to one child |
| `backtest.py` | backtest, differential-evolution fit, quarter-CCV median-OOS score, null-max bar |
| `evolve.py` | islands, parent sampling, model mutation call, evaluation |
| `run.py` | orchestration: `run_search(...)` and CLI |
| `make_diagrams.py` | regenerates the figures in `assets/` |
| `.claude/agents/funsearch-alpha.md` | Claude Code subagent wrapper |

## Notes and limitations

- A passing result indicates a strategy worth further out-of-sample evaluation, not a recommendation to trade. This is not investment advice.
- Transaction costs are a flat per-trade fee; slippage, borrowing, and financing are not modeled.
- Fitting is CPU-intensive: each candidate is fit with differential evolution across 100 quarter splits. Split count, fit budget, and iteration count are configurable.
- Candidate code is executed locally via `exec()` within a restricted namespace, which is a constraint, not a security boundary.
- Single equities have less history and more idiosyncratic and event risk than broad indices; the out-of-sample checks matter more for them.

## License

MIT — see [`LICENSE`](LICENSE).
