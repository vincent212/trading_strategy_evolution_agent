# LATSS on cross-sectional selection — the Magnificent-7 "pick 1 of N" case

Date: 2026-09-07. A test-bed for the LATSS *loop* on a structurally different
problem than single-asset timing: **cross-sectional stock selection**. The object
under test is the loop (LLM evolves a scoring rule → DE fits it → CV selects →
null/holdout validate), not any strategy.

## Question

Give the LLM basic technical-analysis tools and the Magnificent 7. Let it hold
**one** of the 7 at a time (pick 1 of 7). Does the evolved selector beat the
**equal-weight** Mag-7 portfolio, risk-adjusted, out of sample?

## Why this is not a look-ahead / survivorship trap

- **Name-symmetric contract.** The evolved unit is a per-series scorer
  `score(series, tools, p)` that sees ONE anonymous close series at a time — no
  ticker identity, no view of the other names. It therefore *cannot* hardcode "hold
  NVDA"; it can only express a technical rule that ranks a name by its own price
  behaviour. The harness applies it to all 7 and holds the argmax each bar.
- **Survivorship cancels.** The benchmark is equal-weight of the *same* 7, so any
  tailwind from the basket having gone up hits strategy and benchmark equally. It
  is a footnote on external validity, not a source of active alpha.
- **Scored on active Sharpe, not cumulative return.** Concentrating in one name is a
  higher-variance bet; on raw return that can "beat" equal-weight with no skill. The
  fitness is CV median-OOS **active Sharpe vs equal-weight**, and a **random-picker
  null** (hold a uniformly random name each bar) is reported as the real bar — a
  strategy must beat that distribution, not just equal-weight.

## Setup

| | |
|---|---|
| Universe | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA |
| Data | yfinance daily adjusted close, 2015-01-02 → 2026-09-04 (2936 rows) |
| Pool / holdout | pool = years ≤ 2024 (2516d); **holdout = 2025+ (420d), measured once** |
| Fitness | CV median-OOS active Sharpe vs equal-weight, 16 random quarter splits |
| Params | fit per split by SciPy differential evolution (budget 60), 1-bar lag, 5bp cost |
| Nulls | selection-aware shift-the-signal skill p-value; random-picker distribution |
| Search | LATSS loop, 8 iterations, LLM = Claude Code subagent (file-handoff, no API key) |
| Code | `data_mag7.py`, `backtest_xs.py`, `run_mag7.py` |

**Tools advertised** (13, all causal, single-series): `sma ema roc zscore rsi
realized_vol rolling_high rolling_low drawdown breakout vol_regime pctile_rank
crossover`.

## Result

The LLM searched competently — pool CV active Sharpe rose from the best seed
**+0.425** to the champion **+0.644** over 8 iterations, and it correctly learned
from the leaderboard that volatility terms hurt and dropped them. **But the champion
did not generalize.**

| | Pool CV (≤2024) | Holdout 2025+ (420d) |
|---|---|---|
| **Champion** | active Sharpe **+0.644**, skill p=**0.15** | active Sharpe **−0.37** |
| Equal-weight | — | Sharpe +0.81 (+36.2%) |
| Strategy standalone | — | Sharpe +0.40 (+12.8%) |
| Random-picker null q95 | — | **+0.58** |

**Per-year holdout** (fixed params fit on ≤2024, measured once):

| | Strategy | Equal-weight | Active |
|---|---|---|---|
| 2025 (250d) | Sharpe +1.83, +48.5% | Sharpe +0.93, +26.5% | Sharpe **+0.44**, +22.0pp |
| 2026 (170d, to Sep 4) | Sharpe −1.39, −20.0% | Sharpe +0.65, +8.6% | Sharpe **−1.62**, −28.6pp |

The pooled −0.37 hides a split: the champion **beat equal-weight clearly in 2025**
then **reversed hard in 2026**.

## Champion

```python
def param_space():
    return {"n1": ("int", 60, 120), "n2": ("int", 20, 60),
            "fast": ("int", 10, 40), "slow": ("int", 40, 120), "b": ("float", 0.4, 1.0)}

def score(series, tools, p):
    m1 = tools.roc(series, p["n1"]); m2 = tools.roc(series, p["n2"])
    mom = 0.5 * (m1 + m2)                       # dual-horizon momentum
    up = tools.crossover(tools.ema(series, p["fast"]), tools.ema(series, p["slow"]))
    gate = p["b"] + (1.0 - p["b"]) * up
    return mom * gate
```

Fitted: `n1=113, n2=57, fast=29, slow=73, b=0.670`. Effectively a **multi-horizon
momentum selector** (hold the strongest blended trend). Caveat: the EMA-crossover
gate is nearly inert — `crossover()` fires only on the exact crossing bar (0
otherwise), so the gate sits at ≈0.67 almost every day; the selection is essentially
`0.5·(roc(113)+roc(57))`.

## Verdict

On real Mag-7 pick-1, the LLM finds in-sample structure but there is **no
demonstrated selection skill** over equal-weight out of sample: the champion loses
on the pooled 2025–26 holdout, sits below the random-picker null, and its in-sample
skill p-value never reaches significance (0.15). One good year (2025) followed by a
sharp 2026 reversal. This is the calibrated null the plan anticipated on real
assets — **the loop is a competent searcher; the alpha is not there in this test.**

## Follow-up runs — concentration (k) and calibration window

All runs: 8 LATSS iterations, 16 CV splits, DE budget 60, 5bp cost, Claude Code
subagent as the mutation operator.

> **⚠️ CORRECTION (measurement bug).** The original holdout measurement computed the
> champion's indicators on the *holdout slice alone*, so long-lookback selectors
> (e.g. `roc` n≈119) sat in **cash** through the warmup — on a 170-day 2026 holdout,
> up to 100% of it — which silently read as flat/negative results. Fixed in
> `run_mag7.py` (scores are now computed on the full continuous panel, then sliced).
> **Re-measured, warm-indicator holdout active Sharpe:** run 2 −0.82 (2025+26); run 3
> **+1.24** (beat EW in 2026: +32.4% vs +7.7%); run 4 −1.98; run 5 +0.06 (matched).
> The "Holdout active Sharpe" column in the table below shows the *original buggy*
> values and is superseded by these. Caveat: the 2026 holdout is 170 days / 7 names
> with a random-picker q95 ≈ +1.19, so even run 3's +1.24 is marginal and noisy.

| # | Pick k | In-sample (calibration) | Holdout | Pool CV active Sharpe (skill p) | Holdout active Sharpe | Champion the LLM found |
|---|---|---|---|---|---|---|
| 1 | 1 | 2015–2024 | 2025+2026 | +0.644 (p=0.15) | **−0.37** | multi-horizon momentum |
| 2 | 3 | 2015–2024 | 2025+2026 | +0.261 (p=0.02) | **−0.21** | momentum + z-tilt |
| 3 | 1 | 2020–2025 | 2026 only | +0.444 (p=0.73) | **+0.05** | momentum × calm-vol-regime gate |
| 4 | 3 | 2020–2025 | 2026 only | +0.278 (p=0.54) | **−0.50** | plain momentum (vol-gate NOT found) |
| 5 | 1 | 2012–2025† | 2026 only | **+1.110** (p=0.02) | **−0.65** | double-denoised momentum + low-vol tilt |
| 6 | 1 | 2015–2025 | 2026 only | +0.620 (p=0.02) | **+0.35** ✓ | momentum×vol-regime − drawdown penalty |

✓ Run 6 was measured with the **fixed** harness — its +0.35 is a correct warm-indicator
value (strategy +12.4% vs EW +7.7% in 2026), still below the random-picker q95 (+1.23).

† Requested as "since 2005", but the Mag-7 panel inner-joins on dates where all 7
names trade, and META's IPO (2012-05-18) binds — so the real calibration window
starts mid-2012, not 2005.

Per-year holdout (runs 1–2, combined window decomposed):

| | 2025 active | 2026 active |
|---|---|---|
| Run 1 (k=1) | Sharpe +0.44 (+22pp) | Sharpe −1.62 (−29pp) |
| Run 2 (k=3) | Sharpe +0.28 (+14pp) | Sharpe −1.57 (−19pp) |

### Finding 1 — over the combined 2025+2026 holdout, it does not beat equal-weight

At neither concentration does the selector beat equal-weight over 2025+2026: it
underperforms on cumulative return (k=1: +12.8% vs +36.2%; k=3: +29.4% vs +36.2%)
and its excess return over equal-weight has a negative Sharpe (−0.37, −0.21). The
2025 edge is more than given back by the 2026 drawdown. (k=3's *standalone* Sharpe
+0.89 slightly exceeds equal-weight's +0.81 — a volatility-reduction effect, not
outperformance; it still trails on return and information ratio.)

### Finding 2 — momentum-only selectors underperform in 2026; the one exception was a stochastic search outcome, not a property of the calibration window

Calibrated only through 2024 (runs 1–2), the loop settles on **naive momentum**,
which **blows up in 2026** (active Sharpe ≈ −1.6). Calibrated **through 2025** the
result depends on what the (non-deterministic) search happens to find:

- **Pick-1 (run 3)** evolved a **vol-regime-gated momentum** — `mom × (1 + w·vol_regime)`,
  tilting toward calm-vol names — which **tracked equal-weight in 2026** (active
  Sharpe +0.05) instead of collapsing.
- **Pick-3 (run 4)**, *same* window, **kept plain momentum** (none of its 8 mutations
  beat the momentum seed) and **underperformed in 2026** (active Sharpe −0.50).

So the defensive vol-gate was **one stochastic search outcome, not a reliable product
of the calibration window**. The subagent mutation operator is non-deterministic;
run-to-run it lands on different champions, and *whether it happens to find a defensive
structure* — not k, and not the calibration window per se — is what moved the 2026
result. What actually holds up across runs: **momentum-only selection underperforms in
2026**, and the single run that avoided that did so by finding a vol-regime gate it did
not rediscover on the next run. Even that best case is a tie, not a win (skill p=0.73;
and over a 170-day/7-name holdout the random-picker null is uninformative, q95 ≥ +0.80).

### Finding 3 — RETRACTED (was an artifact of the measurement bug)

The original Finding 3 claimed a clean overfitting signature: run 5 (longest
calibration, highest in-sample CV +1.11) supposedly "collapsed" out of sample
(−0.65). **That was the cash-in-warmup bug, not a real result.** Re-measured
correctly, run 5's 2026 holdout **matched** equal-weight (active Sharpe +0.06), and
run 3 (lower in-sample, +0.44) actually **beat** equal-weight in 2026 (+1.24). A mild
inverse in-sample/OOS relation may still exist across these two, but the dramatic
"more fitting → collapse" claim does not survive correct measurement and is withdrawn.

**Overall (corrected):** across pick-1 / pick-3 and every calibration window, LATSS
searches competently. On the (short, noisy) 2026 holdout the corrected picture is
mixed rather than uniformly null: **pick-1 champions matched or beat equal-weight**
(run 3 +1.24, run 5 +0.06) while **pick-3 champions lost** (run 4 −1.98, run 2 −1.91 in
2026). But the 2026 holdout is 170 days over 7 names with a random-picker q95 ≈ +1.19,
so a single champion clearing it (run 3) is marginal and not robust. The honest read
remains **no *reliable* out-of-sample edge** — the results are dominated by run-to-run
champion variance and a holdout too short to distinguish skill from luck — but the
earlier "it gets killed in 2026" narrative was largely a measurement artifact, now
fixed. Longer/walk-forward holdouts are needed to say anything firmer.

## Reproduce

```
cd /Users/vm/llm-assisted-trading-strategy-search
# pick-1, holdout 2025+2026 (runs 1):
LLM_PROVIDER=subagent .venv/bin/python -u run_mag7.py --iterations 8 --holdout-year 2025 --splits 16 --fit-budget 60
# pick-3, calibrate through 2025, holdout 2026 (run 4), isolated dir:
LLM_PROVIDER=subagent SUBAGENT_DIR=$PWD/runs_2026_k3 .venv/bin/python -u run_mag7.py \
  --iterations 8 --start 2020-01-01 --holdout-year 2026 --top-k 3 --splits 16 --fit-budget 60 --out-dir $PWD/runs_2026_k3
```
(with a Claude Code session servicing `runs/mutation_request.json`), or set
`ANTHROPIC_API_KEY` and drop `LLM_PROVIDER=subagent` to use the API directly.
Saved run: `runs/mag7_20260907_183117.json`.
