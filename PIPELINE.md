# How the pipeline actually works

This documents the real behaviour of the code as it stands, not an idealized version.
Read it top to bottom; each section names the file and function that implements it.

---

## 0. One-paragraph summary

An evolutionary loop uses an LLM as a mutation operator to evolve the *structure* of a
trainable trading strategy for one ticker. Each structure declares free parameters; a numerical
optimizer fits those parameters. Structures are ranked by their **cross-validated out-of-sample
median score** (the fitness). A separate, **report-only** statistical test — a selection-aware
"shift-the-signal" permutation test — is computed for every candidate and headlined on the
champion; it answers "is the timing real?" and does **not** influence selection. The final
champion is fitted on all in-sample data and measured **once** on a sealed holdout period, against
buy-and-hold. Statistical significance (the skill test) and economic significance (beating
buy-and-hold) are deliberately kept as two separate questions.

---

## 1. Data split — `run.py: run_search`

```
full  = prices(ticker, 2017-01-01 .. today)          # data.py: get_data (Yahoo, cached)
pool  = full[ year <  holdout_year ]                 # everything the search may touch
hold  = full[ year >= holdout_year ]                 # SEALED — measured once, at the very end
```

- `holdout_year` defaults to 2026. The NVDA experiments used `--holdout-year 2025`, so **2025 and
  2026 were both sealed** and the pool was 2017–2024 (2012 rows).
- A hard `assert` guarantees no holdout row is in `pool`. Only `pool` reaches the search, fit, CV,
  and skill test. `full` is used solely for the final holdout measurement.

## 2. The score a everything maximizes — `backtest.py: _score_arr`

```
objective = "sharpe":  annualized Sharpe of the net-return stream
objective = "return":  annualized_return  −  10 · max(0, min_sharpe − Sharpe)   # soft Sharpe floor
```

The `10·` penalty is a **soft** floor (default `min_sharpe = 0.8`): return-maximizing strategies
are pushed to keep Sharpe up, but a strategy can still dip below the floor and pay a penalty rather
than being hard-rejected. (Caveat: the `10×` mixes return units with Sharpe units; at that
coefficient it behaves close to a hard floor. Not yet re-tuned.)

## 3. Backtest — `backtest.py: run_backtest`

```
pos = clip(signal, −1, 1).shift(1)          # decide at t−1, hold at t  → no look-ahead
net = pos · pct_change(close) − cost · |turnover(pos)|
```

`cost` defaults to 0.0005 per unit turnover. The 1-bar lag is the only thing preventing
look-ahead; every tool in `alpha_tools.py` is causal.

## 4. Fitness = cross-validated median OOS — `backtest.py: ccv_median_oos`

This is the number the search selects on.

```
splits = 100 quarter-based CV splits of pool          # make_quarter_splits (train 75% / test 25% of quarters)
for each split (train, test):
    p   = argmax_p  _score_arr( backtest(structure(pool, p))[train] )   # FIT on train — fit_on_mask
    s_i = _score_arr( backtest(structure(pool, p))[test] )              # score OOS on test
fitness = median_i(s_i)                                                 # diag["median_oos"]
```

- `fit_on_mask` fits parameters with SciPy `differential_evolution` (continuous search), budget
  `fit_budget` (default 200 function evals — `maxiter` is derived so total evals ≈ `fit_budget`).
  This is the expensive part: ~100 fits × ~200 backtests each ≈ **2×10⁴** backtests per candidate.
- Reported alongside: OOS Sharpe, OOS annualized return, fraction of positive splits.

## 5. The evolution loop — `evolve.py: Evolver.run / step`

```
seed 4 islands with the seed families                # strategy_seed.py: SEEDS
for it in 1 .. iterations:
    explore = (it % 5 == 0)                           # every 5th iteration = random-jump turn
    parents = whole scored history, best-first        # prompt.py: build_user_prompt
    code    = LLM.mutate(system, parents)             # llm.py: Anthropic | OpenAI-compatible | subagent
    child   = evaluate(code)                           # see §6
    if child is None and it was a CODE error:
        retry up to 2× by sending the error back to the LLM   # _self_correct
    if child survived: add to its island; update global best
    every reset_every iterations: reset the weak islands
champion = highest-fitness program ever seen
```

The LLM is only a **mutation operator**. It never sees prices; it sees the scored history of
structures and proposes a new structure. Providers (`llm.py`): the Anthropic API, any
OpenAI-compatible endpoint (`LLM_BASE_URL`, e.g. local Ollama), or a **Claude Code subagent** over
a filesystem handoff (`LLM_PROVIDER=subagent`) — the NVDA runs used the subagent path, with a
background agent generating each mutation.

## 6. Evaluating one candidate — `evolve.py: Evolver.evaluate`

In order:
1. **Repair** — strip forbidden imports, prefix bare tool calls (`_repair`).
2. **Dedup** — exact normalized-code cache, plus a behavioural cache keyed on the position series at
   midpoint params (skipped when that signal is ~constant, to avoid false-merging distinct
   structures). Duplicates return the cached program and are not re-scored.
3. **Compile + validity** — run once at midpoint params; must return a `pandas.Series`.
4. **Augment space** — any `p["x"]` used but not declared is given an inferred range (`_augment_space`).
5. **Fitness** — `ccv_median_oos` (§4). This is what selection uses.
6. **Skill p-value (REPORT-ONLY)** — the shift-the-signal test (§7). Attached to the program's
   diagnostics and logged; **does not gate selection**.

Nothing here hard-rejects a candidate on statistical grounds. The only rejections are malformed
code, non-`Series` output, non-finite fitness, and duplicates.

## 7. The skill test — selection-aware shift-the-signal — `backtest.py: shift_null_pvalue`

The question: *could random re-timing of this structure's own positions, under a fixed uniform
search over P configs, have produced its in-sample score?* If yes, the apparent edge is not timing
skill.

```
configs = sample P configs uniformly from the structure's param space     # sample_configs
for each config: position series on the REAL pool                          # clip to [−1,1]
real  = max over configs of  _score_position(pos)                          # the selection you do
for each fixed random circular shift k (common random numbers):
    null_k = max over configs of  _score_position( roll(pos, k) )          # same P, re-timed
b = #{ null_k >= real };  p = (b + 1) / (m + 1)     # NOT b/m — a permutation p is never 0
```

Why this null, and not the earlier ones:

- **Shift preserves the return path and each config's exposure profile exactly** (average exposure,
  on/off pattern, autocorrelation) and destroys **only** the alignment between signal and return —
  the one thing fitting can fake. So **drift and exposure level cancel**; only timing skill is
  tested. (The earlier sign-flip null destroyed the drift and just tested "are you net long?"; the
  block bootstrap over-corrected and was poorly calibrated. Both are abandoned. The dead
  `build_null_closes` / `null_max_bar_ccv` helpers remain in `backtest.py` but are unused.)
- **Selection-aware, but only w.r.t. uniform sampling**: `real` and every null draw take the max
  over the *same* P uniformly-sampled configs, so the "I tried many settings and kept the best"
  inflation appears on both sides and cancels. Caveat: the *fitness* (§4) selects with
  `differential_evolution`, which searches materially harder than a uniform max over P — on a
  no-skill surface DE extracts ~0.19 more Sharpe from noise at 6 knobs than max-of-256 uniform. So
  this test is an internally-valid test of *the structure under a fixed uniform search*; it does
  **not** fully price in the DE fit that produced the champion, and would under-correct a
  DE-overfit champion. (Diagnostic: compare the test's `real` to the recorded DE fit score; if they
  diverge, P is too small.)
- **Shift offsets exclude near-identity shifts** (`make_shift_offsets`, drawn from
  `[min_gap, n − min_gap]`, `min_gap = 250 > longest lookback`). A shift of a few bars barely moves a
  slow position, so it is not a genuine null draw and inflates p. This matters: excluding them moves
  seed[0] on the NVDA pool from p = 0.16 to p = **0.04**.
- **Buy-and-hold lands exactly on the bar** (a constant position is unchanged by a shift, so its
  real score equals every null score → p = 1.0). That is the correct zero point: the test is
  **indifferent** to pure exposure, and measures exposure *management*, blind to exposure *level*.

Calibration (real NVDA pool, verified, with the near-identity fix + `(b+1)/(m+1)`): buy-and-hold
p = 1.00 (lands on the bar); the four seeds p ≈ **0.04 / 0.47 / 0.40 / 0.24** — seed[0] is
borderline at 0.04, but that is 1 of 4 tests with no multiplicity correction, so it is weak
evidence, not a finding; a look-ahead perfect-timing position p = the floor `1/(m+1)` (it maxes out
the statistic; the estimator never reports 0).

Cost: cheap — P strategy evaluations + P×(shifts) array scorings (no re-fitting). Runs on every
candidate. Shift offsets are generated once at run start and reused for every candidate (common
random numbers), so the bar is a consistent comparison across candidates. Per-candidate m is small
(50 → floor ~0.02); the champion recompute uses m ≥ 300 (floor ~0.003).

**This is report-only.** `n_skill_significant` counts candidates with p < 0.05; the per-candidate
p is logged as `skill_p=`; selection remains purely by fitness (§4).

## 8. Champion finalization — `run.py: run_search`

1. **High-resolution skill p-value** — `shift_null_pvalue` again on the champion with more configs
   and shifts (≥256 × ≥300). This is the headline **statistical** significance.
2. **Fit on all of pool** — `fit_full` (§4 fit over the whole pool, larger budget).
3. **Holdout measurement (once)** — backtest on `full`, then for **each** sealed year report the
   strategy's Sharpe and total return **against buy-and-hold** for that year (`holdout_by_year`),
   plus the combined holdout. This is the **economic** significance check.

The two verdicts are separate by design:
- **Skill test** (in-sample, during search): is the timing real? Says nothing about whether the
  strategy is worth owning.
- **Buy-and-hold** (out-of-sample, sealed holdout): is it worth doing instead of the obvious
  passive alternative? Says nothing about whether the edge is real.
Neither implies the other. The misleading combination — an in-sample *selected* champion compared
to buy-and-hold with no selection correction — is exactly what the pipeline avoids by putting the
selection-aware test in-sample and the benchmark out-of-sample.

## 9. What the NVDA experiments found

Under the corrected skill test, the evolved structures' champion p-values sit well above 0.05, so
the test **fails to detect** timing skill on NVDA — and on the sealed holdout the champion does not
beat buy-and-hold. State it as "not detected," **not** "no skill exists": the test's power against
the *exposure-management* class (cut risk in genuinely worse windows) is low — synthetic checks put
it around ~16% at a realistic effect size, versus ~100% against directional timing. That is exactly
the mechanism the champion is described as using (risk control, not return), so a high p-value here
is consistent with both "no skill" and "real but modest skill this sample cannot resolve." The
defensible reading: the search finds structures that fit the in-sample data; their timing is not
shown to beat random re-timing of their own exposure; and buy-and-hold — a strong benchmark for a
single high-drift name — is not beaten out of sample. (Power figure caveat: it depends on the
injected effect size; the qualitative gap vs directional timing is robust, the exact 16% is not.)

## 10. Configuration & files

| file | role |
|---|---|
| `data.py` | Yahoo price data (any ticker), cached |
| `alpha_tools.py` | fixed causal indicator library the strategies may call |
| `params.py` | encode/sample/decode a `param_space()` for the optimizer |
| `strategy_seed.py` | the `param_space()` + `strategy(data, tools, p)` contract and seed families |
| `prompt.py` | mutation prompt: whole scored history → one child structure |
| `backtest.py` | backtest, DE fit, quarter-CV median-OOS fitness, shift-the-signal skill test |
| `evolve.py` | islands, parent sampling, LLM mutation call, evaluation, report-only skill p-value |
| `run.py` | orchestration (`run_search`) + CLI; champion finalization + holdout report |
| `llm.py` | provider shim: Anthropic, OpenAI-compatible (Ollama/etc.), or Claude Code subagent |

Key CLI flags (`python run.py --help`): `--ticker --holdout-year --objective {sharpe,return}
--min-sharpe --iterations --reset-every --null-gate-configs --null-gate-shifts --no-null-gate`.

## 11. Known limitations / not yet done

- **Low power against exposure-management skill** (§9): the skill test detects directional timing
  well but risk-control timing only ~16% of the time at a realistic effect size, so a negative
  result means "not detected," not "absent."
- **Skill test under-corrects for the real search** (§7): it selects by a uniform max over P
  configs; the fitness selects by `differential_evolution`, which digs deeper into noise. The
  p-value is valid for the structure-under-uniform-search, not for the DE-fitted champion.
- **No multiple-testing correction**: `n_skill_significant` counts p<0.05 across ~50 candidates,
  where ~2.5 are expected by chance; it needs a Benjamini–Hochberg / FDR adjustment. And the
  champion's p is *selected-on* (fitness correlates with the skill statistic), so it is biased low
  — fine while results are negative, but it would overstate a borderline champion.
- **Adjacent-quarter CV contamination**: the backtest runs on the whole pool then masks (needed for
  indicator warm-up, causally correct), so each test quarter's opening positions warm up on
  training-quarter prices the params were chosen on. Not look-ahead, but a mild optimism in the OOS
  estimate. Contiguous blocks + a short embargo (hv-block CV, Racine 2000) would reduce it.
- **Fitness still uses per-split `differential_evolution`** (the ~2×10⁴-backtests path). The
  precompute-the-config-matrix idea (sample N configs once, argmax per split) is implemented for
  the *skill test* but **not** for the CV fitness, so exhaustive splits are not yet cheap there.
- **Sharpe has no risk-free adjustment** (§2); over 2017–2024 (near-zero rates → a hiking cycle) the
  excess-return Sharpe is modestly lower.
- **`min_sharpe` penalty units** (`10×`) mix return and Sharpe; not re-tuned.
- **Behavioural dedup** keys on the midpoint-param signal; two structures identical at midpoint but
  different elsewhere could false-merge.
- **Dead code**: `build_null_closes`, `insample_null_scores`, `null_max_bar_ccv` (block-bootstrap
  era) remain in `backtest.py` but are no longer called.
