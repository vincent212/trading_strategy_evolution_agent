# How the pipeline actually works

This documents the real behaviour of the code as it stands, not an idealized version.
Read it top to bottom; each section names the file and function that implements it.

---

## 0. One-paragraph summary

An evolutionary loop uses an LLM as a mutation operator to evolve the *structure* of a
trainable trading strategy for one ticker. Each structure declares free parameters; a numerical
optimizer fits those parameters. Structures are ranked by their **cross-validated out-of-sample
outperformance of buy-and-hold** (the fitness — scored on the *active return*, strategy minus
buy-and-hold, so merely holding the asset scores 0). A **report-only** statistical test — a
selection-aware "shift-the-signal" permutation test on the same active return — is computed for
every candidate and headlined on the champion; it answers "is the outperformance real, or
monkey-generatable?" The final champion is fitted on all in-sample data and measured **once** on a
sealed holdout period, against buy-and-hold. A real find must both **beat buy-and-hold** and **clear
the skill test**; the two are kept as distinct questions (economic vs statistical significance).

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

## 4. Fitness = cross-validated median OOS **outperformance of buy-and-hold** — `backtest.py: ccv_median_oos`

This is the number the search selects on. With `vs_buyhold` on (the default), every score is on the
**active return** — the strategy's net return **minus buy-and-hold's** (a constant fully-long
position, `backtest.py: buyhold_returns`) — so the fitness is risk-adjusted **outperformance of
buy-and-hold**, not raw return.

```
bench   = buyhold_returns(pool.close)                 # net return of a constant fully-long position
splits  = 100 quarter-based CV splits of pool          # make_quarter_splits (train 75% / test 25%)
for each split (train, test):
    p   = argmax_p  _score_arr( (backtest(structure(pool,p)) − bench)[train] )   # FIT on active return
    s_i = _score_arr( (backtest(structure(pool,p)) − bench)[test] )              # score OOS on active
fitness = median_i(s_i)                                                          # diag["median_oos"]
```

- **Why active return.** Raw return rewards mere exposure/drift — being long a trending stock scores
  well with zero timing skill, so a raw-return search selects overfit long-biased "champions."
  Scoring `strategy − buy-and-hold` makes **buy-and-hold itself score exactly 0**, so the search is
  credited only for exposure it timed *better than passively holding*. For a single stock, "be
  exposed to the asset at the right time" is the skill, and it shows up here as beating buy-and-hold.
- **Asset-dependent.** `vs_buyhold` is a flag (default on; `--no-vs-buyhold` scores raw). On for
  single stocks / equity ETFs; off for futures spreads, market-neutral/pairs, mean-reverting or
  cash-like assets where there is no persistent long drift to hold.
- `fit_on_mask` fits parameters with SciPy `differential_evolution` (continuous search), budget
  `fit_budget` (default 200 function evals — `maxiter` is derived so total evals ≈ `fit_budget`).
  This is the expensive part: ~100 fits × ~200 backtests each ≈ **2×10⁴** backtests per candidate.
- Reported alongside: OOS (active) Sharpe = information ratio vs buy-and-hold, OOS excess annualized
  return, fraction of positive splits. A strategy that merely holds the asset lands at ~0 on all of
  these; a positive fitness means it beat buy-and-hold on the held-out quarters, risk-adjusted.

## 5. The evolution loop — `evolve.py: Evolver.run / step`

```
seed the population with the seed families            # strategy_seed.py: SEEDS
for it in 1 .. iterations:
    explore = (it % 5 == 0)                           # every 5th iteration = random-jump turn
    history = whole scored history, best-first         # prompt.py: build_user_prompt
    code    = LLM.mutate(system+theme, history)        # llm.py: Anthropic | OpenAI-compatible | subagent
    child   = evaluate(code)                           # see §6
    if child is None and it was a CODE error:
        retry up to 2× by sending the error back to the LLM   # _self_correct
    if child survived: add to the pool; update global best
    every reset_every iterations: cull the weakest half of the pool  # db.reset()
champion = highest-fitness program ever seen
```

There is **one population** (no islands — see §11). The `system` prompt carries the contract + rules
+ an editable **investment-theme** paragraph (`prompt.py: system_prompt` / `theme.txt`, e.g. "mostly
long, add on low-vol dips, cut exposure in high vol") that steers the model toward realistic
single-stock structures. The LLM is only a **mutation operator**. It never sees prices; it sees the
scored history of
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
skill. With `vs_buyhold` on, it scores the **same active return** as the fitness (§4), so it asks
whether the **outperformance of buy-and-hold** is real or monkey-generatable — not whether raw
exposure is. Buy-and-hold's active return is 0, so it scores 0 and lands at p = 1 (neutral). The two
requirements are unified: a real find must **beat buy-and-hold** (positive fitness) **and** clear the
skill test (**not monkey-generatable**).

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
  block bootstrap over-corrected and was poorly calibrated. Both were abandoned and their helpers
  deleted.)
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

Calibration (real NVDA pool, verified, with the near-identity fix + `(b+1)/(m+1)`, on **raw**
returns): buy-and-hold p = 1.00 (lands on the bar); the four seeds p ≈ **0.04 / 0.47 / 0.40 /
0.24** — seed[0] is borderline at 0.04, but that is 1 of 4 tests with no multiplicity correction, so
it is weak evidence, not a finding; a look-ahead perfect-timing position p = the floor `1/(m+1)`
(the estimator never reports 0). With `vs_buyhold` on the test scores the active return, so seed
values differ, but the invariants hold either way — buy-and-hold p = 1.00 (active return 0),
perfect timing at the floor.

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

The completed NVDA runs so far used the **raw-return** fitness (before the excess-vs-buy-and-hold
change in §4). Two independent 50-iteration searches produced structures with high CV fitness
(median OOS Sharpe ~1.3) that **did not generalize**: one collapsed into buy-and-hold out-of-sample
(its rare short never fired in 2025–2026), the other **lost** to buy-and-hold both holdout years
(−8.0% vs +38.9% in 2025; +5.2% vs +20.2% in 2026). Champion skill p-values on the corrected test
were borderline/insignificant (0.050 and 0.070). Blunt reading: raw-return fitness rewards being
long a trending stock, so it selected overfit long-biased champions with no real edge.

That is exactly why the fitness was changed to **outperformance of buy-and-hold** (§4): under raw
return, "just hold NVDA" is a great score; under active return it scores 0, and the search is only
credited for beating it. The expected result on NVDA with the new fitness is that **little or
nothing reliably beats buy-and-hold out-of-sample** — i.e. ~0 champion fitness — which is the
correct, honest answer for a single high-drift name, not an overfit "champion." (Pending: a re-run
under the excess-vs-buy-and-hold fitness to confirm.)

Note on skill-test power: it detects directional timing well but *exposure-management* timing (cut
risk in genuinely worse windows) only ~16% of the time at a realistic effect size in synthetic
checks, so a non-significant p means "not detected," not "no skill exists" (exact figure depends on
the injected effect size; the qualitative gap is robust).

## 10. Configuration & files

| file | role |
|---|---|
| `data.py` | Yahoo price data (any ticker), cached |
| `alpha_tools.py` | fixed causal indicator library the strategies may call |
| `params.py` | encode/sample/decode a `param_space()` for the optimizer |
| `strategy_seed.py` | the `param_space()` + `strategy(data, tools, p)` contract and seed families |
| `prompt.py` | mutation prompt: contract + rules + investment theme (system) and whole scored history (user) |
| `theme.txt` | editable investment-theme paragraph injected into the system prompt (swap without code) |
| `backtest.py` | backtest, DE fit, quarter-CV median-OOS (active) fitness, buy&hold benchmark, shift-the-signal skill test |
| `evolve.py` | single population + periodic cull, LLM mutation call (system+theme), evaluation, report-only skill p-value |
| `run.py` | orchestration (`run_search`) + CLI; champion finalization + holdout report |
| `llm.py` | provider shim: Anthropic, OpenAI-compatible (Ollama/etc.), or Claude Code subagent |

Key CLI flags (`python run.py --help`): `--ticker --holdout-year --objective {sharpe,return}
--min-sharpe --no-vs-buyhold --iterations --reset-every --null-gate-configs --null-gate-shifts
--no-null-gate`. `--no-vs-buyhold` scores raw returns instead of active (excess-over-buy-and-hold);
keep the default (active) for single stocks, use `--no-vs-buyhold` for assets with no long drift.

## 11. Known limitations / not yet done

- **Islands were removed** (they were vestigial — the prompt always used the whole population and the
  per-island parent sampler was never called, so there was no isolation and no crossover). Now a
  single pool with a periodic cull of the weakest half (`db.reset()`). If parallel-lineage diversity
  ever becomes the bottleneck, real island isolation (island-local prompts + migration) could be
  added back deliberately.
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
