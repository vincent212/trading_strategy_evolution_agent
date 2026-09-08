# LATSS paper frame — validating the loop, and why overfitting forces simplicity

Working frame for the LATSS article. This reframes the paper away from "we found a
strategy" (we didn't; on real tools LATSS just overfits) toward a **framework
validation plus a methodological result about overfitting and Occam's razor**.

## Thesis

1. **LATSS is a framework, not a strategy.** It turns predictive *tools* into a
   selection rule. Whether it produces alpha depends entirely on whether the tools
   predict — so we validate it with a **synthetic tool of known signal strength**
   and ask only: *does LATSS recover the signal that is there?*
2. **There is no hold-out, on purpose.** With an unconstrained code-writing search,
   *all data in our possession is effectively in-sample* — overfitting cannot be
   prevented, so a "hold-out" is a fiction (the search can, and eventually will,
   fit it too). We therefore backtest on **all data 2015→today (2026 included)** and
   judge skill entirely **in-sample against a Rademacher/null-max bar**.
3. **The bar is necessary but not sufficient.** A Rademacher bar (the highest
   in-sample Sharpe the *strategy class* can extract from sign-flipped/noise returns)
   is the correct skill test. But it is only valid for a **bounded** class. An
   unconstrained LLM has effectively unbounded capacity, so — given enough search and
   no simplicity prior — it will find a **complex** strategy that clears both CV *and*
   the bar **on data with no real signal at all** (a false positive the bar cannot
   bound).
4. **Therefore the practitioner must impose simplicity a priori (Occam).** Since
   overfitting is unpreventable and all data is in-sample, the only defense is to
   restrict the strategy class to simple forms. A smaller class has a lower bar, so it
   both (a) resists sneak-through and (b) lets LATSS certify *weaker* real signals.

## Synthetic test-bed

- Universe: the 7 Magnificent-7 names; pick 1 of 7, long-only, active return vs the
  equal-weight basket. All data 2015-01-02 → today, in-sample.
- Tools: **four opaquely-named per-name features** (`feat_a…feat_d`). The LLM is not
  told which, if any, predict. In the positive setups, one hidden slot carries a
  signal for one target name (GOOGL): its forward *excess* return over the universe,
  standardized, plus noise, with a tunable strength `α` (rho). Everything else is
  persistent noise.
- Gate: the **Rademacher bar** — sign-flip the returns to ±1, refit the *same strategy
  class the search may emit*, take the best-in-class CV Sharpe; report mean and q95.
  A champion "wins" (has certifiable in-sample skill) iff its CV Sharpe clears the
  bar with headroom **H = CV² / bar² ≥ 2.5** (the article's genuine-but-weak line).
  The shift-the-signal p-value is removed; do not report it.

## Skill bars — three complementary tests (report all three)

To preempt the "you only tried Rademacher" criticism, every run reports **three**
independent bars. They measure different things and their *contrast* is a result. All
three are implemented in `backtest_xs.py` (`vc_bar`, `rademacher_bar`, `deflated_sharpe`)
so the experiments are reproducible from the public repo.

- **VC / Vapnik bar (analytical worst-case).** bar ≈ √(2·h·log(e·m/h) / Y), h = VC
  dimension of the class. Measured for our m≈2,936, Y≈11.7:
  | class | VC dim h | VC bar (ann. Sharpe) |
  |---|---|---|
  | single-feature / linear-4 (constrained) | ~5 | **2.5** |
  | gated / nonlinear feature class | ~15 | **4.0** |
  | unconstrained LLM (arbitrary code) | **∞** | **∞** |
  The last row is the crisp result: **no finite VC bar exists for arbitrary-code search.**
- **Rademacher bar (empirical, operative).** Sign-flip the returns, refit the class,
  best-in-class CV Sharpe; ≤ VC bar because it measures overfitting on *this* data, not
  the worst case. Measured (rich class): **+0.69 (honest/purged CV), +0.60 (random CV).**
  This is the tight, practical hurdle — and (Sweep 3) an unbounded class can sneak past it.
- **Deflated Sharpe Ratio (DSR, Bailey–López de Prado).** Deflates the observed Sharpe for
  trial count N, skew, kurtosis, and length. Reported for completeness, but with the
  explicit caveat (which the null-max article argues): **DSR keys on trial count, which is
  meaningless for exhaustive/automated search and perversely rewards under-searching** —
  capacity (VC/Rademacher), not trials, is the right invariant.

**How they line up as an argument:** the unconstrained LLM has *no finite VC bar*, an
*empirical Rademacher bar it can overfit past* (Sweep 3), and a *DSR that isn't even
well-defined without an arbitrary N* — three independent reasons to constrain the class.
The constrained linear class has a finite VC bar (~2.5), a low Rademacher bar, and a
well-defined DSR — so a real (even weak) signal can be certified against all three.

## Experiments

**Sweep 1 — signal-recovery threshold, unconstrained LLM.** Dial `α` from weak → strong.
At each level, run the LATSS search (LLM may emit any function of the features) and
record whether the champion clears the Rademacher bar with headroom. *Expected:* below
some **α\*** LATSS cannot recover the signal (its edge is buried under the class's noise-
fitting capacity); above α\* it does. Report α\* — the weakest alpha the unconstrained
loop can certify. (We cannot expect it to find arbitrarily weak alpha; the goal is to
map where the threshold sits.)

**Sweep 2 — constrain the class, re-sweep.** Restrict the prompt so the LLM may emit
only a **simple form** (e.g. a linear combination of the features — no gating, clipping,
or nonlinearity), and recompute the bar over that same restricted class. Re-sweep `α`.
*Expected:* the threshold drops to **α\*\_simple < α\***: a lower-capacity class has a lower
bar, so LATSS certifies a **weaker** real signal. This is the payoff — simplicity buys
sensitivity to genuine (weak) alpha.

**Sweep 3 — sneak-through (the money result).** Run the **unconstrained** LLM on **pure
noise (α = 0)** with **many iterations**. *Expected:* with enough search it eventually
produces a complex strategy that clears both CV and the (properly-computed) Rademacher
bar — a false positive. This demonstrates directly that the bar cannot defend an
unbounded class, and that in-sample validation alone is not enough without a class
constraint.

## CV/OOS robustness (a third axis — the current CV may be too easy)

The in-sample "OOS" score is only as trustworthy as the cross-validation behind it, and
the current one is suspect: **random 75/25 quarter splits, median-OOS**. Problems: (a)
adjacent quarters overlap, so train leaks into test; (b) splits are not chronological, so
it measures within-era resampling, not generalization across time; (c) the median over
random subsets is a low hurdle. A too-easy CV inflates every champion's score and shifts
the recovery threshold α\* artificially low. So we run every sweep under **several CV
computations** and check that conclusions are stable:

- **Current** — random quarter splits, median OOS (baseline, likely too permissive).
- **Purged + embargoed walk-forward** — contiguous chronological folds; purge training
  observations overlapping the test window and embargo a gap (≥ longest feature lookback)
  so there is no leakage. The honest CV.
- **Combinatorial purged CV (CPCV)** — many purged train/test combinations for a
  distribution of OOS scores rather than one path.

Report α\* (and the Rademacher headroom) under each; if the threshold moves a lot, the CV
was doing the work, not the signal — itself a finding.

### Result R0 — the CV is load-bearing (measured)

Direct comparison, all data 2015→2026 in-sample, **near-perfect injected signal** (the
target's forward-excess oracle, rho = 0.98), unconstrained (rich) strategy class:

| CV method | signal CV (feat_c) | Rademacher bar q95 | headroom H | verdict |
|---|---|---|---|---|
| **Random quarter (current)** | **+0.91** | +0.60 | **2.2×** | clears |
| **Purged k-fold (honest)** | **+0.67** | +0.69 | **0.9×** | **buried** |

The random-quarter CV **manufactures the recovery**: it simultaneously *inflates* the
signal's score (+0.91 vs +0.67 — adjacent-quarter leakage lets the fit borrow test
information) and *lowers* the bar (+0.60 vs +0.69), which flips the verdict from "buried"
to "clears." Under the honest purged/embargoed CV, **even a near-perfect signal does not
clear the Rademacher bar with the unconstrained class** (H = 0.9×): the class can fit
+0.69 of pure sign-flipped noise, more than the signal's honest CV edge (+0.67).

Two things this licenses in the paper:
1. **A permissive CV is itself an overfitting channel** — independent of the strategy
   search. Leaky/random resampling inflates apparent skill and must be reported; the
   honest CV is purged, embargoed, chronological.
2. **Under an honest CV, the unconstrained class cannot certify even a strong signal.**
   So constraining the class (Sweep 2) is not an optional comparison — it is *required*
   to certify anything in-sample. Honest CV + valid bar ⇒ you are forced to simplicity.

## What each result licenses in the paper

- Sweeps 1–2 together: *LATSS works — it recovers injected signal — and constraining the
  strategy class (simplicity) strictly improves the weakest signal it can certify.*
- Sweep 3: *an unconstrained LLM will manufacture in-sample-significant strategies from
  noise that pass even a correct skill bar; overfitting is unpreventable.*
- Combined conclusion: *because all available data is effectively in-sample and
  overfitting cannot be prevented, a quant using LLM-driven search must adopt a priori
  simplicity heuristics (Occam) — the skill bar is necessary but only meaningful over a
  deliberately bounded, simple class.*

## Method notes / honesty

- No hold-out is used or reported anywhere; skill is judged only vs the in-sample
  Rademacher bar over the exact class the search may emit.
- The Rademacher bar must be computed over the **same** class the LLM is allowed to
  produce; a bar over a smaller class is invalid (it under-estimates and lets overfits
  through — an error to avoid, not a knob to tune).
- Reference for the bar: the null-max / empirical-Rademacher construction
  (vincentmayeski.substack.com/p/what-is-the-null-max-bar-and-could).
- The prior real-tool Mag-7 runs (which only overfit and never cleared a valid bar) are
  the empirical illustration of point 2 — all-in-sample, unpreventable overfitting.
