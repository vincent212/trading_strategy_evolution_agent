"""
Evaluation engine for TRAINABLE strategies.

Pipeline (per candidate):
  1. run_backtest / sharpe   — a fitted strategy -> net return stream -> score.
                               A 1-bar execution lag is applied here.
  2. make_quarter_splits     — cut the (pre-holdout) history into calendar quarters
                               and draw 100 random 75/25 quarter splits.
  3. fit_on_mask             — on each split's 75% train quarters, tune the strategy's
                               params with SciPy differential evolution to MAXIMIZE the
                               train objective. (This is 'training' — the LLM doesn't do it.)
  4. ccv_median_oos          — score the fitted params on the 25% test quarters -> one
                               OOS score per split; take the MEDIAN over the 100 splits.
                               That median is the fitness the evolutionary loop maximizes.
  5. shift_null_pvalue       — REPORT-ONLY skill test: selection-aware shift-the-signal
                               permutation p-value (does the timing beat a random re-timing
                               of the structure's own positions?). Logged, not a filter.

The signal is always computed on the FULL continuous series (so indicators stay
causal); the train/test split only selects which dates' returns are scored. Whole
quarters are kept intact and never shuffled. NOTE: the block-bootstrap helpers
(build_null_closes, insample_null_scores, null_max_bar_ccv) are a superseded null
model, retained but no longer called — see PIPELINE.md §11.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import params as P

PERIODS_PER_YEAR = 252


# ---- core backtest ----------------------------------------------------------

def run_backtest(signal: pd.Series, close: pd.Series,
                 cost_per_turn: float = 0.0005) -> pd.Series:
    """Fitted signal -> net daily returns, with a 1-bar execution lag and costs."""
    asset_ret = close.pct_change().fillna(0.0)
    pos = signal.reindex(close.index).replace([np.inf, -np.inf], np.nan)
    pos = pos.fillna(0.0).clip(-1.0, 1.0).shift(1).fillna(0.0)   # decide t-1, hold t
    turnover = pos.diff().abs().fillna(0.0)
    return pos * asset_ret - cost_per_turn * turnover


def sharpe(returns: pd.Series, annualize: bool = True) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    s = r.mean() / r.std()
    return float(s * np.sqrt(PERIODS_PER_YEAR)) if annualize else float(s)


def _sharpe_arr(r: np.ndarray, min_obs: int = 20) -> float:
    r = r[np.isfinite(r)]
    if r.size < min_obs:
        return 0.0
    sd = r.std()
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(PERIODS_PER_YEAR))


def _ann_return_arr(r: np.ndarray, min_obs: int = 20) -> float:
    """Annualized (arithmetic) mean return of a net-return stream — smooth for optimizing."""
    r = r[np.isfinite(r)]
    if r.size < min_obs:
        return 0.0
    m = r.mean()
    return float(m * PERIODS_PER_YEAR) if np.isfinite(m) else 0.0


def _score_arr(r: np.ndarray, objective: str = "sharpe",
               min_sharpe: float = 0.8, penalty: float = 10.0) -> float:
    """The scalar the search MAXIMIZES for a return stream.
      objective='sharpe' -> annualized Sharpe (default; the original behaviour).
      objective='return' -> annualized return, but SOFT-penalized below a Sharpe floor, so the
                            fit prefers return among strategies that keep Sharpe >= min_sharpe.
    """
    if objective == "sharpe":
        return _sharpe_arr(r)
    sh = _sharpe_arr(r)
    ret = _ann_return_arr(r)
    if sh >= min_sharpe:
        return ret
    return ret - penalty * (min_sharpe - sh)     # pull the fit back toward the Sharpe floor


# ---- quarter cross-validation splits ---------------------------------------

def make_quarter_splits(index: pd.DatetimeIndex, n_splits: int = 100,
                        train_frac: float = 0.75, seed: int = 0):
    """Return n_splits (train_mask, test_mask) boolean arrays over `index`.
    Whole calendar quarters are randomly assigned to train (75%) or test (25%)."""
    q = pd.PeriodIndex(index, freq="Q")
    codes, uniq = pd.factorize(q, sort=True)
    n_q = len(uniq)
    if n_q < 4:
        raise ValueError(f"need >=4 quarters to cross-validate, got {n_q}")
    n_train = max(1, min(n_q - 1, int(round(n_q * train_frac))))
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(n_splits):
        perm = rng.permutation(n_q)
        is_train_q = np.zeros(n_q, dtype=bool)
        is_train_q[perm[:n_train]] = True
        train_mask = is_train_q[codes]
        splits.append((train_mask, ~train_mask))
    return splits


# ---- fit (training) ---------------------------------------------------------

def fit_on_mask(strategy_fn, space: dict, data: pd.DataFrame, tools,
                mask: np.ndarray, budget: int = 200, cost: float = 0.0005,
                seed: int = 0, objective: str = "sharpe", min_sharpe: float = 0.8) -> dict:
    """Tune params to MAXIMIZE the objective over the `mask` dates via differential evolution.
    Returns the fitted params dict (empty if the strategy declares no parameters)."""
    names, bounds, kinds, extra = P.parse_space(space)
    close = data["close"]
    if not bounds:
        return {}

    def neg_score(x):
        p = P.decode(x, names, kinds, extra)
        try:
            r = run_backtest(strategy_fn(data, tools, p), close, cost).to_numpy()
            return -_score_arr(r[mask], objective, min_sharpe)
        except Exception:
            return 10.0

    popsize = 8
    maxiter = max(1, int(round(budget / (popsize * len(bounds)))) - 1)
    res = differential_evolution(
        neg_score, bounds, popsize=popsize, maxiter=maxiter, seed=seed,
        polish=False, tol=0.01, mutation=(0.5, 1.0), recombination=0.7)
    return P.decode(res.x, names, kinds, extra)


# ---- cross-validated OOS (the fitness) -------------------------------------

def ccv_median_oos(strategy_fn, space, data, tools, splits, budget=200,
                   cost=0.0005, jobs=1, seed=0, objective="sharpe", min_sharpe=0.8) -> dict:
    """Fit on each split's train quarters, score OOS on its test quarters. `median_oos` is the
    median per-split OOS OBJECTIVE score (the fitness the evolutionary loop maximizes); the
    OOS Sharpe and OOS annual return are always reported alongside it."""
    def one(i):
        train_mask, test_mask = splits[i]
        p = fit_on_mask(strategy_fn, space, data, tools, train_mask, budget, cost,
                        seed + i, objective, min_sharpe)
        r = run_backtest(strategy_fn(data, tools, p), data["close"], cost).to_numpy()
        rt = r[test_mask]
        return (_score_arr(rt, objective, min_sharpe), _sharpe_arr(rt), _ann_return_arr(rt))

    n = len(splits)
    if jobs and jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            rows = list(ex.map(one, range(n)))
    else:
        rows = [one(i) for i in range(n)]
    rows = np.asarray(rows, dtype=float)
    score, shp, ret = rows[:, 0], rows[:, 1], rows[:, 2]
    return {
        "median_oos": float(np.median(score)),       # fitness (objective units)
        "mean_oos": float(np.mean(score)),
        "std_oos": float(np.std(score)),
        "frac_positive": float(np.mean(score > 0)),
        "median_sharpe": float(np.median(shp)),
        "median_return": float(np.median(ret)),
        "objective": objective,
        "n_splits": n,
    }


def fit_full(strategy_fn, space, data, tools, budget=400, cost=0.0005, seed=0,
             objective="sharpe", min_sharpe=0.8) -> dict:
    """Fit params on ALL of `data` (used for the champion before the final holdout)."""
    mask = np.ones(len(data), dtype=bool)
    return fit_on_mask(strategy_fn, space, data, tools, mask, budget, cost, seed,
                       objective, min_sharpe)


# ---- in-sample (no-CV) null-max bar: the per-candidate monkeys gate ----------

def insample_score(strategy_fn, space, data, tools, budget=200, cost=0.0005, seed=0,
                   objective="sharpe", min_sharpe=0.8) -> float:
    """Fit params on ALL of `data` and score on that SAME data — the in-sample (no-CV) score."""
    mask = np.ones(len(data), dtype=bool)
    p = fit_on_mask(strategy_fn, space, data, tools, mask, budget, cost, seed,
                    objective, min_sharpe)
    r = run_backtest(strategy_fn(data, tools, p), data["close"], cost).to_numpy()
    return _score_arr(r[mask], objective, min_sharpe)


# ---- selection-aware SHIFT-THE-SIGNAL skill test (the real per-candidate gate) --------------

def sample_configs(space, n_configs=64, seed=0):
    """Sample n_configs parameter dicts uniformly from the structure's param space.
    A parameterless structure (e.g. buy & hold) yields exactly one config."""
    names, bounds, kinds, extra = P.parse_space(space)
    if not bounds:
        return [{}]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_configs)):
        x = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds])
        out.append(P.decode(x, names, kinds, extra))
    return out


def _score_position(pos, asset_ret, cost, objective, min_sharpe):
    """Score a raw position array exactly like run_backtest: 1-bar execution lag, cost on turnover."""
    held = np.empty_like(pos)
    held[0] = 0.0
    held[1:] = pos[:-1]                                   # decide t-1, hold t (no lookahead)
    turn = np.abs(np.diff(held, prepend=0.0))
    r = held * asset_ret - cost * turn
    return _score_arr(r, objective, min_sharpe)


def make_shift_offsets(n, n_shifts, seed=0, min_gap=250):
    """Random circular shift offsets that EXCLUDE near-identity shifts (within min_gap of 0 or n).
    A shift of a few bars barely moves a slow position, so it is not a genuine null draw and inflates
    the p-value. Draw from [min_gap, n - min_gap]; min_gap should exceed the longest indicator
    lookback (~250). Clamped for short series so the range is always valid."""
    lo = min(int(min_gap), max(1, n // 4))
    hi = max(lo + 1, n - lo)
    return np.random.default_rng(seed).integers(lo, hi, size=int(n_shifts))


def shift_null_pvalue(strategy_fn, space, data, tools, n_configs=64, shift_offsets=None,
                      n_shifts=50, cost=0.0005, seed=0, objective="sharpe", min_sharpe=0.8):
    """Selection-aware SHIFT-THE-SIGNAL skill test (in-sample, per candidate).

    The null keeps the asset returns and each config's exposure profile EXACTLY, and destroys ONLY
    the alignment between signal and return by circularly shifting the position series. So the drift
    and the exposure level cancel — the test isolates timing skill (exposure MANAGEMENT), the one
    thing fitting can fake. Buy & hold (a constant position, unchanged by a shift) lands exactly on
    the bar by construction, which is the correct zero point.

    Selection-aware: `real` and every null draw take the MAX over the SAME n_configs sampled configs,
    so the 'I tried many settings and kept the best' inflation appears on both sides and cancels.

    p = fraction of null draws (each a max over configs under a random shift) that meet or beat the
    real max. p<0.05 => timing skill beyond what searching this structure this hard can fake."""
    close = data["close"]
    asset_ret = close.pct_change().fillna(0.0).to_numpy()
    n = len(asset_ret)
    configs = sample_configs(space, n_configs, seed)
    pos_list = []
    for p in configs:
        try:
            with np.errstate(divide="ignore", invalid="ignore"):   # constant-0 signals warn benignly
                sig = strategy_fn(data, tools, p)
            pos = (sig.reindex(close.index).replace([np.inf, -np.inf], np.nan)
                   .fillna(0.0).clip(-1.0, 1.0).to_numpy())
        except Exception:
            pos = np.zeros(n)
        pos_list.append(np.ascontiguousarray(pos, dtype=float))

    def maxscore(shift):
        best = float("-inf")
        for pos in pos_list:
            pp = pos if shift == 0 else np.roll(pos, int(shift))
            s = _score_position(pp, asset_ret, cost, objective, min_sharpe)
            if s > best:
                best = s
        return best

    real = maxscore(0)
    if shift_offsets is None:
        shift_offsets = make_shift_offsets(n, n_shifts, seed + 1)
    null = np.array([maxscore(k) for k in shift_offsets], dtype=float)
    # (b+1)/(m+1), NOT b/m: the naive fraction can return exactly 0, which is not a valid p-value
    # and is biased low by ~1/m (Phipson & Smyth 2010). The floor is 1/(m+1).
    b, m = int((null >= real).sum()), int(len(null))
    pval = float((b + 1) / (m + 1))
    exposure = float(np.mean([np.mean(np.abs(pp)) for pp in pos_list])) if pos_list else float("nan")
    return {"pvalue": pval, "real": float(real), "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "null_q95": float(np.quantile(null, 0.95)),
            "n_configs": len(configs), "n_shifts": int(len(null)), "exposure": exposure}


def build_null_closes(returns, index, n_paths=50, block=21, seed=0, base=100.0):
    """Circular block-bootstrap of `returns` -> n_paths fake 'close' Series.

    Why not sign-flip: flipping signs forces E[return]=0, so the null has NO DRIFT. On a
    trending asset (NVDA pool ~1.2 Sharpe) that makes the bar trivially beatable by anything
    net-long — even parameter-free buy & hold clears it. A block bootstrap PRESERVES the drift,
    the volatility clustering and short-range autocorrelation, while DESTROYING the long-range
    timing structure a strategy would need real skill to exploit. So the monkeys get the drift
    too, and only genuine timing skill clears the bar.

    Generated ONCE and reused for every candidate (COMMON RANDOM NUMBERS): the gate threshold is
    then a consistent comparison across candidates, not noise re-randomised per candidate (which
    would give two identical structures opposite verdicts)."""
    r = np.asarray(returns, dtype=float)
    T = len(r)
    rng = np.random.default_rng(seed)
    closes = []
    for _ in range(int(n_paths)):
        out = np.empty(T)
        i = 0
        while i < T:
            s = int(rng.integers(0, T))
            L = min(int(block), T - i)
            out[i:i + L] = r[(s + np.arange(L)) % T]     # circular block
            i += L
        closes.append(pd.Series(base * np.cumprod(1.0 + out), index=index))
    return closes


def insample_null_scores(strategy_fn, space, real_data, tools, null_closes,
                         budget=200, cost=0.0005, seed=0,
                         objective="sharpe", min_sharpe=0.8) -> np.ndarray:
    """In-sample score of ONE structure on each precomputed null close path. Same
    fit-then-score-on-the-same-data procedure as the real in-sample score, so the overfitting is
    matched and cancels — a legitimate permutation test. The caller compares the real in-sample
    score to (mean + c*sd) of these draws (ADDITIVE, sign-safe — a multiplicative bar inverts
    when the score is negative, which the return-minus-Sharpe-penalty objective is routinely)."""
    vals = np.empty(len(null_closes))
    for j, fc in enumerate(null_closes):
        fdata = real_data.copy()
        fdata["close"] = fc
        vals[j] = insample_score(strategy_fn, space, fdata, tools, budget, cost,
                                 seed + j, objective, min_sharpe)
    return vals


# ---- the null-max bar (capacity gate, on the CCV OOS) -----------------------

def null_max_bar_ccv(strategy_fn, space, data, tools, splits, budget=200,
                     cost=0.0005, n_sims=10, seed=0, jobs=1, quantile=0.95,
                     objective="sharpe", min_sharpe=0.8, block=21) -> dict:
    """Run the SAME fit+CCV on drift-preserving block-bootstrap nulls, n_sims times, and return
    the noise ceiling (both a high quantile and mean/sd for an additive verdict).

    Uses the block bootstrap, NOT sign-flip: sign-flip zeroed the drift and made this bar
    trivially beatable by any net-long strategy on a trending asset (see build_null_closes)."""
    closes = build_null_closes(data["close"].pct_change().fillna(0.0).to_numpy(),
                               data.index, n_sims, block, seed)
    meds = np.empty(n_sims)
    for s, fc in enumerate(closes):
        fdata = data.copy()
        fdata["close"] = fc
        res = ccv_median_oos(strategy_fn, space, fdata, tools, splits, budget,
                             cost, jobs, seed=10_000 + s, objective=objective,
                             min_sharpe=min_sharpe)
        meds[s] = res["median_oos"]
    return {
        "bar": float(np.quantile(meds, quantile)),
        "mean_noise_median": float(meds.mean()),
        "std_noise_median": float(meds.std()),
        "max_noise_median": float(meds.max()),
        "n_sims": int(n_sims),
    }
