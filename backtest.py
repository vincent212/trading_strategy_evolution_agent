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
quarters are kept intact and never shuffled.
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
    import alpha_tools
    lev = float(alpha_tools.MAX_LEVERAGE)                        # same position cap as clip_signal
    asset_ret = close.pct_change().fillna(0.0)
    if not isinstance(signal, pd.Series):                       # np.where output: align positionally
        signal = pd.Series(np.asarray(signal, dtype=float).ravel(), index=close.index)
    pos = signal.reindex(close.index).replace([np.inf, -np.inf], np.nan)
    pos = pos.fillna(0.0).clip(-lev, lev).shift(1).fillna(0.0)   # decide t-1, hold t
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


def _max_drawdown_arr(r: np.ndarray) -> float:
    """Max drawdown of a net-return stream, as a POSITIVE fraction (0.30 = a 30% drawdown)."""
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + r)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    return float(-dd.min())


def _cagr_arr(r: np.ndarray) -> float:
    """Compound annual growth rate of a net-return stream."""
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    total = float(np.prod(1.0 + r))
    yrs = r.size / PERIODS_PER_YEAR
    return float(total ** (1.0 / yrs) - 1.0) if yrs > 0 and total > 0 else float("nan")


def _mar_arr(r: np.ndarray) -> float:
    """MAR ratio = CAGR / max drawdown — return per unit of worst-case pain. Leverage inflates
    CAGR but inflates drawdown just as much, so MAR is where levered strategies get exposed."""
    mdd = _max_drawdown_arr(r)
    return float(_cagr_arr(r) / mdd) if mdd > 1e-9 else float("nan")


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

def buyhold_returns(close, cost: float = 0.0005) -> np.ndarray:
    """Net-return stream of a constant fully-long position — the buy-and-hold benchmark."""
    return run_backtest(pd.Series(1.0, index=close.index), close, cost).to_numpy()


def fit_on_mask(strategy_fn, space: dict, data: pd.DataFrame, tools,
                mask: np.ndarray, budget: int = 200, cost: float = 0.0005,
                seed: int = 0, objective: str = "sharpe", min_sharpe: float = 0.8,
                benchmark_ret=None) -> dict:
    """Tune params to MAXIMIZE the objective over the `mask` dates via differential evolution.
    If benchmark_ret is given, the strategy is scored on its ACTIVE return (strategy − benchmark),
    so the fit optimizes risk-adjusted OUTPERFORMANCE of the benchmark, not raw exposure.
    Returns the fitted params dict (empty if the strategy declares no parameters)."""
    names, bounds, kinds, extra = P.parse_space(space)
    close = data["close"]
    if not bounds:
        return {}

    def neg_score(x):
        p = P.decode(x, names, kinds, extra)
        try:
            r = run_backtest(strategy_fn(data, tools, p), close, cost).to_numpy()
            if benchmark_ret is not None:
                r = r - benchmark_ret                        # active return vs buy-and-hold
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
                   cost=0.0005, jobs=1, seed=0, objective="sharpe", min_sharpe=0.8,
                   benchmark_ret=None) -> dict:
    """Fit on each split's train quarters, score OOS on its test quarters. `median_oos` is the
    median per-split OOS OBJECTIVE score (the fitness the evolutionary loop maximizes); the
    OOS Sharpe and OOS annual return are always reported alongside it. If benchmark_ret is given,
    every score is on the ACTIVE return (strategy − benchmark), so the fitness is risk-adjusted
    OUTPERFORMANCE of buy-and-hold — a strategy that merely holds the asset scores 0, not positive."""
    def one(i):
        train_mask, test_mask = splits[i]
        p = fit_on_mask(strategy_fn, space, data, tools, train_mask, budget, cost,
                        seed + i, objective, min_sharpe, benchmark_ret=benchmark_ret)
        r = run_backtest(strategy_fn(data, tools, p), data["close"], cost).to_numpy()
        if benchmark_ret is not None:
            r = r - benchmark_ret                            # active return vs buy-and-hold
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
             objective="sharpe", min_sharpe=0.8, benchmark_ret=None) -> dict:
    """Fit params on ALL of `data` (used for the champion before the final holdout)."""
    mask = np.ones(len(data), dtype=bool)
    return fit_on_mask(strategy_fn, space, data, tools, mask, budget, cost, seed,
                       objective, min_sharpe, benchmark_ret=benchmark_ret)


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


def _score_position(pos, asset_ret, cost, objective, min_sharpe, benchmark_ret=None):
    """Score a raw position array exactly like run_backtest: 1-bar execution lag, cost on turnover.
    Keep this numerically identical to run_backtest — test_shift_gate.py::test_score_position_matches
    guards it (the skill test must price trades the same way the fitness/holdout do). If benchmark_ret
    is given, score the ACTIVE return (position − benchmark) so the skill test matches the excess-vs-
    buy-and-hold fitness (buy-and-hold then scores 0 and the shift test is neutral to it)."""
    held = np.empty(len(pos), dtype=float)               # float: never truncate an int input
    held[0] = 0.0
    held[1:] = pos[:-1]                                   # decide t-1, hold t (no lookahead)
    turn = np.abs(np.diff(held, prepend=0.0))
    r = held * asset_ret - cost * turn
    if benchmark_ret is not None:
        r = r - benchmark_ret
    return _score_arr(r, objective, min_sharpe)


def make_shift_offsets(n, n_shifts, seed=0, min_gap=250):
    """Random circular shift offsets that EXCLUDE near-identity shifts (within min_gap of 0 or n).
    A shift of a few bars barely moves a slow position, so it is not a genuine null draw and inflates
    the p-value. Draws from [min_gap, n - min_gap]; min_gap should exceed the longest indicator
    lookback (~250).

    The exclusion band needs 2*min_gap < n. If the series is too short to honour min_gap (< ~3x it),
    we keep the widest valid band (n//3 each side) so the offsets stay valid, and WARN — because the
    near-identity exclusion the p-value relies on is then weaker than advertised."""
    import warnings
    gap = int(min_gap)
    if 2 * gap >= n:                                    # too short to honour the requested gap
        gap = max(1, n // 3)
        warnings.warn(f"make_shift_offsets: n={n} too short for min_gap={min_gap}; using {gap}. "
                      f"Near-identity shift exclusion is weaker than advertised — the skill "
                      f"p-value may be inflated for slow structures on this ticker.", stacklevel=2)
    hi = max(gap + 1, n - gap)
    return np.random.default_rng(seed).integers(gap, hi, size=int(n_shifts))


def shift_null_pvalue(strategy_fn, space, data, tools, n_configs=64, shift_offsets=None,
                      n_shifts=50, cost=0.0005, seed=0, objective="sharpe", min_sharpe=0.8,
                      benchmark_ret=None):
    """Selection-aware SHIFT-THE-SIGNAL skill test (in-sample, per candidate). Guards against luck:
    is the score reproducible by a random re-timing of the structure's own positions? When
    benchmark_ret is given it scores the ACTIVE return (strategy − benchmark), matching the
    excess-vs-buy-and-hold fitness — so it asks whether the OUTPERFORMANCE of buy-and-hold is real
    or monkey-generatable, not whether raw exposure is.

    The null keeps the asset returns and each config's exposure profile EXACTLY, and destroys ONLY
    the alignment between signal and return by circularly shifting the position series. Buy & hold
    (a constant position, unchanged by a shift) lands exactly on the bar by construction.

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
            import alpha_tools
            lev = float(alpha_tools.MAX_LEVERAGE)
            if not isinstance(sig, pd.Series):                 # np.where output: align positionally
                sig = pd.Series(np.asarray(sig, dtype=float).ravel(), index=close.index)
            pos = (sig.reindex(close.index).replace([np.inf, -np.inf], np.nan)
                   .fillna(0.0).clip(-lev, lev).to_numpy())
        except Exception:
            pos = np.zeros(n)
        pos_list.append(np.asarray(pos, dtype=float))    # .to_numpy() is already C-contiguous float

    def maxscore(shift):
        # np.roll is circular, so a shifted draw has one wrap-seam transition the unshifted `real`
        # does not — a one-directional cost asymmetry of at most ~2*cost over ~n bars. Verified
        # immaterial (<0.001 total return over 2000 bars); left as-is rather than special-cased.
        best = float("-inf")
        for pos in pos_list:
            pp = pos if shift == 0 else np.roll(pos, int(shift))
            s = _score_position(pp, asset_ret, cost, objective, min_sharpe, benchmark_ret)
            if s > best:
                best = s
        return best

    real = maxscore(0)
    if shift_offsets is None:
        shift_offsets = make_shift_offsets(n, n_shifts, seed + 1)
    null = np.array([maxscore(k) for k in shift_offsets], dtype=float)
    if null.size == 0:                                   # no shifts (e.g. n_shifts=0): undefined test
        return {"pvalue": float("nan"), "real": float(real), "null_mean": float("nan"),
                "null_sd": float("nan"), "null_q95": float("nan"),
                "n_configs": len(configs), "n_shifts": 0,
                "exposure": float(np.mean([np.mean(np.abs(pp)) for pp in pos_list]))
                if pos_list else float("nan")}
    # (b+1)/(m+1), NOT b/m: the naive fraction can return exactly 0, which is not a valid p-value
    # and is biased low by ~1/m (Phipson & Smyth 2010). The floor is 1/(m+1).
    b, m = int((null >= real).sum()), int(len(null))
    pval = float((b + 1) / (m + 1))
    exposure = float(np.mean([np.mean(np.abs(pp)) for pp in pos_list])) if pos_list else float("nan")
    return {"pvalue": pval, "real": float(real), "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "null_q95": float(np.quantile(null, 0.95)),
            "n_configs": len(configs), "n_shifts": int(len(null)), "exposure": exposure}
