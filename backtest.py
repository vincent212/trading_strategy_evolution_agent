"""
Evaluation engine for TRAINABLE strategies.

Pipeline (per candidate):
  1. run_backtest / sharpe   — a fitted strategy -> net return stream -> Sharpe.
                               A 1-bar execution lag is applied here.
  2. make_quarter_splits     — cut the (pre-holdout) history into calendar quarters
                               and draw 100 random 75/25 quarter splits.
  3. fit_on_mask             — on each split's 75% train quarters, tune the strategy's
                               params with SciPy differential evolution to MAXIMIZE
                               train Sharpe. (This is 'training' — the LLM doesn't do it.)
  4. ccv_median_oos          — score the fitted params on the 25% test quarters -> one
                               OOS Sharpe per split; take the MEDIAN over the 100 splits.
                               That median is the fitness the evolutionary loop maximizes.
  5. null_max_bar_ccv        — run the SAME fit+CCV on sign-flipped (pure-noise) returns;
                               the best median-OOS the search can wring out of noise is
                               the null-max bar. The real median OOS must clear it.

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
                seed: int = 0) -> dict:
    """Tune params to MAXIMIZE Sharpe over the `mask` dates via differential evolution.
    Returns the fitted params dict (empty if the strategy declares no parameters)."""
    names, bounds, kinds, extra = P.parse_space(space)
    close = data["close"]
    if not bounds:
        return {}

    def neg_sharpe(x):
        p = P.decode(x, names, kinds, extra)
        try:
            r = run_backtest(strategy_fn(data, tools, p), close, cost).to_numpy()
            return -_sharpe_arr(r[mask])
        except Exception:
            return 10.0

    popsize = 8
    maxiter = max(1, int(round(budget / (popsize * len(bounds)))) - 1)
    res = differential_evolution(
        neg_sharpe, bounds, popsize=popsize, maxiter=maxiter, seed=seed,
        polish=False, tol=0.01, mutation=(0.5, 1.0), recombination=0.7)
    return P.decode(res.x, names, kinds, extra)


def _oos_sharpe(strategy_fn, data, tools, p, mask, cost) -> float:
    r = run_backtest(strategy_fn(data, tools, p), data["close"], cost).to_numpy()
    return _sharpe_arr(r[mask])


# ---- cross-validated OOS (the fitness) -------------------------------------

def ccv_median_oos(strategy_fn, space, data, tools, splits, budget=200,
                   cost=0.0005, jobs=1, seed=0) -> dict:
    """Fit on each split's train quarters, score OOS on its test quarters; return the
    median (and spread) of the per-split OOS Sharpes."""
    def one(i):
        train_mask, test_mask = splits[i]
        p = fit_on_mask(strategy_fn, space, data, tools, train_mask, budget, cost, seed + i)
        return _oos_sharpe(strategy_fn, data, tools, p, test_mask, cost)

    n = len(splits)
    if jobs and jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            oos = list(ex.map(one, range(n)))
    else:
        oos = [one(i) for i in range(n)]
    oos = np.asarray(oos, dtype=float)
    return {
        "median_oos": float(np.median(oos)),
        "mean_oos": float(np.mean(oos)),
        "std_oos": float(np.std(oos)),
        "frac_positive": float(np.mean(oos > 0)),
        "n_splits": n,
    }


def fit_full(strategy_fn, space, data, tools, budget=400, cost=0.0005, seed=0) -> dict:
    """Fit params on ALL of `data` (used for the champion before the final holdout)."""
    mask = np.ones(len(data), dtype=bool)
    return fit_on_mask(strategy_fn, space, data, tools, mask, budget, cost, seed)


# ---- the null-max bar (capacity gate, on the CCV OOS) -----------------------

def null_max_bar_ccv(strategy_fn, space, data, tools, splits, budget=200,
                     cost=0.0005, n_sims=10, seed=0, jobs=1, quantile=0.95) -> dict:
    """Run the SAME fit+CCV on sign-flipped (pure-noise) returns, n_sims times, and
    return a high quantile of the resulting median-OOS values as the noise ceiling.

    A real strategy's median OOS Sharpe must clear this bar: on genuine noise, fitting
    the train quarters cannot generalize to the test quarters, so the median-OOS the
    search can achieve here is what luck alone buys. Fed the CCV OOS number, not an
    in-sample one.
    """
    ret = data["close"].pct_change().fillna(0.0).to_numpy()
    idx = data.index
    rng = np.random.default_rng(seed)
    meds = np.empty(n_sims)
    for s in range(n_sims):
        eps = rng.choice([-1.0, 1.0], size=len(ret))
        fake_close = pd.Series(100.0 * np.cumprod(1.0 + ret * eps), index=idx)
        fdata = data.copy()
        fdata["close"] = fake_close
        res = ccv_median_oos(strategy_fn, space, fdata, tools, splits, budget,
                             cost, jobs, seed=10_000 + s)
        meds[s] = res["median_oos"]
    return {
        "bar": float(np.quantile(meds, quantile)),
        "mean_noise_median": float(meds.mean()),
        "max_noise_median": float(meds.max()),
        "n_sims": int(n_sims),
    }
