"""
Cross-sectional evaluation engine for the "pick 1 of N" selection experiment.

Contract (deliberately NAME-SYMMETRIC — the reason the Mag-7 universe is not a
look-ahead trap): the evolved candidate defines

    param_space() -> dict of tunable knobs
    score(series, tools, p) -> a pd.Series of per-bar SCORES for ONE price series

`score` sees a SINGLE anonymous close series at a time. It has no ticker identity
and no view of the other names, so it CANNOT hardcode "hold NVDA" — it can only
express a technical rule ("prefer the name with the strongest momentum", etc.).
The harness applies `score` to every name, and each bar HOLDS the single name with
the highest score (argmax -> one-hot weight). Selection is therefore an emergent,
causal function of technical measurements, evaluated identically across names.

Scoring mirrors backtest.py: 1-bar execution lag, cost on turnover, and fitness =
risk-adjusted ACTIVE return vs the EQUAL-WEIGHT benchmark (so merely holding the
basket scores 0). CV splits, DE fit, the shift-the-signal null, and the scalar
objective are all reused from backtest.py — only the return construction differs.
"""
from __future__ import annotations
import inspect
import numpy as np
import pandas as pd

import params as P
import backtest as bt

PERIODS_PER_YEAR = bt.PERIODS_PER_YEAR


# ---- score matrix + selection ----------------------------------------------

def score_matrix(score_fn, panel: pd.DataFrame, tools, p, feats=None) -> np.ndarray:
    """Apply the per-series `score` to every column -> a T x N score array.
    Two contracts, auto-detected by arity: score(series, tools, p) [price only] or
    score(series, feats, tools, p) [price + per-name extra features], where `feats`
    is a dict {name: DataFrame T x N}; each name receives {feat: feat[name]}.
    Non-finite scores (warm-up NaNs, div-by-zero) become -inf so that name is simply
    not selectable that bar rather than poisoning the argmax."""
    try:
        n_par = len(inspect.signature(score_fn).parameters)
    except (TypeError, ValueError):
        n_par = 3
    cols = []
    for name in panel.columns:
        if n_par >= 4:
            nf = {k: v[name] for k, v in (feats or {}).items()}
            s = score_fn(panel[name], nf, tools, p)
        else:
            s = score_fn(panel[name], tools, p)
        if not isinstance(s, pd.Series):
            s = pd.Series(np.asarray(s, dtype=float).ravel(), index=panel.index)
        elif not s.index.equals(panel.index):
            if len(s) == len(panel):
                s = pd.Series(s.to_numpy(dtype=float), index=panel.index)   # positional align (e.g. reset_index)
            else:
                raise ValueError("score() returned a Series whose index does not match the price index")
        cols.append(s.reindex(panel.index).to_numpy(dtype=float))
    M = np.column_stack(cols)                     # T x N
    M = np.where(np.isnan(M), -np.inf, M)         # warm-up NaN / div-by-zero -> not selectable
    M[np.isposinf(M)] = np.finfo(float).max       # +inf ("maximal") stays selectable, not inverted to -inf
    return M


def top_k_weights(scores: np.ndarray, k: int = 1) -> np.ndarray:
    """T x N weights: EQUAL-WEIGHT (1/k each) the k highest-scoring names per bar.
    k=1 recovers one-hot 'pick the single best'. A row with fewer than k finite
    (selectable) names weights the available ones at 1/k each (partly in cash); an
    all -inf row -> cash. Ties resolve by column order — symmetric across names."""
    T, N = scores.shape
    k = int(min(max(k, 1), N))
    W = np.zeros((T, N), dtype=float)
    S = np.where(np.isfinite(scores), scores, -np.inf)
    top = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]   # T x k: the k highest per row
    rows = np.repeat(np.arange(T), k)
    cols = top.ravel()
    sel = S[rows, cols] > -np.inf                          # weight only finite (live) picks
    W[rows[sel], cols[sel]] = 1.0 / k
    return W


def one_hot_weights(scores: np.ndarray) -> np.ndarray:
    """Hold the single highest-scoring name each bar (k=1)."""
    return top_k_weights(scores, 1)


# ---- portfolio return construction -----------------------------------------

def _returns_matrix(panel: pd.DataFrame) -> np.ndarray:
    """T x N simple returns of each name (row 0 = 0)."""
    return panel.pct_change().fillna(0.0).to_numpy()


def portfolio_returns(weights: np.ndarray, rets: np.ndarray,
                      cost_per_turn: float = 0.0005) -> np.ndarray:
    """Weights (T x N) -> net portfolio return stream, with a 1-bar execution lag
    and cost on L1 turnover. Decide at t-1, hold at t: the weights are shifted one
    bar before they earn returns (identical convention to backtest.run_backtest)."""
    held = np.zeros_like(weights)
    held[1:] = weights[:-1]                       # target weights held over day t (decide t-1)
    gross = np.sum(held * rets, axis=1)
    # DRIFT-AWARE turnover: between rebalances the held weights drift with returns, so the
    # cost of moving to today's target is measured against YESTERDAY'S DRIFTED weights, not
    # its target. For a one-hot hold of the same name this is 0 (no rebalancing); for a
    # multi-name basket (incl. the equal-weight benchmark) it correctly charges the small
    # daily cost of resetting drifted weights — so the benchmark is not costless either.
    with np.errstate(invalid="ignore", divide="ignore"):
        drifted = held * (1.0 + rets) / (1.0 + gross)[:, None]      # actual weights at END of day t
    drifted = np.nan_to_num(drifted, nan=0.0, posinf=0.0, neginf=0.0)
    pre = np.zeros_like(held)
    pre[1:] = drifted[:-1]                          # weights actually held at the START of day t
    turnover = np.sum(np.abs(held - pre), axis=1)
    return gross - cost_per_turn * turnover


def equalweight_returns(rets: np.ndarray, cost_per_turn: float = 0.0005) -> np.ndarray:
    """Benchmark: 1/N target weights, daily-rebalanced. Routed through portfolio_returns,
    which now charges the (small) drift-rebalancing turnover — so the benchmark pays a
    realistic cost, not zero."""
    T, N = rets.shape
    W = np.full((T, N), 1.0 / N)
    return portfolio_returns(W, rets, cost_per_turn)


# ---- fitness: fit params, cross-validate on active return vs equal-weight ---

def _active(score_fn, panel, rets, tools, p, cost, bench_ret, k=1, feats=None):
    W = top_k_weights(score_matrix(score_fn, panel, tools, p, feats), k)
    return portfolio_returns(W, rets, cost) - bench_ret


def fit_on_mask(score_fn, space, panel, rets, tools, mask, bench_ret,
                budget=200, cost=0.0005, seed=0, objective="sharpe", min_sharpe=0.8, k=1,
                feats=None):
    """Tune params to MAXIMIZE the active (vs equal-weight) objective over `mask`."""
    from scipy.optimize import differential_evolution
    names, bounds, kinds, extra = P.parse_space(space)
    if not bounds:
        return {}

    def neg(x):
        pp = P.decode(x, names, kinds, extra)
        try:
            r = _active(score_fn, panel, rets, tools, pp, cost, bench_ret, k, feats)
            return -bt._score_arr(r[mask], objective, min_sharpe)
        except Exception:
            return 10.0

    popsize = 8
    maxiter = max(1, int(round(budget / (popsize * len(bounds)))) - 1)
    res = differential_evolution(neg, bounds, popsize=popsize, maxiter=maxiter, seed=seed,
                                 polish=False, tol=0.01, mutation=(0.5, 1.0), recombination=0.7)
    return P.decode(res.x, names, kinds, extra)


def ccv_median_oos(score_fn, space, panel, rets, tools, splits, bench_ret,
                   budget=200, cost=0.0005, seed=0, objective="sharpe", min_sharpe=0.8, k=1,
                   feats=None) -> dict:
    """Fit on each split's train quarters, score OOS on its test quarters; fitness =
    median per-split OOS active objective (vs equal-weight)."""
    rows = []
    for i, (train_mask, test_mask) in enumerate(splits):
        p = fit_on_mask(score_fn, space, panel, rets, tools, train_mask, bench_ret,
                        budget, cost, seed + i, objective, min_sharpe, k, feats)
        r = _active(score_fn, panel, rets, tools, p, cost, bench_ret, k, feats)
        rt = r[test_mask]
        rows.append((bt._score_arr(rt, objective, min_sharpe), bt._sharpe_arr(rt),
                     bt._ann_return_arr(rt)))
    rows = np.asarray(rows, dtype=float)
    score, shp, ret = rows[:, 0], rows[:, 1], rows[:, 2]
    return {
        "median_oos": float(np.median(score)), "mean_oos": float(np.mean(score)),
        "std_oos": float(np.std(score)), "frac_positive": float(np.mean(score > 0)),
        "median_sharpe": float(np.median(shp)), "median_return": float(np.median(ret)),
        "objective": objective, "n_splits": len(splits),
    }


def fit_full(score_fn, space, panel, rets, tools, bench_ret,
             budget=400, cost=0.0005, seed=0, objective="sharpe", min_sharpe=0.8, k=1,
             feats=None) -> dict:
    mask = np.ones(len(panel), dtype=bool)
    return fit_on_mask(score_fn, space, panel, rets, tools, mask, bench_ret,
                       budget, cost, seed, objective, min_sharpe, k, feats)


# ---- selection-aware shift-the-signal null (cross-sectional) ---------------

def shift_null_pvalue(score_fn, space, panel, rets, tools, bench_ret,
                      n_configs=64, shift_offsets=None, n_shifts=50, cost=0.0005,
                      seed=0, objective="sharpe", min_sharpe=0.8, k=1, feats=None) -> dict:
    """Cross-sectional analogue of backtest.shift_null_pvalue. For each sampled
    config we build the one-hot selection matrix and score its ACTIVE return; the
    null circularly SHIFTS the whole selection matrix in time (destroying the
    signal<->return alignment while keeping each name's return path and the
    selection's turnover/exposure profile). Selection-aware: real and every null
    draw take the MAX over the same configs. p = fraction of null maxima >= real."""
    n = len(panel)
    configs = bt.sample_configs(space, n_configs, seed)
    W_list = []
    for p in configs:
        try:
            W = top_k_weights(score_matrix(score_fn, panel, tools, p, feats), k)
        except Exception:
            W = np.zeros((n, rets.shape[1]))
        W_list.append(W)

    def maxscore(shift):
        best = float("-inf")
        for W in W_list:
            WW = W if shift == 0 else np.roll(W, int(shift), axis=0)
            r = portfolio_returns(WW, rets, cost) - bench_ret
            s = bt._score_arr(r, objective, min_sharpe)
            if s > best:
                best = s
        return best

    real = maxscore(0)
    if shift_offsets is None:
        shift_offsets = bt.make_shift_offsets(n, n_shifts, seed + 1)
    null = np.array([maxscore(k) for k in shift_offsets], dtype=float)
    if null.size == 0:
        return {"pvalue": float("nan"), "real": float(real), "n_configs": len(configs), "n_shifts": 0}
    b, m = int((null >= real).sum()), int(len(null))
    return {"pvalue": float((b + 1) / (m + 1)), "real": float(real),
            "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "n_configs": len(configs), "n_shifts": int(len(null))}


# ---- null-max bar (replaces the invalid shift-the-signal skill gate) --------

def rademacher_bar(candidates, panel, rets, tools, splits, bench_ret, n_scramble=30,
                   cost=0.0005, seed=0, objective="sharpe", min_sharpe=0.8, k=1, feats=None,
                   fit_budget=60, quantile=0.95):
    """Empirical Rademacher / null-max bar: the highest in-sample CV score the SEARCH
    CLASS can extract from data with the signal destroyed. Each scramble SIGN-FLIPS every
    return to ±1 (destroying any feature->return relationship), then FITS each candidate's
    parameters to the scrambled returns and takes the BEST-in-class CV score. Repeated,
    this is the distribution of best-noise-fits; mean and q95 are the bar. A champion has
    certifiable in-sample skill iff its real CV score clears the bar with headroom
    H = CV^2 / bar_q^2 (>= ~2.5 = genuine-but-weak).

    CRITICAL: `candidates` must span the SAME class the search may emit — a bar over a
    smaller class under-estimates and lets overfits through. Pass (score_fn, space) pairs
    representing the allowed strategy family (rich for an unconstrained LLM, linear-only
    for a constrained one)."""
    rng = np.random.default_rng(seed)
    T, N = rets.shape
    bars = []
    for _ in range(n_scramble):
        rscr = rets * rng.choice([-1.0, 1.0], size=(T, N))       # sign-flip -> pure noise, magnitude kept
        bscr = equalweight_returns(rscr, cost) if bench_ret is not None else None
        best = -np.inf
        for score_fn, space in candidates:
            try:
                d = ccv_median_oos(score_fn, space, panel, rscr, tools, splits, bscr,
                                   budget=fit_budget, cost=cost, k=k, feats=feats)
                best = max(best, d["median_oos"])                # best-in-class fit to this scramble
            except Exception:
                pass
        bars.append(best)
    bars = np.array(bars, dtype=float)
    q = float(np.quantile(bars, quantile))
    return {"bar_mean": float(bars.mean()), "bar_q": q, "bar_max": float(bars.max()),
            "quantile": quantile, "n_scramble": int(n_scramble)}


# ---- VC (Vapnik) worst-case bar and Deflated Sharpe — reported alongside Rademacher ----

def vc_bar(vc_dim, n_obs, years, periods_per_year=PERIODS_PER_YEAR):
    """Analytical worst-case null-max bar from VC theory (the article's Bar 1): the highest
    ANNUALIZED Sharpe a strategy class of VC dimension `vc_dim` can reach by pure luck over
    `n_obs` samples / `years` years,  bar ≈ sqrt( 2·h·log(e·m/h) / Y ).

    Returns +inf when vc_dim is infinite (an unbounded class, e.g. an arbitrary-code LLM
    search) — the rigorous statement that no capacity bar defends such a class. A linear
    model on d features has h ≈ d+1; a single-feature pick has h ≈ (n_features)+1."""
    h = float(vc_dim)
    if not np.isfinite(h) or h <= 0:
        return float("inf")
    m = float(n_obs)
    return float(np.sqrt(2.0 * h * np.log(np.e * m / h) / float(years)))


def deflated_sharpe(champ_ret, candidate_sharpes, n_trials=None):
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014): the probability the champion's
    Sharpe reflects real skill after correcting for selection over `n_trials`, for the
    return stream's skew/kurtosis, and for sample length. `champ_ret` = the champion's net
    return stream; `candidate_sharpes` = the per-observation Sharpes of ALL strategies the
    search evaluated (used to estimate the trial-Sharpe dispersion and the trial count).

    Returned as {"dsr", "sr0", "n_trials"}. CAVEAT (state it in the paper): DSR keys on the
    trial count, which is ill-defined for exhaustive/automated search — capacity-based bars
    (VC / Rademacher) are the right invariant. We report DSR only to preempt the standard
    reviewer ask."""
    from scipy.stats import norm, skew as _skew, kurtosis as _kurt
    r = np.asarray(champ_ret, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3 or r.std() == 0:
        return {"dsr": float("nan"), "sr0": float("nan"), "n_trials": 0}
    sr = r.mean() / r.std()                                   # PER-OBSERVATION Sharpe (not annualized)
    g3 = float(_skew(r)); g4 = float(_kurt(r, fisher=False))  # skew, (non-excess) kurtosis
    cand = np.asarray(candidate_sharpes, dtype=float); cand = cand[np.isfinite(cand)]
    N = int(n_trials) if n_trials else max(2, cand.size)
    sr_std = float(cand.std()) if cand.size > 1 else abs(sr)  # dispersion of trial Sharpes
    gamma = 0.5772156649015329                                # Euler–Mascheroni
    sr0 = sr_std * ((1 - gamma) * norm.ppf(1 - 1.0 / N) + gamma * norm.ppf(1 - 1.0 / (N * np.e)))
    denom = np.sqrt(max(1e-12, 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2))
    z = (sr - sr0) * np.sqrt(max(1, r.size - 1)) / denom
    return {"dsr": float(norm.cdf(z)), "sr0": float(sr0), "n_trials": int(N)}


# ---- random-picker null (the key control for this experiment) ---------------

def random_picker_distribution(rets: np.ndarray, bench_ret, n_draws=1000, cost=0.0005,
                               seed=0, objective="sharpe", min_sharpe=0.8, k=1) -> dict:
    """Distribution of the active objective for a RANDOM picker: each draw holds a
    uniformly random k-of-N subset each bar, equal-weighted (same turnover profile as
    a real per-bar top-k selector). This is the control that isolates SELECTION SKILL
    from the mere fact that concentrating in k names is a higher-variance bet — a
    strategy must beat this distribution, not just equal-weight. Vectorized: a random
    score matrix run through top_k_weights gives a uniform random k-subset per bar."""
    T, N = rets.shape
    rng = np.random.default_rng(seed)
    scores = np.empty(n_draws)
    for d in range(n_draws):
        W = top_k_weights(rng.random((T, N)), k)
        r = portfolio_returns(W, rets, cost) - bench_ret
        scores[d] = bt._score_arr(r, objective, min_sharpe)
    return {"mean": float(scores.mean()), "std": float(scores.std()),
            "q95": float(np.quantile(scores, 0.95)), "q99": float(np.quantile(scores, 0.99)),
            "n_draws": int(n_draws)}
