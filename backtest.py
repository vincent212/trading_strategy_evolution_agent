"""
Evaluation engine.

Three layers, from cheapest to strictest:

1. run_backtest / sharpe        — turn a signal into a net-of-cost return stream
                                  and score it. Applies a 1-bar execution lag.

2. block_cv_score               — the objective the evolutionary loop optimizes.
                                  Splits the TRAIN period into K contiguous blocks
                                  (with an embargo gap) and takes the MEDIAN Sharpe
                                  across blocks, penalized for turnover and for
                                  trading too rarely. A strategy that only works in
                                  one window scores badly here. This is what the LLM
                                  sees — never a single-window in-sample Sharpe.

3. rademacher_null_max_bar /    — the accept/reject GATE, applied once at the end to
   vc_null_max_bar                the best survivor. The "null-max bar" is CAPACITY-
                                  based (statistical learning theory), not trials-
                                  based: it measures the best Sharpe the evolved
                                  strategy *population* can extract from sign-flipped
                                  (pure-noise) returns. If the best real in-sample
                                  Sharpe doesn't clear that bar with headroom, the
                                  edge is indistinguishable from noise-fitting.
                                  (deflated_sharpe_ratio is kept only as a trials-
                                  based secondary cross-check — see the note there.)

The held-out TEST period is never used by layers 1-2; run_final_report scores the
survivor on it as the honest out-of-sample check.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

PERIODS_PER_YEAR = 252
EULER_MASCHERONI = 0.5772156649


# ---- core backtest ----------------------------------------------------------

def run_backtest(signal: pd.Series, close: pd.Series,
                 cost_per_turn: float = 0.0005) -> pd.Series:
    """
    Convert a raw signal into a net daily return stream.

    Execution lag: position on day t is the signal from day t-1 (.shift(1)) — you
    decide at yesterday's close, you hold today. Costs charged on |change in position|.
    """
    asset_ret = close.pct_change().fillna(0.0)
    pos = signal.reindex(close.index)
    pos = pos.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    pos = pos.shift(1).fillna(0.0)                      # <-- the execution lag
    turnover = pos.diff().abs().fillna(0.0)
    return pos * asset_ret - cost_per_turn * turnover


def sharpe(returns: pd.Series, annualize: bool = True) -> float:
    """Annualized Sharpe of a return stream (0 if degenerate)."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    s = r.mean() / r.std()
    return float(s * np.sqrt(PERIODS_PER_YEAR)) if annualize else float(s)


def turnover_of(signal: pd.Series, close: pd.Series) -> float:
    pos = signal.reindex(close.index).fillna(0.0).clip(-1, 1).shift(1).fillna(0.0)
    return float(pos.diff().abs().mean())


# ---- layer 2: the loop's objective -----------------------------------------

def block_cv_score(signal: pd.Series, close: pd.Series, k: int = 5,
                   embargo: int = 5, cost_per_turn: float = 0.0005,
                   turnover_penalty: float = 0.5,
                   min_avg_turnover: float = 1e-4) -> dict:
    """
    Median Sharpe across K contiguous time blocks of the (training) period, minus a
    turnover penalty. Returns a dict with the scalar `score` plus diagnostics.

    Why blocks and not classic train/test CV: the strategy has no separate `fit`
    step — the LLM bakes its parameters into the code, so *the LLM is the fitter*.
    Block-wise Sharpe measures whether the same fixed strategy holds up across
    different regimes/windows, which is the robustness signal we actually want.
    The embargo drops a few bars at each block boundary to avoid a position opened
    in one block being scored across the seam.
    """
    ret = run_backtest(signal, close, cost_per_turn)
    n = len(ret)
    if n < k * 20:
        return {"score": -9.99, "block_sharpes": [], "full_sharpe": 0.0,
                "avg_turnover": 0.0, "reason": "too few observations"}

    bounds = np.linspace(0, n, k + 1).astype(int)
    block_sharpes = []
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        lo_e = lo + (embargo if i > 0 else 0)
        seg = ret.iloc[lo_e:hi]
        block_sharpes.append(sharpe(seg))

    avg_turnover = turnover_of(signal, close)
    full_sharpe = sharpe(ret)
    median_sharpe = float(np.median(block_sharpes))

    # Penalize churn; reject strategies that essentially never trade.
    score = median_sharpe - turnover_penalty * avg_turnover
    if avg_turnover < min_avg_turnover:
        score = -9.99  # a flat / never-trading strategy is not a strategy

    return {
        "score": float(score),
        "median_sharpe": median_sharpe,
        "block_sharpes": [round(b, 3) for b in block_sharpes],
        "full_sharpe": float(full_sharpe),
        "avg_turnover": float(avg_turnover),
        "reason": "ok",
    }


# ---- layer 3: the multiple-testing gate ------------------------------------

def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark_ann: float = 0.0) -> float:
    """
    P(true Sharpe > benchmark), correcting for track-record length, skew, and
    kurtosis (Bailey & Lopez de Prado). Works in per-observation Sharpe units.
    """
    r = returns.dropna()
    n = len(r)
    if n < 3 or r.std() == 0:
        return 0.0
    sr = r.mean() / r.std()                              # per-period Sharpe
    sr_star = sr_benchmark_ann / np.sqrt(PERIODS_PER_YEAR)
    skew = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
    kurt = float(((r - r.mean()) ** 4).mean() / r.std() ** 4)
    denom = np.sqrt(1 - skew * sr + ((kurt - 1) / 4) * sr ** 2)
    if denom <= 0:
        return 0.0
    return float(norm.cdf((sr - sr_star) * np.sqrt(n - 1) / denom))


def _positions_matrix(signals, close, cost_per_turn):
    """Stack a population of signals into a fixed, lagged position matrix (n x S)
    plus a matching per-day cost matrix. These encode the *expressive capacity* of
    the strategy class and stay fixed while we scramble the returns."""
    idx = close.index
    asset_ret = close.pct_change().fillna(0.0).to_numpy()
    pos_cols, cost_cols = [], []
    for sig in signals:
        pos = (sig.reindex(idx).replace([np.inf, -np.inf], np.nan)
               .fillna(0.0).clip(-1.0, 1.0).shift(1).fillna(0.0).to_numpy())
        pos_cols.append(pos)
        cost_cols.append(cost_per_turn * np.abs(np.diff(pos, prepend=0.0)))
    return np.column_stack(pos_cols), np.column_stack(cost_cols), asset_ret


def rademacher_null_max_bar(signals, close: pd.Series, n_sims: int = 200,
                            seed: int = 0, aggregate: str = "mean",
                            cost_per_turn: float = 0.0005) -> dict:
    """
    The article's operative gate: the empirical Rademacher-complexity "null-max bar".

    The bar is the best Sharpe your *actual evolved strategy population* can achieve
    on PURE NOISE. Each strategy's real position sequence is held FIXED (that encodes
    how expressive the search is); then we repeatedly flip the sign of each day's SPY
    return -- Rademacher labels e_t in {-1,+1} -- to destroy any genuine signal, score
    every strategy on the scrambled returns, and record the MAX Sharpe across the
    population. Averaging that best-of-population over many sign-flips gives the bar.

    Capacity-based, NOT trials-based. A more expressive population fits noise better
    and so raises its own bar. Unlike the Deflated Sharpe Ratio, searching *fewer*
    candidates does not lower the bar -- there is no incentive to under-search. (DSR
    counts how many you tried; this measures what your class can express.)

    aggregate: "mean" (the article's default) or a quantile in (0,1) as a string,
    e.g. "0.95" for a stricter bar.
    """
    if len(signals) == 0:
        return {"bar": 0.0, "n_strategies": 0, "n_sims": n_sims}
    P, C, asset_ret = _positions_matrix(signals, close, cost_per_turn)
    rng = np.random.default_rng(seed)
    best = np.empty(n_sims)
    for s in range(n_sims):
        eps = rng.choice([-1.0, 1.0], size=len(asset_ret))
        pnl = P * (eps * asset_ret)[:, None] - C          # flip returns, keep positions
        mu, sd = pnl.mean(axis=0), pnl.std(axis=0)
        sr = np.where(sd > 0, mu / sd, 0.0) * np.sqrt(PERIODS_PER_YEAR)
        best[s] = sr.max()
    bar = best.mean() if aggregate == "mean" else float(np.quantile(best, float(aggregate)))
    return {
        "bar": float(bar),
        "n_strategies": len(signals),
        "n_sims": n_sims,
        "best_of_population_std": float(best.std()),
    }


def vc_null_max_bar(h: int, n_obs: int, years: float | None = None) -> float:
    """
    Analytic worst-case ("adversarial market") ceiling from VC theory:
        null_max ~= sqrt( 2h * log(e*m/h) / Y )
    with m = number of sessions and Y = years. The article treats this as an
    UNFALSIFIABLE ceiling, not an operative test.

    h = VC dimension ~ number of behaviorally distinct strategies the class can
    express. For a free-form LLM-generated program class h is not rigorously
    computable, so pass a proxy (e.g. the count of distinct strategies the loop
    produced) and read the result as illustrative.
    """
    m = int(n_obs)
    Y = float(years) if years else n_obs / PERIODS_PER_YEAR
    h = max(int(h), 1)
    return float(np.sqrt(2.0 * h * np.log(np.e * m / h) / Y))


# ---- optional secondary cross-check: the trials-based DSR the article critiques --
# Kept for comparison only. It counts how many strategies you *tried*, which rewards
# under-searching; the Rademacher bar above is the primary gate. Use as a sanity
# cross-reference, not the decision.

def expected_max_sharpe_ann(n_trials: int, sharpe_variance_ann: float) -> float:
    """False Strategy Theorem expected-max annualized Sharpe (Bailey & Lopez de Prado)."""
    if n_trials < 2 or sharpe_variance_ann <= 0:
        return 0.0
    sigma = np.sqrt(sharpe_variance_ann)
    g = EULER_MASCHERONI
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sigma * ((1 - g) * z1 + g * z2))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          sharpe_variance_ann: float) -> dict:
    """Trials-based DSR (secondary cross-check only — see note above)."""
    sr_star = expected_max_sharpe_ann(n_trials, sharpe_variance_ann)
    dsr = probabilistic_sharpe_ratio(returns, sr_benchmark_ann=sr_star)
    return {"dsr": float(dsr), "trials_null_bar_ann": float(sr_star),
            "strategy_sharpe_ann": sharpe(returns), "passes": bool(dsr > 0.95)}
