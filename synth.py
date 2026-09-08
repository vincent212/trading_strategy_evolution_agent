"""
Synthetic feature generator for the FRAMEWORK-VALIDATION experiment.

We validate the LATSS loop with controlled ground truth instead of hoping real
markets are predictable. The evolved scorer sees only OPAQUELY-NAMED per-name
features (feat_a..feat_d) — the LLM is NOT told which, if any, predicts:

  * Negative control (predictor=False): ALL four features are pure (persistent)
    noise with zero forward-return correlation. The loop should find nothing that
    survives the holdout / null.
  * Positive control (predictor=True): one hidden slot carries, for ONE target
    name (GOOGL), a signal = (its forward-horizon return, standardized) + noise,
    scaled so an argmax-selector on that feature concentrates on the target and
    realizes an annualized Sharpe of ~1. Every other name/feature is noise. The
    loop SHOULD discover that feature and exploit it (i.e. end up trading GOOGL).

The signal deliberately uses future returns — it is a synthetic oracle-with-noise
for validation, NOT a tradable signal, and is labelled as such.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_NAMES = ["feat_a", "feat_b", "feat_c", "feat_d"]   # opaque; no hint which predicts
PREDICTOR_SLOT = "feat_c"                                   # hidden: the signal-carrying slot in the +control


def _ar1_noise(T: int, N: int, seed: int, phi: float = 0.9) -> np.ndarray:
    """Persistent (AR(1)) unit-variance noise — looks like a plausible signal, predicts nothing."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((T, N))
    x = np.empty((T, N))
    x[0] = e[0]
    s = np.sqrt(1.0 - phi ** 2)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + s * e[t]
    return x


def make_features(panel: pd.DataFrame, predictor: bool = False, target: str = "GOOGL",
                  horizon: int = 21, rho: float = 0.5, mag: float = 1.0, base: float = 4.0,
                  seed: int = 0) -> dict:
    """Return {feature_name: T x N DataFrame} aligned to panel. See module docstring.
    rho = signal-to-noise of the injected predictor; mag = magnitude that makes the
    target's predictor dominate the cross-sectional argmax (so selection concentrates
    on the target). Both are calibrated (see __main__) for Sharpe ~1 + concentration."""
    rho = float(min(max(rho, 0.0), 1.0))                  # SNR in [0,1]; rho>1 -> sqrt(1-rho^2) NaN kills the signal
    idx = panel.index
    cols = list(panel.columns)
    T, N = len(idx), len(cols)
    feats = {}
    for i, fname in enumerate(FEATURE_NAMES):
        noise = _ar1_noise(T, N, seed * 100 + i)
        df = pd.DataFrame(noise, index=idx, columns=cols)
        if predictor and fname == PREDICTOR_SLOT:
            # ORACLE (look-ahead by design): the target's forward EXCESS return over the universe —
            # i.e. how much GOOGL will out/under-perform the equal-weight basket. This is what a
            # selector must predict to BEAT equal-weight (predicting absolute return can't, since a
            # single name doesn't beat a diversified basket risk-adjusted).
            fwd_all = panel.shift(-horizon) / panel - 1.0
            fwd = fwd_all[target] - fwd_all.mean(axis=1)                   # forward excess return of the target
            mu, sd = fwd.mean(), fwd.std() + 1e-9
            z = ((fwd - mu) / sd).fillna(0.0).to_numpy()
            tcol = cols.index(target)
            # base (>0) keeps the target the top pick most bars (concentrate); rho*z injects the
            # timing signal that shaves its worst-predicted stretches; the rest is noise.
            sig = base + rho * z + np.sqrt(1.0 - rho ** 2) * noise[:, tcol]
            df[target] = mag * sig                                        # dominates the unit-variance noise features
        feats[fname] = df
    return feats


if __name__ == "__main__":
    # calibrate the positive control: sweep (base, rho) for GOOGL concentration + active Sharpe
    import data_mag7, backtest as bt, backtest_xs as xs
    panel = data_mag7.get_panel(); rets = xs._returns_matrix(panel); idx = panel.index
    cols = list(panel.columns); tgt = cols.index("GOOGL")
    bh = xs.portfolio_returns(np.tile(np.eye(len(cols))[tgt], (len(idx), 1)), rets, 0.0005)
    print(f"reference: GOOGL buy&hold Sharpe {bt._sharpe_arr(bh):+.2f}  ann.ret {bt._ann_return_arr(bh):+.0%}\n")
    for base in (2.0, 4.0, 6.0):
        for rho in (0.3, 0.6):
            f = make_features(panel, predictor=True, rho=rho, base=base, seed=0)
            W = xs.top_k_weights(f[PREDICTOR_SLOT].to_numpy(), 1)          # score = the predictor feature
            r = xs.portfolio_returns(W, rets, 0.0005)
            frac = float((W.argmax(1) == tgt).mean())
            m26 = idx.year == 2026
            print(f"base={base} rho={rho}: Sharpe {bt._sharpe_arr(r):+.2f} ann.ret {bt._ann_return_arr(r):+.0%} "
                  f"GOOGL-held {frac:.0%} | 2026 Sharpe {bt._sharpe_arr(r[m26]):+.2f} ret {tot26 if (tot26:=float(np.prod(1+r[m26])-1)) else 0:+.0%}")
