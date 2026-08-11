"""
Regression tests for the selection-aware shift-the-signal skill test (backtest.py).

Locks in the calibration invariants that are easy to break:
  - buy & hold (a constant position) lands EXACTLY on the bar (p = 1.0),
  - perfect look-ahead timing hits the p-value FLOOR 1/(m+1) and never 0
    (note: shift(-1), because _score_position applies its own +1 execution lag),
  - shift offsets exclude near-identity shifts.

Run: python test_shift_gate.py     (or pytest)
"""
import numpy as np
import pandas as pd
import backtest as bt


def _pool(n=1200, seed=1):
    rng = np.random.default_rng(seed)
    ret = rng.normal(1.2 / 252, 1 / np.sqrt(252), n)          # NVDA-like drift
    idx = pd.bdate_range("2017-01-01", periods=n)
    return pd.DataFrame({"close": 100.0 * np.cumprod(1.0 + ret)}, index=idx)


def test_buy_and_hold_lands_on_the_bar():
    pool = _pool()

    def bh(data, tools, p):
        return data["close"] * 0 + 1.0                         # constant fully-long position

    r = bt.shift_null_pvalue(bh, {}, pool, None, n_configs=8,
                             cost=5e-4, seed=0, objective="return", min_sharpe=0.8)
    assert abs(r["pvalue"] - 1.0) < 1e-12                      # invariant under circular shift
    assert abs(r["real"] - r["null_q95"]) < 1e-9              # real == null exactly


def test_perfect_timing_hits_the_floor():
    pool = _pool()
    ar = pool["close"].pct_change().fillna(0.0).to_numpy()
    perfect = np.sign(np.roll(ar, -1))                         # shift(-1): counter the +1 lag
    offs = bt.make_shift_offsets(len(ar), 200, seed=1)
    null = np.array([bt._score_position(np.roll(perfect, int(k)), ar, 5e-4, "return", 0.8)
                     for k in offs])
    real = bt._score_position(perfect, ar, 5e-4, "return", 0.8)
    b, m = int((null >= real).sum()), len(null)
    assert b == 0                                              # nothing beats perfect timing
    assert abs((b + 1) / (m + 1) - 1.0 / (m + 1)) < 1e-12     # exactly the floor, never 0


def test_shift_offsets_exclude_near_identity():
    n, L = 1200, 250
    offs = bt.make_shift_offsets(n, 500, seed=2, min_gap=L)
    assert offs.min() >= L
    assert offs.max() <= n - L


if __name__ == "__main__":
    test_buy_and_hold_lands_on_the_bar()
    test_perfect_timing_hits_the_floor()
    test_shift_offsets_exclude_near_identity()
    print("all skill-gate regression tests passed")
