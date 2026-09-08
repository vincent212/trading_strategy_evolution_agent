"""Smoke test for the cross-sectional pick-1-of-7 plumbing (no LLM).
Loads the Mag 7, runs a few name-symmetric selector seeds through the evaluator,
and reports active Sharpe vs equal-weight plus the random-picker null bar."""
from __future__ import annotations
import numpy as np

import data_mag7
import backtest_xs as xs
import backtest as bt
import params as P
import alpha_tools as tools


def momentum_score(series, tools, p):
    return tools.roc(series, p["n"])                       # prefer strongest trailing momentum


def lowvol_score(series, tools, p):
    return -tools.realized_vol(series, p["n"])             # prefer the calmest name


def meanrev_score(series, tools, p):
    return -tools.zscore(series, p["n"])                   # prefer the most oversold name


SEEDS = {
    "momentum": (momentum_score, {"n": ("int", 20, 120)}),
    "lowvol":   (lowvol_score,   {"n": ("int", 20, 120)}),
    "meanrev":  (meanrev_score,  {"n": ("int", 5, 40)}),
}


def main():
    panel = data_mag7.get_panel()
    rets = xs._returns_matrix(panel)
    bench = xs.equalweight_returns(rets)
    print(f"panel: {list(panel.columns)}  {len(panel)} rows  "
          f"{panel.index[0].date()}..{panel.index[-1].date()}")
    print(f"equal-weight benchmark: Sharpe {bt._sharpe_arr(bench):+.2f}  "
          f"ann.ret {bt._ann_return_arr(bench):+.1%}")

    rp = xs.random_picker_distribution(rets, bench, n_draws=500, seed=0)
    print(f"random-picker null (active Sharpe vs EW): mean {rp['mean']:+.2f}  "
          f"q95 {rp['q95']:+.2f}  q99 {rp['q99']:+.2f}   <- a real strategy must clear q95")

    for name, (fn, space) in SEEDS.items():
        p = P.midpoint(space)
        r = xs._active(fn, panel, rets, tools, p, 0.0005, bench)
        print(f"  {name:9s} (midpoint {p}): active Sharpe {bt._sharpe_arr(r):+.2f}  "
              f"active ann.ret {bt._ann_return_arr(r):+.1%}")


if __name__ == "__main__":
    main()
