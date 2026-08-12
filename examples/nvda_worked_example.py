"""Worked example — the NVDA champion from a 50/50-CV, leverage-2x, skill-gated run.

Reproduces the in-sample (2017-2024) and sealed-holdout (2025-2026) comparison of the
evolved champion against buy-and-hold, on every metric that matters: total return, Sharpe,
max drawdown, MAR, and ACTIVE Sharpe (= Sharpe of strategy-minus-buy&hold, the fitness).

The point of the example: the objective was "beat buy-and-hold on RETURN while keeping a
healthy standalone Sharpe" — NOT "beat buy-and-hold's Sharpe." The champion does exactly
that: higher return than B&H in every year, standalone Sharpe ~0.85-1.6 (comfortably above
0.8), positive ACTIVE Sharpe out of sample — but its 1.48x leverage means it does NOT beat
B&H on standalone Sharpe or MAR out of sample. Beating a benchmark on return is not the same
as dominating it on risk. See the README "Worked example" section.

Run from the repo root:   python examples/nvda_worked_example.py
Set your own transaction cost:   python examples/nvda_worked_example.py --cost 0.001   (0 = frictionless)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

import alpha_tools
import backtest as bt
import data as data_mod
import evolve as evolve_mod

# --- match the run's config exactly (see README) ---
alpha_tools.MAX_LEVERAGE = 2.0          # --max-leverage 2
bt.STOP_SLIP = 0.001
DEFAULT_COST = 0.0005                    # 5 bps per unit of turnover; override with --cost
HOLDOUT_YEAR = 2025                     # 2025-2026 sealed; only 2017-2024 was searched/fit
tools = alpha_tools

# The evolved champion (structure written by the LLM; parameters fit by the optimizer).
# A rolling percentile-rank / vol-regime switch: in a CALM regime lever by position-in-range
# (1.0 / 1.7 / 2.0 as price climbs its own range); in a WILD regime short only an ESTABLISHED
# breakdown (price stuck at range-bottom for the whole `look_n` window, not a one-bar wick).
CHAMPION = """
def param_space():
    return {
        "rank_n": ("int", 40, 150),
        "hi_p":   ("float", 0.45, 0.75),
        "top_p":  ("float", 0.75, 0.95),
        "lo_p":   ("float", 0.15, 0.45),
        "vol_n":  ("int", 20, 50),
        "vol_hi": ("float", 0.4, 1.5),
        "look_n": ("int", 3, 12),
    }

def strategy(data, tools, p):
    close = data["close"]
    r    = tools.pctile_rank(close, p["rank_n"])
    wild = (tools.vol_regime(close, p["vol_n"], -0.5, p["vol_hi"]) < 0).astype(float)
    high = (r > p["hi_p"]).astype(float)
    top  = (r > p["top_p"]).astype(float)
    low  = (r < p["lo_p"]).astype(float)
    persist = (tools.sma(low, p["look_n"]) > 0.8).astype(float)   # established breakdown, not a wick
    calm_leg = 1.0 + 0.7 * high + 0.3 * top                       # 1.0 / 1.7 / 2.0 up the range
    wild_leg = 1.0 - 1.8 * persist                                # short only a persistent range-bottom
    sig  = (1.0 - wild) * calm_leg + wild * wild_leg
    return tools.clip_signal(sig)
"""

# Parameters the optimizer fit on the full 2017-2024 pool (from the run's finalization log).
FITTED_PARAMS = {
    "rank_n": 143, "hi_p": 0.46192520808937604, "top_p": 0.8781362265159486,
    "lo_p": 0.1809200689025941, "vol_n": 30, "vol_hi": 1.1700869415414297, "look_n": 9,
}


def _total(x):
    x = x[np.isfinite(x)]
    return float(np.prod(1.0 + x) - 1.0)


def main(cost=DEFAULT_COST):
    strat, _ = evolve_mod.compile_strategy(CHAMPION)
    full = data_mod.get_data(ticker="NVDA", start="2017-01-01", end=None, refresh=False)
    years = np.asarray(full.index.year)

    strat_ret = bt.run_backtest(strat(full, tools, FITTED_PARAMS), full, cost).to_numpy()
    bh_ret = bt.run_backtest(pd.Series(1.0, index=full.index), full["close"], cost).to_numpy()
    print(f"transaction cost: {cost:.4%} per unit turnover\n")

    def row(mask, label):
        s, b = strat_ret[mask], bh_ret[mask]
        print(f"  {label:14s} | strat {_total(s)*100:+10.1f}% Sh {bt._sharpe_arr(s):+.2f} "
              f"DD {bt._max_drawdown_arr(s)*100:4.0f}% MAR {bt._mar_arr(s):5.2f}"
              f"  | B&H {_total(b)*100:+9.1f}% Sh {bt._sharpe_arr(b):+.2f} "
              f"DD {bt._max_drawdown_arr(b)*100:4.0f}% MAR {bt._mar_arr(b):5.2f}"
              f"  | active-Sh {bt._sharpe_arr(s - b):+.2f}")

    print("IN-SAMPLE (searched & fit on 2017-2024)")
    row(years < HOLDOUT_YEAR, "FULL POOL")
    for y in range(2017, HOLDOUT_YEAR):
        row(years == y, str(y))

    print("\nOUT-OF-SAMPLE (sealed holdout — measured once, never searched)")
    for y in sorted(set(years[years >= HOLDOUT_YEAR].tolist())):
        row(years == y, str(y))
    row(years >= HOLDOUT_YEAR, "2025+2026")

    print("\nactive-Sh = Sharpe of (strategy - buy&hold) daily returns = the fitness (information ratio).")
    print("Reading: beats B&H on RETURN every year and on ACTIVE Sharpe out of sample, but its 1.48x")
    print("leverage makes standalone Sharpe/MAR out of sample WORSE than simply holding NVDA.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost", type=float, default=DEFAULT_COST,
                    help="transaction cost per unit of turnover (default 0.0005 = 5 bps; 0 = frictionless)")
    main(cost=ap.parse_args().cost)
