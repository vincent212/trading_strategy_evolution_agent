"""
Orchestration. Searches for a strategy on a ticker using pre-holdout data, gates the
champion with the null-max bar, and measures it once on the held-out current year.

Usable as a CLI (`python run.py --ticker NVDA`) or a call (`from run import run_search`).
"""
from __future__ import annotations
import os
import json
import argparse
from datetime import datetime, timezone

import numpy as np

import data as data_mod
import backtest as bt
import evolve as evolve_mod
from strategy_seed import SEED_CODE


def run_search(*, ticker="NVDA", start="2017-01-01", end=None, iterations=300,
               n_islands=4, k_parents=2, model="claude-haiku-4-5", cost=0.0005,
               n_splits=100, fit_budget=200, champion_budget=400, null_sims=10,
               headroom=1.5, holdout_year=2026, reset_every=50, jobs=1, seed=0,
               refresh_data=False, out_dir=None, log=print) -> dict:
    np.random.seed(seed)
    from anthropic import Anthropic
    client = Anthropic()                        # reads ANTHROPIC_API_KEY

    full = data_mod.get_data(ticker=ticker, start=start, end=end, refresh=refresh_data)
    pool = full[full.index.year < holdout_year].copy()
    hold = full[full.index.year >= holdout_year].copy()
    if len(pool) < 300:
        raise RuntimeError(f"insufficient pre-{holdout_year} data for {ticker}: {len(pool)} rows")
    # Hard guarantee: the holdout year must never reach the evolution/fit/CV/gate.
    # Only `pool` is passed to those; `full` is used solely for the final measurement.
    assert int((pool.index.year >= holdout_year).sum()) == 0, \
        "holdout year leaked into the training pool"
    log(f"{ticker}: pool {pool.index[0].date()}..{pool.index[-1].date()} ({len(pool)} rows) | "
        f"holdout {holdout_year}: {len(hold)} rows")

    splits = bt.make_quarter_splits(pool.index, n_splits=n_splits, train_frac=0.75, seed=seed)

    ev = evolve_mod.Evolver(pool, splits, client, model=model, n_islands=n_islands,
                            k_parents=k_parents, fit_budget=fit_budget, cost=cost,
                            jobs=jobs, seed=seed, log=log)
    ev.seed(SEED_CODE)
    best = ev.run(iterations, reset_every=reset_every)
    if best is None:
        raise RuntimeError("no strategy survived the search")

    tools = ev.tools
    strat, space = evolve_mod.compile_strategy(best.code)
    median_oos = float(best.diagnostics["median_oos"])

    # null-max bar: best median-OOS the same search extracts from sign-flipped noise
    log("computing null-max bar ...")
    nb = bt.null_max_bar_ccv(strat, space, pool, tools, splits, budget=fit_budget,
                             cost=cost, n_sims=null_sims, seed=seed, jobs=jobs)
    bar = float(nb["bar"])
    passes = (median_oos > 0.0) and (median_oos >= headroom * max(bar, 0.0))

    # final: fit champion on all pool, measure once on the held-out year
    p_full = bt.fit_full(strat, space, pool, tools, budget=champion_budget, cost=cost, seed=seed)
    hold_sharpe = None
    if len(hold) > 20:
        full_ret = bt.run_backtest(strat(full, tools, p_full), full["close"], cost)
        hold_mask = (full.index.year >= holdout_year)
        hold_sharpe = bt._sharpe_arr(full_ret.to_numpy()[np.asarray(hold_mask)])

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": dict(ticker=ticker, start=start, iterations=iterations, model=model,
                       n_islands=n_islands, n_splits=n_splits, fit_budget=fit_budget,
                       cost=cost, headroom=headroom, holdout_year=holdout_year),
        "search": dict(evaluated=ev.n_evaluated, rejected=ev.n_rejected,
                       population=len(ev.db.all_programs())),
        "champion": {
            "code": best.code,
            "fitted_params_full": p_full,
            "median_oos_sharpe": median_oos,
            "mean_oos_sharpe": float(best.diagnostics.get("mean_oos", float("nan"))),
            "oos_std": float(best.diagnostics.get("std_oos", float("nan"))),
            "frac_positive_splits": float(best.diagnostics.get("frac_positive", float("nan"))),
            "holdout_year_sharpe": hold_sharpe,
        },
        "null_max_bar": {
            "bar": bar,
            "headroom_required": headroom,
            "mean_noise_median": float(nb["mean_noise_median"]),
            "PASSES": bool(passes),
        },
    }
    _print_report(result, log)

    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"run_{data_mod._safe(ticker)}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log(f"saved -> {path}")
    result["_path"] = path
    return result


def _print_report(r, log):
    c, g = r["champion"], r["null_max_bar"]
    log("\n" + "=" * 70)
    log(f"CHAMPION  ({r['config']['ticker']})")
    log("=" * 70)
    log(c["code"].strip())
    log("-" * 70)
    log(f"fitted params (full pool) : {c['fitted_params_full']}")
    log(f"median OOS Sharpe (CV)    : {c['median_oos_sharpe']:.3f}  "
        f"(mean {c['mean_oos_sharpe']:.3f}, std {c['oos_std']:.3f}, "
        f"positive splits {c['frac_positive_splits']:.0%})")
    hs = c["holdout_year_sharpe"]
    log(f"holdout {r['config']['holdout_year']} Sharpe      : "
        + ("n/a" if hs is None else f"{hs:.3f}") + "   (measured once, never searched)")
    log("-" * 70)
    log(f"null-max bar              : {g['bar']:.3f}  "
        f"(noise-median mean {g['mean_noise_median']:.3f})")
    log(f"verdict                   : "
        + ("PASS" if g["PASSES"] else "REJECT")
        + f"  (need median OOS >= {g['headroom_required']} x bar)")
    log("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Evolve a trainable trading strategy for a ticker")
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--islands", type=int, default=4)
    ap.add_argument("--parents", type=int, default=2)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--fit-budget", type=int, default=200)
    ap.add_argument("--champion-budget", type=int, default=400)
    ap.add_argument("--null-sims", type=int, default=10)
    ap.add_argument("--headroom", type=float, default=1.5)
    ap.add_argument("--holdout-year", type=int, default=2026)
    ap.add_argument("--reset-every", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh-data", action="store_true")
    a = ap.parse_args()
    run_search(ticker=a.ticker, start=a.start, end=a.end, iterations=a.iterations,
               n_islands=a.islands, k_parents=a.parents, model=a.model, cost=a.cost,
               n_splits=a.splits, fit_budget=a.fit_budget, champion_budget=a.champion_budget,
               null_sims=a.null_sims, headroom=a.headroom, holdout_year=a.holdout_year,
               reset_every=a.reset_every, jobs=a.jobs, seed=a.seed,
               refresh_data=a.refresh_data)


if __name__ == "__main__":
    main()
