"""
Orchestration entry point. Runs the FunSearch loop end-to-end on SPY and applies
the null-max-bar gate to the best survivor.

Usable two ways:
  1. CLI:        python run.py --iterations 300
  2. As a call:  from run import run_search; run_search(iterations=300)

The `run_search` function returns a result dict, which is what makes this launchable
from an orchestrator / Claude subagent rather than only as a standalone script.
"""
from __future__ import annotations
import os
import json
import argparse
from datetime import datetime, timezone

import data as data_mod
import backtest as bt
import evolve as evolve_mod
from strategy_seed import SEED_CODE


def _split(df, train_frac):
    n = len(df)
    cut = int(n * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def run_search(*, ticker="SPY", iterations=300, start="2005-01-01", end=None,
               train_frac=0.7, n_islands=4, k_parents=2, model="claude-haiku-4-5",
               cost=0.0005, k_folds=5, embargo=5, mc_sims=200, headroom=1.5, seed=0,
               refresh_data=False, out_dir=None, log=print) -> dict:
    """Run the full search + gate on `ticker`. Returns a result dict (also written to out_dir)."""
    import numpy as np
    np.random.seed(seed)

    from anthropic import Anthropic
    client = Anthropic()                    # reads ANTHROPIC_API_KEY

    df = data_mod.get_data(ticker=ticker, start=start, end=end, refresh=refresh_data)
    train, test = _split(df, train_frac)
    log(f"{ticker}: {len(df)} rows | train {train.index[0].date()}..{train.index[-1].date()} "
        f"({len(train)}) | test {test.index[0].date()}..{test.index[-1].date()} ({len(test)})")

    ev = evolve_mod.Evolver(
        data_train=train, close_train=train["close"], client=client, model=model,
        n_islands=n_islands, k_parents=k_parents,
        cv_kwargs=dict(k=k_folds, embargo=embargo, cost_per_turn=cost), log=log)
    ev.seed(SEED_CODE)
    best = ev.run(iterations=iterations)
    if best is None:
        raise RuntimeError("no strategies survived — check API key / data")

    # ---- the null-max-bar gate (capacity-based, on the TRAIN population) ----
    population = ev.db.all_programs()
    signals = [p.signal for p in population]
    distinct = len({p.code for p in population})

    rade = bt.rademacher_null_max_bar(signals, train["close"], n_sims=mc_sims,
                                      seed=seed, cost_per_turn=cost)
    vc_bar = bt.vc_null_max_bar(h=distinct, n_obs=len(train))
    best_ret_train = bt.run_backtest(best.signal, train["close"], cost)
    best_sharpe_train = bt.sharpe(best_ret_train)
    headroom_ratio = best_sharpe_train / rade["bar"] if rade["bar"] > 0 else float("inf")
    passes = best_sharpe_train >= headroom * rade["bar"]

    # ---- honest out-of-sample check (never touched by search or gate) ----
    fn = evolve_mod.compile_strategy(best.code)
    test_signal = fn(test, evolve_mod.alpha_tools_module())
    test_ret = bt.run_backtest(test_signal, test["close"], cost)
    oos_sharpe = bt.sharpe(test_ret)

    # ---- secondary trials-based cross-check (the DSR the article critiques) ----
    import numpy as np
    tr_sharpes = np.array([bt.sharpe(bt.run_backtest(p.signal, train["close"], cost))
                           for p in population])
    dsr = bt.deflated_sharpe_ratio(best_ret_train, n_trials=len(population),
                                   sharpe_variance_ann=float(tr_sharpes.var()))

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": dict(ticker=ticker, iterations=iterations, model=model,
                       n_islands=n_islands, k_parents=k_parents, cost=cost,
                       k_folds=k_folds, headroom=headroom, train_frac=train_frac),
        "search": dict(evaluated=ev.n_evaluated, rejected=ev.n_rejected,
                       distinct_strategies=distinct, population=len(population)),
        "best_strategy": {
            "code": best.code,
            "train_score": best.score,
            "train_sharpe": best_sharpe_train,
            "train_block_sharpes": best.diagnostics.get("block_sharpes"),
            "oos_test_sharpe": oos_sharpe,
        },
        "null_max_bar": {
            "rademacher_bar": rade["bar"],
            "vc_ceiling": vc_bar,
            "headroom_ratio": headroom_ratio,
            "headroom_required": headroom,
            "PASSES_GATE": passes,
        },
        "secondary_dsr": dsr,
    }

    _print_report(result, log)

    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"run_{data_mod._safe(ticker)}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"\nsaved -> {path}")
    result["_path"] = path
    return result


def _print_report(r, log):
    b = r["best_strategy"]
    g = r["null_max_bar"]
    log("\n" + "=" * 68)
    log(f"BEST STRATEGY  ({r['config']['ticker']})")
    log("=" * 68)
    log(b["code"].strip())
    log("-" * 68)
    log(f"train Sharpe        : {b['train_sharpe']:.3f}")
    log(f"train block Sharpes : {b['train_block_sharpes']}")
    log(f"OOS test Sharpe     : {b['oos_test_sharpe']:.3f}   <- honest, never searched")
    log("-" * 68)
    log(f"Rademacher null-max bar : {g['rademacher_bar']:.3f}  (noise ceiling of the population)")
    log(f"VC worst-case ceiling   : {g['vc_ceiling']:.3f}  (unfalsifiable)")
    log(f"headroom ratio          : {g['headroom_ratio']:.2f}x  (need >= {g['headroom_required']}x)")
    verdict = "PASS — clears the noise bar" if g["PASSES_GATE"] else "REJECT — indistinguishable from noise"
    log(f"VERDICT                 : {verdict}")
    log(f"(secondary DSR cross-check: {r['secondary_dsr']['dsr']:.3f})")
    log("=" * 68)


def main():
    ap = argparse.ArgumentParser(description="FunSearch for stock/ETF trading strategies")
    ap.add_argument("--ticker", default="SPY", help="any Yahoo Finance symbol, e.g. AAPL, TSLA, QQQ")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--islands", type=int, default=4)
    ap.add_argument("--parents", type=int, default=2)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--mc-sims", type=int, default=200)
    ap.add_argument("--headroom", type=float, default=1.5)
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()
    run_search(ticker=args.ticker, iterations=args.iterations, start=args.start,
               end=args.end, train_frac=args.train_frac, n_islands=args.islands,
               k_parents=args.parents, model=args.model, cost=args.cost,
               k_folds=args.folds, mc_sims=args.mc_sims, headroom=args.headroom,
               refresh_data=args.refresh_data)


if __name__ == "__main__":
    main()
