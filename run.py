"""
Orchestration. Searches for a strategy on a ticker using pre-holdout data (selection by CV
median-OOS fitness), scores a shift-the-signal skill p-value per candidate, and measures the
champion once on the sealed holdout years against buy-and-hold. The skill p-value is report-only
by default; with --skill-gate it becomes a hard gate that REJECTS mutations with no timing skill.

Usable as a CLI (`python run.py --ticker NVDA`) or a call (`from run import run_search`).
"""
from __future__ import annotations
import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone

import numpy as np

import data as data_mod
import backtest as bt
import evolve as evolve_mod
from strategy_seed import SEEDS


def _make_logger(out_dir, tag):
    """Logger that writes to stdout AND out_dir/<tag>_progress.log, flushing every
    record so progress is visible live even when the run is backgrounded."""
    logger = logging.getLogger(f"funsearch.{tag}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(out_dir, f"{tag}_progress.log"))):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger.info


def run_search(*, ticker="NVDA", start="2017-01-01", end=None, iterations=300,
               model="claude-haiku-4-5", cost=0.0005,
               n_splits=100, fit_budget=200, champion_budget=400,
               holdout_year=2026, reset_every=50, jobs=1, seed=0,
               objective="sharpe", min_sharpe=0.8, vs_buyhold=True, max_leverage=1.0,
               stop_slippage=0.001, theme_file=None,
               null_gate=True, null_gate_configs=64, null_gate_shifts=50,
               skill_gate=False, skill_pmax=0.10,
               refresh_data=False, out_dir=None, log=print) -> dict:
    np.random.seed(seed)
    if skill_gate and (not null_gate or null_gate_shifts < 1):
        raise ValueError("--skill-gate needs the skill test running with >=1 shift "
                         "(keep the skill test on and --null-gate-shifts >= 1); otherwise the "
                         "p-value is NaN and the gate would silently accept everything.")
    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)
    if log is print:                            # default -> proper flushed logging
        log = _make_logger(out_dir, data_mod._safe(ticker))

    import llm
    client = llm.make_client(model)             # Anthropic, or OpenAI-compatible via LLM_BASE_URL
    log(f"llm backend: {client.backend}")

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

    log(f"objective: {objective}" + (f" (min Sharpe {min_sharpe})" if objective == "return" else "")
        + (f"  |  fitness = risk-adjusted OUTPERFORMANCE of buy-and-hold (active return)"
           if vs_buyhold else "  |  fitness = raw strategy return"))
    log(f"shift-the-signal skill p-value: " + ("ON" if null_gate else "OFF")
        + (f" (selection-aware, {null_gate_configs} configs x {null_gate_shifts} shifts, on the "
           f"active return)" if null_gate else "")
        + (f"  |  SKILL GATE: reject mutations with skill_p >= {skill_pmax:g} (no timing skill)"
           if (null_gate and skill_gate) else "  |  report-only (no skill gate)"))
    import alpha_tools
    alpha_tools.MAX_LEVERAGE = float(max_leverage)      # position cap for clip_signal + backtest
    log(f"max leverage: {max_leverage:g}x"
        + ("  (>1 lets a strategy beat buy-and-hold on RETURN by levering up in good regimes)"
           if max_leverage > 1.0 else "  (fully long/short only — cannot exceed buy-and-hold exposure)"))
    bt.STOP_SLIP = stop_slippage                            # slippage on a stopped day (fixed realism)
    log(f"intraday stop-loss: per-strategy (declare a `stop_loss` param, 0=off; optimizer tunes it); "
        f"slippage {stop_slippage:.1%} on a stopped day, no overnight stop")
    bench = bt.buyhold_returns(pool["close"], cost) if vs_buyhold else None
    import prompt as prompt_mod
    theme = prompt_mod.load_theme(theme_file)
    log(f"investment theme: {len(theme)} chars"
        + (f" from {theme_file}" if theme_file else " (theme.txt / default)"))
    ev = evolve_mod.Evolver(pool, splits, client, model=model,
                            fit_budget=fit_budget, cost=cost,
                            jobs=jobs, seed=seed, log=log,
                            objective=objective, min_sharpe=min_sharpe, vs_buyhold=vs_buyhold,
                            theme=theme, max_leverage=max_leverage,
                            null_gate=null_gate, null_gate_configs=null_gate_configs,
                            null_gate_shifts=null_gate_shifts,
                            skill_gate=skill_gate, skill_pmax=skill_pmax)
    ev.seed(SEEDS)                              # plant the seed families into the population
    best = ev.run(iterations, reset_every=reset_every)
    if best is None:
        raise RuntimeError("no strategy survived the search")

    tools = ev.tools
    strat, space = evolve_mod.compile_strategy(best.code)
    median_oos = float(best.diagnostics["median_oos"])

    # champion SKILL p-value: selection-aware shift-the-signal test at higher resolution than the
    # per-candidate one (more configs and shifts). This is the STATISTICAL significance check
    # ("is the timing real?"), in-sample. Economic significance ("worth owning vs passive?") is the
    # separate buy&hold comparison on the sealed holdout below — the two are kept distinct on purpose.
    champ_skill = {}
    champ_skill_p = float(best.diagnostics.get("skill_pvalue", float("nan")))
    if null_gate:
        log("computing champion skill p-value (selection-aware shift-the-signal) ...")
        try:                                    # report-only diagnostic: never abort finalization
            champ_skill = bt.shift_null_pvalue(strat, space, pool, tools,
                                               n_configs=max(256, null_gate_configs * 4),
                                               n_shifts=max(300, null_gate_shifts * 4),
                                               cost=cost, seed=seed,
                                               objective=objective, min_sharpe=min_sharpe,
                                               benchmark_ret=bench)
            champ_skill_p = float(champ_skill["pvalue"])
        except Exception as e:
            log(f"champion skill p-value failed ({type(e).__name__}: {e}); reporting n/a")

    # final: fit champion on all pool, measure once on the held-out year
    p_full = bt.fit_full(strat, space, pool, tools, budget=champion_budget, cost=cost, seed=seed,
                         objective=objective, min_sharpe=min_sharpe, benchmark_ret=bench)
    hold_sharpe = hold_return = None
    hold_by_year = {}                                   # per held-out year: strategy vs buy&hold
    if len(hold) > 20:
        import pandas as pd
        years = np.asarray(full.index.year)
        strat_ret = bt.run_backtest(strat(full, tools, p_full), full, cost,
                                    stop=p_full.get("stop_loss", 0.0)).to_numpy()
        # true always-long benchmark: a constant-1 position independent of the champion's signal.
        # (clip(1,1) on the signal would leave the champion's own NaNs as NaN -> fillna(0) -> cash.)
        bh_ret = bt.run_backtest(pd.Series(1.0, index=full.index), full["close"], cost).to_numpy()
        hm = years >= holdout_year
        hr = strat_ret[hm]
        hold_sharpe = bt._sharpe_arr(hr)
        hold_return = float(np.prod(1.0 + hr[np.isfinite(hr)]) - 1.0)   # total holdout return
        for y in sorted(set(years[hm].tolist())):
            ym = years == y
            sr, br = strat_ret[ym], bh_ret[ym]
            hold_by_year[str(int(y))] = {
                "strategy_sharpe": bt._sharpe_arr(sr),
                "strategy_return": float(np.prod(1.0 + sr[np.isfinite(sr)]) - 1.0),
                "strategy_maxdd": bt._max_drawdown_arr(sr),
                "strategy_mar": bt._mar_arr(sr),
                "buyhold_sharpe": bt._sharpe_arr(br),
                "buyhold_return": float(np.prod(1.0 + br[np.isfinite(br)]) - 1.0),
                "buyhold_maxdd": bt._max_drawdown_arr(br),
                "buyhold_mar": bt._mar_arr(br),
            }
        # in-sample (pool 2017..holdout) champion risk profile — where leverage shows its bill
        pm = years < holdout_year
        sp, bp = strat_ret[pm], bh_ret[pm]
        insample_risk = {
            "strategy_sharpe": bt._sharpe_arr(sp), "strategy_cagr": bt._cagr_arr(sp),
            "strategy_maxdd": bt._max_drawdown_arr(sp), "strategy_mar": bt._mar_arr(sp),
            "buyhold_sharpe": bt._sharpe_arr(bp), "buyhold_cagr": bt._cagr_arr(bp),
            "buyhold_maxdd": bt._max_drawdown_arr(bp), "buyhold_mar": bt._mar_arr(bp),
            "avg_exposure": float(np.nanmean(bt.as_position_series(
                strat(full, tools, p_full), full.index).clip(-max_leverage, max_leverage)
                .to_numpy()[pm])),
        }
        hold_by_year["_insample"] = insample_risk

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": dict(ticker=ticker, start=start, iterations=iterations,
                       model=getattr(client, "model", model), backend=client.backend,
                       n_splits=n_splits, fit_budget=fit_budget,
                       cost=cost, holdout_year=holdout_year,
                       objective=objective, min_sharpe=min_sharpe, vs_buyhold=vs_buyhold,
                       max_leverage=max_leverage, stop_slippage=stop_slippage),
        "search": dict(evaluated=ev.n_evaluated, rejected=ev.n_rejected,
                       skill_rejected=ev.n_skill_rejected,
                       skill_significant=ev.n_skill_significant,
                       population=len(ev.db.all_programs())),
        "champion": {
            "code": best.code,
            "fitted_params_full": p_full,
            "fitness_median_oos": median_oos,
            "median_oos_sharpe": float(best.diagnostics.get("median_sharpe", median_oos)),
            "median_oos_annual_return": float(best.diagnostics.get("median_return", float("nan"))),
            "mean_oos_fitness": float(best.diagnostics.get("mean_oos", float("nan"))),
            "frac_positive_splits": float(best.diagnostics.get("frac_positive", float("nan"))),
            "holdout_year_sharpe": hold_sharpe,
            "holdout_year_return": hold_return,
            "holdout_by_year": hold_by_year,
            "skill_pvalue": champ_skill_p,
        },
        "skill_test": {
            "method": "selection-aware shift-the-signal (in-sample)",
            "pvalue": champ_skill_p,
            "significant": bool(champ_skill_p < 0.05) if champ_skill_p == champ_skill_p else None,
            "real_max_score": float(champ_skill.get("real", float("nan"))),
            "null_q95": float(champ_skill.get("null_q95", float("nan"))),
            "avg_exposure": float(champ_skill.get("exposure", float("nan"))),
            "n_configs": int(champ_skill.get("n_configs", 0)),
            "n_shifts": int(champ_skill.get("n_shifts", 0)),
        },
    }
    _print_report(result, log)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"run_{data_mod._safe(ticker)}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log(f"saved -> {path}")
    result["_path"] = path
    return result


def _print_report(r, log):
    c, g = r["champion"], r.get("skill_test", {})
    log("\n" + "=" * 70)
    log(f"CHAMPION  ({r['config']['ticker']})")
    log("=" * 70)
    log(c["code"].strip())
    log("-" * 70)
    obj = r["config"].get("objective", "sharpe")
    log(f"fitted params (full pool) : {c['fitted_params_full']}")
    log(f"objective                 : {obj}"
        + (f" (min Sharpe {r['config'].get('min_sharpe')})" if obj == "return" else ""))
    vsb = r["config"].get("vs_buyhold", False)
    unit = "excess vs buy&hold" if vsb else "raw"
    log(f"fitness (CV median OOS)   : {c['fitness_median_oos']:.3f}  ({unit}; "
        f"positive splits {c['frac_positive_splits']:.0%})")
    log(f"CV median OOS {'active ' if vsb else ''}Sharpe : {c['median_oos_sharpe']:.3f}"
        + ("   (>0 => beat buy&hold risk-adjusted, in-sample)" if vsb else ""))
    log(f"CV median OOS {'excess ' if vsb else ''}ann.ret: {c['median_oos_annual_return']:+.1%}")
    hs, hr = c["holdout_year_sharpe"], c["holdout_year_return"]
    log(f"holdout {r['config']['holdout_year']} Sharpe      : "
        + ("n/a" if hs is None else f"{hs:.3f}")
        + f"   return " + ("n/a" if hr is None else f"{hr:+.1%}")
        + "   (measured once, never searched)")
    hby = c.get("holdout_by_year", {})
    ins = hby.get("_insample")
    if ins:
        log(f"in-sample risk (pool)     : strategy Sharpe {ins['strategy_sharpe']:+.2f} "
            f"CAGR {ins['strategy_cagr']:+.0%} maxDD -{ins['strategy_maxdd']:.0%} "
            f"MAR {ins['strategy_mar']:.2f} avg-exposure {ins['avg_exposure']:.2f}x")
        log(f"                            buy&hold Sharpe {ins['buyhold_sharpe']:+.2f} "
            f"CAGR {ins['buyhold_cagr']:+.0%} maxDD -{ins['buyhold_maxdd']:.0%} "
            f"MAR {ins['buyhold_mar']:.2f}")
    for y, m in sorted((k, v) for k, v in hby.items() if k != "_insample"):
        log(f"  {y}: strategy Sh {m['strategy_sharpe']:+.2f} ret {m['strategy_return']:+.1%} "
            f"maxDD -{m['strategy_maxdd']:.0%} MAR {m['strategy_mar']:.2f}"
            f"   |  B&H Sh {m['buyhold_sharpe']:+.2f} ret {m['buyhold_return']:+.1%} "
            f"maxDD -{m['buyhold_maxdd']:.0%} MAR {m['buyhold_mar']:.2f}")
    log("-" * 70)
    sp = c.get("skill_pvalue", float("nan"))
    if sp == sp:                                          # not NaN
        log(f"SKILL TEST (statistical significance, in-sample)")
        log(f"  method                  : selection-aware shift-the-signal "
            f"({g.get('n_configs', 0)} configs x {g.get('n_shifts', 0)} shifts)")
        log(f"  avg exposure            : {g.get('avg_exposure', float('nan')):.0%}")
        log(f"  skill p-value           : {sp:.3f}  "
            + ("SIGNIFICANT timing skill (p<0.05) — beats its own shuffled timing"
               if sp < 0.05 else
               "NOT distinguishable from chance — random re-timing of its own positions matches it"))
        log(f"  economic significance   : judged separately vs buy&hold on the sealed holdout above")
    log("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Evolve a trainable trading strategy for a ticker")
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--fit-budget", type=int, default=200)
    ap.add_argument("--champion-budget", type=int, default=400)
    ap.add_argument("--holdout-year", type=int, default=2026)
    ap.add_argument("--reset-every", type=int, default=50)
    ap.add_argument("--objective", choices=["sharpe", "return"], default="sharpe",
                    help="maximize Sharpe (default) or annual return subject to a Sharpe floor")
    ap.add_argument("--min-sharpe", type=float, default=0.8,
                    help="Sharpe floor when --objective return (soft-penalized below it)")
    ap.add_argument("--no-vs-buyhold", dest="vs_buyhold", action="store_false",
                    help="score raw strategy returns instead of active (excess-over-buy-and-hold)")
    ap.add_argument("--max-leverage", type=float, default=1.0,
                    help="position cap (>1 allows leverage — needed to beat buy-and-hold on return)")
    ap.add_argument("--stop-slippage", type=float, default=0.001,
                    help="slippage on a stopped day (0.001 = exit at the stop + 0.1%%); the stop "
                         "LEVEL is per-strategy — a strategy declares a `stop_loss` param the optimizer tunes")
    ap.add_argument("--theme-file", default=None,
                    help="investment-theme paragraph injected into the prompt (default: theme.txt)")
    ap.add_argument("--no-null-gate", dest="null_gate", action="store_false",
                    help="disable the per-candidate shift-the-signal skill p-value (report-only)")
    ap.add_argument("--null-gate-configs", type=int, default=64,
                    help="configs sampled per candidate for the selection-aware skill test")
    ap.add_argument("--null-gate-shifts", type=int, default=50,
                    help="random circular shifts for the skill test (common random numbers)")
    ap.add_argument("--skill-gate", action="store_true",
                    help="REJECT mutations with no timing skill (skill_p >= --skill-pmax); seeds exempt")
    ap.add_argument("--skill-pmax", type=float, default=0.10,
                    help="skill-gate threshold: keep if skill_p < this, reject (no skill) otherwise")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh-data", action="store_true")
    a = ap.parse_args()
    run_search(ticker=a.ticker, start=a.start, end=a.end, iterations=a.iterations,
               model=a.model, cost=a.cost,
               n_splits=a.splits, fit_budget=a.fit_budget, champion_budget=a.champion_budget,
               holdout_year=a.holdout_year,
               reset_every=a.reset_every, jobs=a.jobs, seed=a.seed,
               objective=a.objective, min_sharpe=a.min_sharpe, vs_buyhold=a.vs_buyhold,
               max_leverage=a.max_leverage, stop_slippage=a.stop_slippage,
               theme_file=a.theme_file,
               null_gate=a.null_gate, null_gate_configs=a.null_gate_configs,
               null_gate_shifts=a.null_gate_shifts,
               skill_gate=a.skill_gate, skill_pmax=a.skill_pmax,
               refresh_data=a.refresh_data)


if __name__ == "__main__":
    main()
