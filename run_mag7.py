"""
LATSS on the cross-sectional "pick 1 of N" problem (default: the Magnificent 7).

The evolved unit is a NAME-SYMMETRIC per-series scorer:

    param_space() -> dict of tunable knobs
    score(series, tools, p) -> pd.Series of per-bar scores for ONE price series

Each bar the harness holds the single highest-scoring name (argmax -> one-hot).
The LLM never sees ticker identity, so it designs a technical selection RULE, not a
name pick. Fitness = CV median-OOS active Sharpe vs the equal-weight benchmark; the
champion is measured once on a sealed holdout, against equal-weight AND a
random-picker null.

CLI:  python run_mag7.py --iterations 200 --holdout-year 2024
Reuses: llm.py (client), backtest.py (CV splits, scalar objective, shift offsets),
backtest_xs.py (cross-sectional evaluator), params.py (DE encode/decode).
"""
from __future__ import annotations
import os
import re
import sys
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import data_mag7
import backtest as bt
import backtest_xs as xs
import params as P
import alpha_tools as tools_mod
from evolve import extract_code, _SAFE_BUILTINS, _norm_code


# ---- candidate contract ----------------------------------------------------

def compile_score(code: str):
    """exec a candidate; return (score_fn, param_space_dict). Raises on failure."""
    import pandas as pd
    g = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np}
    exec(code, g)
    fn = g.get("score")
    space_fn = g.get("param_space")
    if not callable(fn):
        raise ValueError("no callable `score` defined")
    space = space_fn() if callable(space_fn) else {}
    if not isinstance(space, dict):
        raise ValueError("param_space() must return a dict")
    return fn, space


def repair(code: str) -> str:
    """Strip forbidden imports and prefix bare tool calls (roc(...) -> tools.roc(...))."""
    lines = [ln for ln in code.splitlines()
             if not (ln.strip().startswith("import ") or ln.strip().startswith("from "))]
    code = "\n".join(lines)
    names = getattr(tools_mod, "TOOL_NAMES", [])
    if names:
        # zero-width lookbehind (not a consuming char class) so NESTED calls like
        # zscore(roc(...)) get BOTH names prefixed — a consuming group leaves the inner
        # call un-prefixed (its preceding '(' was eaten by the outer match) and rejected.
        pat = r'(?<![\w.])(' + "|".join(sorted(names, key=len, reverse=True)) + r')\('
        code = re.sub(pat, lambda m: f"tools.{m.group(1)}(", code)
    return code


def augment_space(code: str, space: dict) -> dict:
    """Declare every p["x"] the scorer reads but forgot to put in param_space()."""
    used = set(re.findall(r'p\[\s*[\'"](\w+)[\'"]\s*\]', code))
    for k in used - set(space):
        lk = k.lower()
        if (lk.endswith("n") or lk in ("fast", "slow", "horizon")
                or any(t in lk for t in ("win", "period", "span", "lookback", "len", "lag"))):
            space[k] = ("int", 5, 120)                       # a lookback window
        elif any(t in lk for t in ("quant", "pctile", "pct")):
            space[k] = ("float", 0.0, 1.0)                   # a quantile/percentile in [0,1]
        elif any(t in lk for t in ("thr", "lvl", "level", "cut", "band")):
            space[k] = ("float", 0.0, 3.0)                   # a threshold level
        else:
            space[k] = ("float", -1.0, 1.0)                  # a weight / tilt
    return space


# ---- seed scorers (name-symmetric, one per family) -------------------------

SEEDS = [
    # cross-sectional momentum: prefer the strongest-trend name
    'def param_space():\n    return {"n": ("int", 20, 120)}\n\n'
    'def score(series, tools, p):\n    return tools.roc(series, p["n"])\n',
    # low-vol: prefer the calmest name
    'def param_space():\n    return {"n": ("int", 20, 120)}\n\n'
    'def score(series, tools, p):\n    return -tools.realized_vol(series, p["n"])\n',
    # mean-reversion: prefer the most oversold name
    'def param_space():\n    return {"n": ("int", 5, 40)}\n\n'
    'def score(series, tools, p):\n    return -tools.zscore(series, p["n"])\n',
    # momentum, vol-adjusted: trend per unit of recent vol
    'def param_space():\n    return {"n": ("int", 20, 120), "vn": ("int", 10, 60)}\n\n'
    'def score(series, tools, p):\n    return tools.roc(series, p["n"]) / (tools.realized_vol(series, p["vn"]) + 1e-9)\n',
]

# earnings mode (contract score(series, feats, tools, p); feats["sue"] = point-in-time SUE).
# The score is ONLY a RANKING to select names; positions are fixed equal-weight (binary
# in/out), never sized by the score. So earnings is combined ADDITIVELY as a ranking term,
# never as a position multiplier.
EARNINGS_SEEDS = [
    # pure earnings surprise: rank names by the biggest recent standardized surprise
    'def param_space():\n    return {}\n\n'
    'def score(series, feats, tools, p):\n    return feats["sue"].fillna(0.0)\n',
    # additive: momentum rank + earnings-surprise rank (comparable scales via pctile_rank)
    'def param_space():\n    return {"n": ("int", 20, 120), "w": ("float", 0.0, 2.0)}\n\n'
    'def score(series, feats, tools, p):\n    mom = tools.pctile_rank(tools.roc(series, p["n"]), 252)\n    return mom + p["w"] * feats["sue"].fillna(0.0)\n',
    # additive: raw momentum + weighted earnings surprise
    'def param_space():\n    return {"n": ("int", 20, 120), "w": ("float", 0.0, 2.0)}\n\n'
    'def score(series, feats, tools, p):\n    return tools.roc(series, p["n"]) + p["w"] * feats["sue"].fillna(0.0)\n',
    # additive: vol-adjusted momentum + weighted earnings surprise
    'def param_space():\n    return {"n": ("int", 20, 120), "vn": ("int", 10, 60), "w": ("float", 0.0, 2.0)}\n\n'
    'def score(series, feats, tools, p):\n    return tools.roc(series, p["n"]) / (tools.realized_vol(series, p["vn"]) + 1e-9) + p["w"] * feats["sue"].fillna(0.0)\n',
]

# framework-validation mode: score ONLY from 4 opaque per-name signals (feat_a..feat_d),
# one of which MAY carry signal (positive control) or none (negative control). Seeds try each
# signal alone plus a tunable linear blend, so the search can discover which one predicts.
SYNTH_SEEDS = [
    'def param_space():\n    return {}\n\ndef score(series, feats, tools, p):\n    return feats["feat_a"].fillna(0.0)\n',
    'def param_space():\n    return {}\n\ndef score(series, feats, tools, p):\n    return feats["feat_b"].fillna(0.0)\n',
    'def param_space():\n    return {}\n\ndef score(series, feats, tools, p):\n    return feats["feat_c"].fillna(0.0)\n',
    'def param_space():\n    return {}\n\ndef score(series, feats, tools, p):\n    return feats["feat_d"].fillna(0.0)\n',
    'def param_space():\n    return {"wa": ("float", -1.0, 1.0), "wb": ("float", -1.0, 1.0), '
    '"wc": ("float", -1.0, 1.0), "wd": ("float", -1.0, 1.0)}\n\n'
    'def score(series, feats, tools, p):\n    return (p["wa"]*feats["feat_a"].fillna(0.0) + p["wb"]*feats["feat_b"].fillna(0.0)'
    ' + p["wc"]*feats["feat_c"].fillna(0.0) + p["wd"]*feats["feat_d"].fillna(0.0))\n',
]


# ---- prompt ----------------------------------------------------------------

TOOL_LINES = (
    "sma(x,n) ema(x,n) roc(x,n) zscore(x,n) rsi(x,n) realized_vol(x,n) "
    "rolling_high(x,n) rolling_low(x,n) drawdown(x,n) breakout(x,n) "
    "vol_regime(x,n,lo,hi) pctile_rank(x,n) crossover(fast,slow)"
)


def system_prompt(k=1):
    hold = ("the single name with the highest score (one-hot)" if k == 1 else
            f"the {k} names with the highest scores, equal-weighted (1/{k} each)")
    return (
        "You design a NAME-SYMMETRIC cross-sectional stock SELECTOR for a fixed universe "
        f"(the Magnificent 7). You write two Python functions:\n"
        "  param_space() -> dict {name: (\"int\"|\"float\", lo, hi) | (\"cat\", [choices])}\n"
        "  score(series, tools, p) -> a pandas Series of per-bar scores for ONE price series.\n\n"
        f"The harness applies score() to EACH name independently and, every bar, HOLDS {hold}. "
        "You see only a single anonymous 'series' (its close "
        "prices) — you have NO ticker identity and NO view of the other names, so you cannot and "
        "must not hardcode a specific stock. Express only a technical rule that ranks a name by its "
        "own price behaviour (momentum, volatility, mean-reversion, breakout, regime, ...).\n\n"
        f"Causal tools available (call as tools.NAME): {TOOL_LINES}. All look backward. Read params "
        "via p[\"name\"]. Higher score = more likely to be held. Return a pd.Series aligned to series. "
        "Pure function: no imports, no I/O, no randomness, never index the future.\n\n"
        "Goal: BEAT the equal-weight portfolio of the 7 on risk-adjusted (Sharpe) ACTIVE return after "
        "costs. Merely holding a trending name is a higher-variance bet that a random picker also makes "
        "and it does NOT beat equal-weight on Sharpe — the selection rule must have real timing skill. "
        "Output ONE ```python block with param_space() and score(). No prose."
    )


def earnings_system_prompt(k=1):
    hold = ("the single name with the highest score (one-hot)" if k == 1 else
            f"the {k} names with the highest scores, equal-weighted (1/{k} each)")
    return (
        "You design a NAME-SYMMETRIC cross-sectional stock SELECTOR for a fixed universe. "
        "You write two Python functions:\n"
        "  param_space() -> dict {name: (\"int\"|\"float\", lo, hi) | (\"cat\", [choices])}\n"
        "  score(series, feats, tools, p) -> a pandas Series of per-bar scores for ONE name.\n\n"
        f"The harness applies score() to EACH name independently and, every bar, HOLDS {hold}. For each "
        "anonymous name you get:\n"
        "  series      : its close prices (a pandas Series).\n"
        "  feats['sue'] : its point-in-time STANDARDIZED EARNINGS SURPRISE, a pandas Series aligned to series. "
        "Positive = the last reported quarterly EPS beat the year-ago quarter (standardized by its own history); "
        "~0/NaN before the first earnings. It STEPS at each earnings release and is constant between releases — "
        "use feats['sue'].fillna(0.0). It is fully point-in-time (stamped at the filing date), no look-ahead.\n\n"
        "No ticker identity is given (name-symmetric): rank a name only by its own price behaviour and its own "
        "earnings surprise. Your score is ONLY a RANKING used to select which names to hold — the harness holds "
        "each selected name at a FIXED EQUAL WEIGHT (binary in/out) and NEVER sizes the position by your score. "
        "You MAY use feats['sue'] or ignore it entirely — your choice; decide whether it helps. If you use it, "
        "combine it ADDITIVELY as a ranking term (e.g. a price signal + w*sue). You MUST NOT use SUE (or anything) "
        "as a POSITION/EXPOSURE MULTIPLIER: `momentum * (1 + (sue>0))` and any `signal * gate` form is FORBIDDEN. "
        "Combine signals by ADDING them, never by multiplying one by a gate.\n\n"
        f"Causal price tools (call as tools.NAME): {TOOL_LINES}. All look backward. Read params via p[\"name\"]. "
        "Higher score = more likely held. Return a pd.Series aligned to series. Pure function: no imports, no I/O, "
        "no randomness, never index the future.\n\n"
        "Goal: BEAT the equal-weight portfolio on risk-adjusted (Sharpe) ACTIVE return after costs. Output ONE "
        "```python block with param_space() and score(). No prose."
    )


def synth_system_prompt(k=1):
    hold = ("the single name with the highest score (one-hot)" if k == 1 else
            f"the {k} names with the highest scores, equal-weighted (1/{k} each)")
    return (
        "You design a cross-sectional stock SELECTOR for a fixed universe of 7 names. You write:\n"
        "  param_space() -> dict {name: (\"int\"|\"float\", lo, hi) | (\"cat\", [choices])}\n"
        "  score(series, feats, tools, p) -> a pandas Series of per-bar scores for ONE name.\n\n"
        f"The harness applies score() to EACH name independently and, every bar, HOLDS {hold}.\n"
        "For each name you are given FOUR opaque per-name signals: feats['feat_a'], feats['feat_b'], "
        "feats['feat_c'], feats['feat_d'] — each a pandas Series aligned to that name. You do NOT know "
        "which of them, if any, carry predictive information about future returns and which are pure "
        "noise; discovering that is part of your job. Rank the names so the held name outperforms. Use "
        "feats['feat_x'].fillna(0.0). Score ONLY from these four signals — do NOT use the raw price "
        "`series` or the technical tools; those are not part of this task.\n\n"
        "Combine the signals however you like (pick one, weight several, gate one on another). Higher "
        "score = more likely held. Return a pd.Series aligned to series. Pure function: no imports, no "
        "I/O, no randomness, never index the future.\n\n"
        "Goal: BEAT the equal-weight portfolio on risk-adjusted (Sharpe) ACTIVE return after costs. If "
        "none of the signals help, you cannot beat it — say so by not overfitting. Output ONE "
        "```python block with param_space() and score(). No prose."
    )


def synth_constrained_system_prompt(k=1):
    """Constrained class: the LLM may emit ONLY a fixed linear combination of the 4 features
    — no gating, clipping, products, thresholds, or nonlinearity. Lower VC/Rademacher bar."""
    hold = ("the single name with the highest score (one-hot)" if k == 1 else
            f"the {k} names with the highest scores, equal-weighted (1/{k} each)")
    return (
        "You design a cross-sectional stock SELECTOR for a fixed universe of 7 names. You write:\n"
        "  param_space() -> dict of the four weights\n"
        "  score(series, feats, tools, p) -> a pandas Series of per-bar scores for ONE name.\n\n"
        f"The harness applies score() to EACH name and, every bar, HOLDS {hold}.\n"
        "For each name you get FOUR opaque per-name signals feats['feat_a'..'feat_d']. You do NOT know which "
        "predict. Some may be noise.\n\n"
        "STRICT CONSTRAINT — you may ONLY emit a LINEAR COMBINATION of the four signals:\n"
        "    score = wa*feats['feat_a'].fillna(0.0) + wb*feats['feat_b'].fillna(0.0)"
        " + wc*feats['feat_c'].fillna(0.0) + wd*feats['feat_d'].fillna(0.0)\n"
        "with the four weights declared as float params. NO gating, NO clipping, NO thresholds, NO products of "
        "signals, NO nonlinear transforms, NO use of the raw price or tools. Only the four weights change.\n\n"
        "Goal: BEAT the equal-weight portfolio on risk-adjusted (Sharpe) ACTIVE return after costs. Output ONE "
        "```python block with param_space() and score(). No prose."
    )


# constrained-class seed: the linear-combination form (the only shape allowed in --constrained)
SYNTH_LINEAR_SEEDS = [SYNTH_SEEDS[4]]


def user_prompt(history):
    lines = ["Strategies tried so far (score = CV median-OOS active Sharpe vs equal-weight; higher is better):\n"]
    for h in sorted(history, key=lambda z: z["score"], reverse=True)[:12]:
        lines.append(f"[score {h['score']:+.3f}]\n```python\n{h['code'].strip()}\n```\n")
    lines.append("Propose ONE improved score() (and its param_space) that should rank names better. "
                 "Combine measurements nonlinearly (e.g. gate momentum on a vol regime). "
                 "Output only the ```python block.")
    return "\n".join(lines)


# ---- evaluate one candidate ------------------------------------------------

def evaluate(code, panel, rets, tools, splits, bench, cost, fit_budget, seed, k=1, feats=None):
    """Compile + CV-score one candidate. Fitness = CV median-OOS active Sharpe. No
    shift-the-signal skill test (it is invalid — see latss_paper_frame.md); skill is
    judged after the search against the Rademacher/VC bars."""
    code = repair(code)
    try:
        fn, space = compile_score(code)
        space = augment_space(code, space)
        _ = xs.score_matrix(fn, panel, tools, P.midpoint(space), feats)   # smoke: finite score matrix
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", code
    try:
        diag = xs.ccv_median_oos(fn, space, panel, rets, tools, splits, bench,
                                 budget=fit_budget, cost=cost, seed=seed, k=k, feats=feats)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", code
    if not np.isfinite(diag["median_oos"]):
        return None, "non-finite fitness", code
    diag["code"] = code
    diag["space"] = space
    return diag, None, code


# ---- main search -----------------------------------------------------------

def run(tickers=data_mag7.MAG7, start="2015-01-01", iterations=200, model="claude-haiku-4-5",
        cost=0.0005, n_splits=60, fit_budget=150, reset_every=40, seed=0, top_k=1,
        earnings=False, synth=None, synth_alpha=0.98, constrained=False, cv_method="purged",
        out_dir=None, refresh_data=False):
    """LATSS search. NO hold-out — all data is in-sample (with unconstrained search you cannot
    truly hold anything out). Skill is judged after the search against the Rademacher / VC /
    Deflated-Sharpe bars, not a hold-out and not the (invalid) shift-the-signal test."""
    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)
    k = int(top_k)

    full = data_mag7.get_panel(tickers, start=start, refresh=refresh_data)
    pool = full                                            # NO holdout — all data is in-sample
    modality = ("synth:" + synth + ("/linear" if constrained else "/free")) if synth else \
               ("technical+earnings(SUE)" if earnings else "technical")
    tools = tools_mod
    pool_rets = xs._returns_matrix(pool)
    pool_bench = xs.equalweight_returns(pool_rets, cost)
    splits = (bt.make_purged_kfold_splits(pool.index, n_folds=6, embargo=250) if cv_method == "purged"
              else bt.make_quarter_splits(pool.index, n_splits=n_splits, seed=seed))
    years = len(pool) / bt.PERIODS_PER_YEAR
    print(f"universe ({len(full.columns)}) {list(full.columns)} | PICK {k} | modality {modality} | "
          f"CV {cv_method} | {pool.index[0].date()}..{pool.index[-1].date()} ({len(pool)}) — ALL in-sample")

    # earnings modality: point-in-time SUE per name, aligned to the FULL panel (warm),
    # then sliced to the pool for CV/fit. Passed as feats to the score() contract.
    feats_full = feats_pool = None
    seeds, prompt_fn = SEEDS, system_prompt
    if earnings:
        import earnings as earn
        def _safe_sue(t):
            try:
                return earn.sue_series(t, full.index, refresh=refresh_data)
            except Exception as e:                          # foreign filer / no EPS / network
                print(f"  SUE unavailable for {t} ({type(e).__name__}) — technicals only")
                return pd.Series(np.nan, index=full.index)
        sue_full = pd.DataFrame({t: _safe_sue(t) for t in full.columns})[list(full.columns)]
        feats_full = {"sue": sue_full}
        feats_pool = {"sue": sue_full.loc[pool.index]}
        seeds, prompt_fn = EARNINGS_SEEDS, earnings_system_prompt
        cov = float(sue_full.loc[pool.index].notna().mean().mean())
        print(f"earnings: SUE loaded for {len(full.columns)} names "
              f"(pool coverage {cov:.0%} of name-days have a live SUE)")
    if synth:
        import synth as synth_mod
        pred = (synth == "predictor")
        sf = synth_mod.make_features(full, predictor=pred, rho=synth_alpha, base=1.0, horizon=21, seed=seed)
        feats_full = feats_pool = sf                       # pool == full (no holdout)
        seeds = SYNTH_LINEAR_SEEDS if constrained else SYNTH_SEEDS
        prompt_fn = synth_constrained_system_prompt if constrained else synth_system_prompt
        print(f"synth: {synth} (alpha/rho={synth_alpha}), "
              + ("ONE hidden feature carries a signal" if pred else "ALL features noise")
              + f"; class = {'LINEAR (constrained)' if constrained else 'free (unconstrained)'}")

    import llm
    client = llm.make_client(model)
    print(f"llm backend: {client.backend}  model: {getattr(client, 'model', model)}")

    # seed the population
    pop = []            # list of diag dicts (each has code/space/median_oos/...)
    for sc in seeds:
        diag, err, _ = evaluate(sc, pool, pool_rets, tools, splits, pool_bench,
                                cost, fit_budget, seed, k, feats_pool)
        if diag is not None:
            pop.append(diag)
            print(f"seed: median_oos={diag['median_oos']:+.3f}")
    if not pop:
        raise RuntimeError("no seed evaluated")

    seen = {_norm_code(d["code"]) for d in pop}
    best = max(pop, key=lambda d: d["median_oos"])
    print(f"best seed median_oos={best['median_oos']:+.3f}")

    for it in range(1, iterations + 1):
        history = [{"code": d["code"], "score": d["median_oos"]} for d in pop]
        try:
            text = client.mutate(prompt_fn(k), user_prompt(history),
                                 model=model, max_tokens=1200, temperature=1.0)
            code = extract_code(text)
        except Exception as e:
            print(f"[{it:4d}] mutate error: {e}")
            continue
        if _norm_code(repair(code)) in seen:
            print(f"[{it:4d}] dup")
            continue
        diag, err, rcode = evaluate(code, pool, pool_rets, tools, splits, pool_bench,
                                    cost, fit_budget, seed, k, feats_pool)
        seen.add(_norm_code(rcode))
        if diag is None:
            print(f"[{it:4d}] REJECT: {err}")
            continue
        pop.append(diag)
        tag = ""
        if diag["median_oos"] > best["median_oos"] + 1e-9:
            best = diag
            tag = "  *** new best ***"
        print(f"[{it:4d}] child median_oos={diag['median_oos']:+.3f} pop={len(pop)}{tag}")
        if it % reset_every == 0 and len(pop) > 6:      # cull weakest half, keep champion
            pop.sort(key=lambda d: d["median_oos"], reverse=True)
            pop = pop[:max(3, len(pop) // 2)]
            if best not in pop:
                pop.append(best)

    # ---- finalize: fit champion on ALL data; judge skill against the three bars (no holdout) ----
    fn, space = compile_score(best["code"])
    full_rets = pool_rets
    bench_full = pool_bench
    p_full = xs.fit_full(fn, space, pool, full_rets, tools, bench_full,
                         budget=fit_budget * 2, cost=cost, seed=seed, k=k, feats=feats_full)
    champ_active = (xs.portfolio_returns(
        xs.top_k_weights(xs.score_matrix(fn, pool, tools, p_full, feats_full), k), full_rets, cost)
        - bench_full)

    # three complementary skill bars, all in-sample (see latss_paper_frame.md)
    n_feats = len(feats_full) if feats_full else 0
    vc_dim = (n_feats + 1) if (synth and constrained) else float("inf")   # linear -> h=d+1; unbounded LLM -> inf
    vc = xs.vc_bar(vc_dim, len(pool), years)
    cand = []
    for d in pop:
        try:
            cand.append((compile_score(d["code"])[0], d["space"]))
        except Exception:
            pass
    radem = xs.rademacher_bar(cand, pool, full_rets, tools, splits, bench_full,
                              n_scramble=15, cost=cost, k=k, feats=feats_full, fit_budget=fit_budget)
    dsr = xs.deflated_sharpe(champ_active,
                             [d["median_oos"] / (bt.PERIODS_PER_YEAR ** 0.5) for d in pop],
                             n_trials=len(pop))
    cv = float(best["median_oos"])
    H = (cv ** 2 / radem["bar_q"] ** 2) if radem["bar_q"] > 1e-9 else float("inf")
    clears = (cv > radem["bar_q"]) and (cv < vc if np.isfinite(vc) else True)

    top3 = sorted(pop, key=lambda d: d["median_oos"], reverse=True)[:3]
    report = {
        "champion_code": best["code"], "pick_k": k, "earnings": bool(earnings), "synth": synth,
        "synth_alpha": synth_alpha if synth else None, "constrained": bool(constrained),
        "cv_method": cv_method, "universe": list(pool.columns), "fitted_params": p_full,
        "years": years, "cv_median_oos_active_sharpe": cv,
        "bars": {"vc_dim": (None if not np.isfinite(vc_dim) else vc_dim),
                 "vc_bar": (None if not np.isfinite(vc) else vc),
                 "rademacher_bar_q95": radem["bar_q"], "rademacher_bar_mean": radem["bar_mean"],
                 "rademacher_headroom": H, "deflated_sharpe": dsr["dsr"],
                 "clears_rademacher": bool(clears)},
        "top_strategies": [{"rank": i + 1, "code": d["code"], "space": d["space"],
                            "median_oos": d["median_oos"]} for i, d in enumerate(top3)],
    }
    print("\n" + "=" * 68 + "\nCHAMPION\n" + "=" * 68)
    print(best["code"].strip())
    print("-" * 68)
    print(f"fitted params        : {p_full}")
    print(f"CV median-OOS active Sharpe ({cv_method}) : {cv:+.3f}")
    print("SKILL BARS (in-sample; NO holdout):")
    print(f"  VC (Vapnik) bar     : "
          + ("inf  (unbounded class — no finite bar)" if not np.isfinite(vc) else f"{vc:+.2f}  (h={vc_dim:g})"))
    print(f"  Rademacher bar q95  : {radem['bar_q']:+.2f}  (mean {radem['bar_mean']:+.2f})  ->  headroom {H:.1f}x")
    print(f"  Deflated Sharpe     : {dsr['dsr']:.3f}")
    verdict = (("CLEARS the Rademacher bar" + (" with headroom>=2.5" if H >= 2.5 else " (weak: headroom<2.5)"))
               if clears else "does NOT clear the Rademacher bar — not distinguishable from noise")
    print(f"  verdict             : {verdict}")
    print("=" * 68)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = (("s" + synth[0] + ("L" if constrained else "")) if synth   # spL=synth predictor linear, etc.
              else ("e" if earnings else ""))
    path = os.path.join(out_dir, f"mag7_k{k}{suffix}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"saved -> {path}")
    return report


def main():
    ap = argparse.ArgumentParser(description="LATSS pick-1-of-N over the Magnificent 7")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--cv-method", choices=["purged", "random"], default="purged",
                    help="purged/embargoed chronological k-fold (honest) or random quarter splits (permissive)")
    ap.add_argument("--constrained", action="store_true",
                    help="synth mode: restrict the LLM to a LINEAR combination of features (low-VC class)")
    ap.add_argument("--synth-alpha", type=float, default=0.98,
                    help="synth predictor signal strength (rho, 0..1); sweep this to find the recovery threshold")
    ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--splits", type=int, default=60)
    ap.add_argument("--fit-budget", type=int, default=150)
    ap.add_argument("--top-k", type=int, default=1, help="hold the top-k names each bar (equal-weighted)")
    ap.add_argument("--earnings", action="store_true",
                    help="add point-in-time SUE (SEC EDGAR) as a per-name feature (score gets feats['sue'])")
    ap.add_argument("--synth", choices=["control", "predictor"], default=None,
                    help="framework-validation mode: 4 opaque synthetic features. "
                         "control = all noise (negative); predictor = one hidden GOOGL signal (positive)")
    ap.add_argument("--tickers", default=None,
                    help="comma-separated universe (default: the Magnificent 7)")
    ap.add_argument("--out-dir", default=None, help="output + champion dir (default runs/)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh-data", action="store_true")
    a = ap.parse_args()
    tickers = [t.strip().upper() for t in a.tickers.split(",")] if a.tickers else data_mag7.MAG7
    run(tickers=tickers, iterations=a.iterations, model=a.model, start=a.start, cost=a.cost,
        n_splits=a.splits, fit_budget=a.fit_budget, seed=a.seed, top_k=a.top_k, earnings=a.earnings,
        synth=a.synth, synth_alpha=a.synth_alpha, constrained=a.constrained, cv_method=a.cv_method,
        out_dir=a.out_dir, refresh_data=a.refresh_data)


if __name__ == "__main__":
    main()
