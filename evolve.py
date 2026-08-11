"""
Evolutionary search over trainable strategies.

A single evolving population. Each step shows the model the WHOLE scored history, asks for an
improved child, evaluates it by quarter cross-validation (fit on train quarters, score OOS on test
quarters, median over 100 splits — on the active return vs buy-and-hold), and inserts it into the
pool. Every `reset_every` steps the lower-scoring half of the pool is culled to purge stagnation
(the champion always survives).

Candidate code is executed via exec() in a restricted namespace. This is a
constraint on convenience, not a security boundary; run only inspected code.
"""
from __future__ import annotations
import re
import random
import numpy as np

import prompt as prompt_mod
import backtest as bt
import params as P

MODEL = "claude-haiku-4-5"
_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "range": range, "len": len,
    "float": float, "int": int, "bool": bool, "round": round, "sum": sum,
    "enumerate": enumerate, "zip": zip, "map": map, "list": list, "dict": dict,
    "tuple": tuple, "sorted": sorted, "True": True, "False": False, "None": None,
}


def extract_code(text: str) -> str:
    m = _CODE_RE.search(text)
    return (m.group(1) if m else text).strip()


def compile_strategy(code: str):
    """exec a candidate; return (strategy_fn, param_space_dict). Raises on failure."""
    import pandas as pd
    g = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np}
    exec(code, g)                                   # see module docstring
    strat = g.get("strategy")
    space_fn = g.get("param_space")
    if not callable(strat):
        raise ValueError("no callable `strategy` defined")
    space = space_fn() if callable(space_fn) else {}
    if not isinstance(space, dict):
        raise ValueError("param_space() must return a dict")
    return strat, space


def alpha_tools_module():
    import alpha_tools
    return alpha_tools


class Program:
    __slots__ = ("code", "score", "diagnostics", "space")

    def __init__(self, code, score, diagnostics, space):
        self.code = code
        self.score = score
        self.diagnostics = diagnostics
        self.space = space

    def as_parent(self):
        return {"code": self.code, "score": self.score, "diagnostics": self.diagnostics}


class ProgramDatabase:
    """A single evolving population. `reset()` periodically culls the weakest half to purge
    stagnation (the champion always survives). There used to be 4 'islands', but the mutation
    prompt always drew on the whole population (all_programs), so they never isolated sub-
    populations and had no crossover — they only ever drove this cull."""
    def __init__(self, seed=0):
        self.pool: list[Program] = []
        self.rng = random.Random(seed)

    def add(self, prog):
        self.pool.append(prog)

    def best(self):
        return max(self.pool, key=lambda p: p.score) if self.pool else None

    def all_programs(self):
        return list(self.pool)

    def reset(self, keep_fraction=0.5):
        """Purge stagnation: keep the top keep_fraction of the pool (champion always survives)."""
        if len(self.pool) < 4:
            return
        self.pool.sort(key=lambda p: p.score, reverse=True)
        self.pool = self.pool[:max(1, int(len(self.pool) * keep_fraction))]


def _norm_code(code: str) -> str:
    """Normalize code for exact-duplicate detection (ignore blank lines / trailing ws)."""
    return "\n".join(ln.rstrip() for ln in code.strip().splitlines() if ln.strip())


class Evolver:
    def __init__(self, data, splits, client, model=MODEL,
                 fit_budget=200, cost=0.0005, jobs=1, seed=0, log=print,
                 objective="sharpe", min_sharpe=0.8, vs_buyhold=True, theme=None,
                 max_leverage=1.0, null_gate=True, null_gate_configs=64, null_gate_shifts=50):
        self.data = data
        self.splits = splits
        self.client = client
        self.model = model
        self.theme = theme          # investment-theme paragraph injected into the system prompt
        self.max_leverage = max_leverage
        self.fit_budget = fit_budget
        self.cost = cost
        self.objective = objective
        self.min_sharpe = min_sharpe
        # vs_buyhold: score the ACTIVE return (strategy − buy-and-hold) everywhere, so the fitness
        # is risk-adjusted OUTPERFORMANCE of buy-and-hold (a strategy that merely holds scores 0),
        # and the skill test asks whether that outperformance is real vs a random re-timing of the
        # same positions. Computed once for the pool; the benchmark return is params-independent.
        self.vs_buyhold = vs_buyhold
        self._bench = bt.buyhold_returns(self.data["close"], cost) if vs_buyhold else None
        # per-candidate SELECTION-AWARE SHIFT-THE-SIGNAL skill p-value (REPORT-ONLY — does NOT
        # filter selection). For every candidate we sample null_gate_configs configs of its
        # structure, take the MAX in-sample score over them (the selection the search does), and
        # compare it to that same max under a fixed set of random circular SHIFTS of the positions.
        # Shifting keeps returns and each config's exposure profile exactly and destroys only the
        # signal<->return alignment, so drift and exposure level cancel and only timing skill is
        # tested. p = fraction of shifted maxima >= the real max. Selection is still by CV
        # median-OOS fitness; this p-value is logged and headlined only. Shift offsets are fixed
        # once (common random numbers) so the bar is a consistent comparison across candidates.
        self.null_gate = null_gate
        self.null_gate_configs = null_gate_configs
        self.null_gate_shifts = null_gate_shifts
        self.n_skill_significant = 0       # candidates with skill p < 0.05
        self._last_skill_p = float("nan")  # most recent candidate's skill p-value (for logging)
        self._shift_offsets = None
        if null_gate:
            self._shift_offsets = bt.make_shift_offsets(len(self.data), null_gate_shifts, seed)
        self.jobs = jobs
        self.seed_val = seed
        self.tools = alpha_tools_module()
        self.db = ProgramDatabase(seed=seed)
        self.log = log
        self.n_evaluated = 0
        self.n_rejected = 0
        self.n_dup = 0
        self._code_cache = {}    # normalized code   -> Program | None
        self._sig_cache = {}     # default-param positions -> Program (behavioral dedup)
        self._last_cached = False
        self._last_code = ""     # raw code of the most recent child (for logging)
        self._n_steps = 0        # mutation counter (drives the periodic exploration turn)
        self._last_explore = False
        self.n_selfcorrected = 0 # rejects rescued by the LLM self-correct retry
        self._last_error = None  # error string of the most recent reject (fed to self-correct)

    def _log_reject(self, reason, code):
        """Print a rejected candidate and why — so rejects are visible (even ones later self-corrected)."""
        self.log(f"    REJECT: {reason}")
        body = "\n".join("      " + ln for ln in code.strip().splitlines())
        self.log(f"    rejected code:\n{body}")

    def evaluate(self, code):
        """Compile, quick-check, then quarter-CCV score. None if invalid.

        Deduplicated: a strategy whose normalized code — or whose positions at the
        default params — was already seen is returned from cache and NOT re-fit.
        self._last_cached flags such a cache hit (so step() won't re-add a clone)."""
        import pandas as pd
        self._last_cached = False
        self._last_skill_p = float("nan")   # reset so a dup/reject row never logs a stale p
        self._last_error = None
        code = self._repair(code)                          # deterministic hygiene fixes
        self._last_code = code                             # log/self-correct see the repaired code
        ckey = _norm_code(code)
        if ckey in self._code_cache:                       # exact / formatting duplicate
            self.n_dup += 1
            self._last_cached = True
            return self._code_cache[ckey]
        try:
            strat, space = compile_strategy(code)
            space = self._augment_space(code, space)       # declare any p["x"] used but missing
            sig = strat(self.data, self.tools, P.midpoint(space))   # cheap validity check
            if isinstance(sig, np.ndarray):                # np.where output -> align positionally
                sig = pd.Series(sig.ravel(), index=self.data.index)
            if not isinstance(sig, pd.Series):
                raise TypeError(f"strategy returned {type(sig).__name__}, expected a Series/array")
        except Exception as e:
            self.n_rejected += 1
            self._last_error = f"{type(e).__name__}: {e}"
            self._log_reject(self._last_error, code)
            self._code_cache[ckey] = None
            return None
        # behavioral duplicate: identical positions at default params == same strategy.
        # Skip when the default-param signal is (near) constant — many distinct strategies
        # produce a flat/zero signal there (e.g. a top-level weight whose midpoint is 0),
        # so hashing it would wrongly merge them.
        arr = sig.reindex(self.data.index).fillna(0.0).to_numpy()
        skey = np.round(arr, 6).tobytes() if float(np.std(arr)) > 1e-9 else None
        if skey is not None and skey in self._sig_cache:
            self.n_dup += 1
            self._last_cached = True
            self._code_cache[ckey] = self._sig_cache[skey]
            return self._sig_cache[skey]
        try:
            diag = bt.ccv_median_oos(strat, space, self.data, self.tools, self.splits,
                                     self.fit_budget, self.cost, self.jobs, self.seed_val,
                                     objective=self.objective, min_sharpe=self.min_sharpe,
                                     benchmark_ret=self._bench)
        except Exception as e:
            self.n_rejected += 1
            self._last_error = f"{type(e).__name__}: {e}"
            self._log_reject(self._last_error, code)
            self._code_cache[ckey] = None
            return None
        score = diag["median_oos"]
        if not np.isfinite(score):
            self.n_rejected += 1
            self._log_reject("non-finite fitness (nan/inf)", code)
            self._code_cache[ckey] = None
            return None
        # RISK PROFILE for the PROMPT — so the model optimizes with drawdown/leverage IN VIEW, not
        # blind to it. Fit once on the full pool, backtest, record max drawdown, MAR and avg exposure.
        diag["maxdd"] = diag["mar"] = diag["avg_exposure"] = float("nan")
        try:
            pf = bt.fit_full(strat, space, self.data, self.tools, budget=self.fit_budget,
                             cost=self.cost, seed=self.seed_val, objective=self.objective,
                             min_sharpe=self.min_sharpe, benchmark_ret=self._bench)
            rr = bt.run_backtest(strat(self.data, self.tools, pf),
                                 self.data["close"], self.cost).to_numpy()
            diag["maxdd"] = bt._max_drawdown_arr(rr)
            diag["mar"] = bt._mar_arr(rr)
            posv = strat(self.data, self.tools, pf)
            posv = posv.to_numpy() if hasattr(posv, "to_numpy") else np.asarray(posv, float).ravel()
            diag["avg_exposure"] = float(np.nanmean(np.clip(posv, -9.0, 9.0)))
        except Exception:
            pass
        # REPORT-ONLY selection-aware shift-the-signal SKILL p-value (does NOT affect selection).
        # Attached to diagnostics and logged; the champion gets a higher-resolution version at the end.
        diag["skill_pvalue"] = float("nan")
        if self.null_gate:
            try:
                res = bt.shift_null_pvalue(strat, space, self.data, self.tools,
                                           n_configs=self.null_gate_configs,
                                           shift_offsets=self._shift_offsets,
                                           cost=self.cost, seed=self.seed_val,
                                           objective=self.objective, min_sharpe=self.min_sharpe,
                                           benchmark_ret=self._bench)
                diag["skill_pvalue"] = res["pvalue"]
                self._last_skill_p = res["pvalue"]
                if res["pvalue"] < 0.05:
                    self.n_skill_significant += 1
            except Exception as e:                      # diagnostic only: never block the search,
                self.log(f"    skill p-value failed ({type(e).__name__}: {e})")   # but surface it
        self.n_evaluated += 1
        prog = Program(code, float(score), diag, space)
        self._code_cache[ckey] = prog
        if skey is not None:
            self._sig_cache[skey] = prog
        return prog

    def seed(self, seed_codes):
        """Plant the seed families into the population. Accepts a single code or a list."""
        if isinstance(seed_codes, str):
            seed_codes = [seed_codes]
        progs = []
        for code in seed_codes:
            pr = self.evaluate(code)
            if pr is not None:
                progs.append(pr)
                self.log(f"seed {len(progs)}: median_oos={pr.score:+.3f}")
        if not progs:
            raise RuntimeError("no seed strategy evaluated — check the contract")
        for pr in progs:
            self.db.add(Program(pr.code, pr.score, pr.diagnostics, pr.space))

    # ---- auto-repair: deterministically fix the common 7B hygiene failures ----------------

    def _repair(self, code: str) -> str:
        """Strip forbidden imports and prefix bare tool calls (crossover(...) -> tools.crossover(...))."""
        lines = [ln for ln in code.splitlines()
                 if not (ln.strip().startswith("import ") or ln.strip().startswith("from "))]
        code = "\n".join(lines)
        names = getattr(self.tools, "TOOL_NAMES", [])
        if names:
            pat = r'(^|[^\w.])(' + "|".join(sorted(names, key=len, reverse=True)) + r')\('
            code = re.sub(pat, lambda m: f"{m.group(1)}tools.{m.group(2)}(", code)
        return code

    def _augment_space(self, code: str, space: dict) -> dict:
        """Declare every p["x"] the strategy reads but forgot to put in param_space()."""
        used = set(re.findall(r'p\[\s*[\'"](\w+)[\'"]\s*\]', code))
        for k in used - set(space):
            space[k] = self._infer_range(k, code)
        return space

    @staticmethod
    def _infer_range(k: str, code: str):
        # categorical if compared against string literals: p["mode"] == "trend"
        choices = re.findall(r'p\[\s*[\'"]' + re.escape(k) + r'[\'"]\s*\]\s*==\s*[\'"](\w+)[\'"]', code)
        if choices:
            return ("cat", sorted(set(choices)))
        lk = k.lower()
        if lk.endswith("_n") or lk.endswith("n") or any(
                t in lk for t in ("win", "window", "period", "span", "lookback", "len", "lag")):
            return ("int", 5, 60)
        if any(t in lk for t in ("thr", "lvl", "level", "cut", "band", "quant")):
            return ("float", 0.0, 2.0)
        return ("float", -1.0, 1.0)         # weights / tilts / generic knobs

    def _self_correct(self, code: str, error: str) -> str | None:
        """One LLM retry: hand back the crash + code and ask for a minimal fix."""
        sys = ("You fix a Python trading strategy that failed to run. Output ONLY the corrected "
               "code as one ```python block with both param_space() and strategy(). No prose.")
        user = (f"This strategy crashed with:\n{error}\n\n```python\n{code}\n```\n\n"
                f"Fix ONLY what caused the crash: declare in param_space() every p['x'] the "
                f"strategy reads, remove any import lines, fix undefined names/typos, and call "
                f"tools as tools.NAME(...). Keep the strategy idea and structure. Return the full "
                f"corrected param_space() + strategy() in one ```python block.")
        try:
            text = self.client.mutate(sys, user, model=self.model, max_tokens=1500, temperature=0.2)
        except Exception:
            return None
        return extract_code(text)

    EXPLORE_EVERY = 5            # every Nth mutation is a random-jump exploration turn
    MAX_FIX = 2                  # LLM self-correct retries after deterministic repair fails

    def _mutate(self, explore=False):
        # Show the WHOLE history (every distinct strategy tried + its score), not just two
        # parents, so the model can see which structures win and avoid repeating them.
        history = [p.as_parent() for p in self.db.all_programs()]
        user = prompt_mod.build_user_prompt(history, explore=explore)
        # The client (Anthropic or OpenAI-compatible) handles provider specifics,
        # including dropping temperature on models that reject it.
        text = self.client.mutate(prompt_mod.system_prompt(self.theme, self.max_leverage),
                                  user, model=self.model, max_tokens=1500, temperature=1.0)
        return extract_code(text)

    def step(self):
        if not self.db.all_programs():
            return None
        self._n_steps += 1
        self._last_explore = (self._n_steps % self.EXPLORE_EVERY == 0)
        try:
            code = self._mutate(explore=self._last_explore)
        except Exception as e:
            self.log(f"  mutation API error: {e}")
            return None
        child = self.evaluate(code)                        # repairs + sets _last_code/_last_error
        attempts = 0                                       # up to MAX_FIX LLM self-correct retries
        while (child is None and not self._last_cached and self._last_error
               and attempts < self.MAX_FIX):
            attempts += 1
            prev = self._last_code
            fixed = self._self_correct(prev, self._last_error)
            if not fixed or _norm_code(fixed) == _norm_code(prev):
                break                                      # model gave nothing new — stop
            child = self.evaluate(fixed)                   # updates _last_code/_last_error for next try
        if child is not None and attempts > 0:
            self.n_selfcorrected += 1
        if child is not None and not self._last_cached:   # don't re-add a duplicate clone
            self.db.add(child)
        return child

    def run(self, iterations, reset_every=50, log_every=1):
        cur = self.db.best()
        best_score = cur.score if cur else float("-inf")
        for it in range(1, iterations + 1):
            child = self.step()
            cur = self.db.best()
            if cur and cur.score > best_score + 1e-9:
                best_score = cur.score
                self.log(f"[{it:4d}]   *** new best median_oos = {best_score:+.3f} ***")
            if it % reset_every == 0:
                self.db.reset()
            if it % log_every == 0:
                cs = ("dup" if self._last_cached else
                      (f"{child.score:+.3f}" if child else "REJECT"))
                sh = (f" Sh={child.diagnostics.get('median_sharpe', float('nan')):+.2f}"
                      if child is not None else "")
                skp = (f" skill_p={self._last_skill_p:.2f}"
                       if child is not None and np.isfinite(self._last_skill_p) else "")
                tag = " EXPLORE" if self._last_explore else ""
                self.log(f"[{it:4d}]{tag} best={best_score:+.3f}  child={cs}{sh}{skp}  "
                         f"pop={len(self.db.all_programs())} rej={self.n_rejected} "
                         f"sig={self.n_skill_significant} dup={self.n_dup} "
                         f"fixed={self.n_selfcorrected}")
                # rejects already logged their code + reason in evaluate(); print accepted/dup here
                if self._last_code and child is not None:
                    body = "\n".join("      " + ln for ln in self._last_code.strip().splitlines())
                    self.log(f"    child code (score={cs}{skp}):\n{body}")
        return self.db.best()
