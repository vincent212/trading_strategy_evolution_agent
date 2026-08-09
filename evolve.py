"""
Evolutionary search over trainable strategies.

The program database is divided into islands (independent sub-populations). Each
step samples two high-scoring parents from one island, asks the model for an
improved child, evaluates the child by quarter cross-validation (fit on train
quarters, score OOS on test quarters, median over 100 splits), and inserts it
back into that island. Every `reset_every` steps the lower-scoring half of the
islands are cleared and reseeded from the best strategy found so far.

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
    def __init__(self, n_islands=4, temperature=0.7, seed=0):
        self.islands: list[list[Program]] = [[] for _ in range(n_islands)]
        self.temperature = temperature
        self.rng = random.Random(seed)

    def add(self, island, prog):
        self.islands[island].append(prog)

    def best(self):
        allp = self.all_programs()
        return max(allp, key=lambda p: p.score) if allp else None

    def all_programs(self):
        return [p for isl in self.islands for p in isl]

    def pick_island(self):
        return self.rng.randrange(len(self.islands))

    def sample_parents(self, island, k=2):
        pop = self.islands[island]
        if not pop:
            return []
        scores = np.array([p.score for p in pop], dtype=float)
        w = np.exp((scores - scores.max()) / max(self.temperature, 1e-6))
        w = w / w.sum()
        k = min(k, len(pop))
        idx = list(np.random.choice(len(pop), size=k, replace=False, p=w))
        return sorted((pop[i] for i in idx), key=lambda p: p.score, reverse=True)

    def reset_weak_islands(self, keep_fraction=0.5):
        best_score = [(i, max((p.score for p in isl), default=-9.99))
                      for i, isl in enumerate(self.islands)]
        best_score.sort(key=lambda t: t[1])
        n_reset = int(len(self.islands) * (1 - keep_fraction))
        survivors = self.all_programs()
        if not survivors:
            return
        champ = max(survivors, key=lambda p: p.score)
        for i, _ in best_score[:n_reset]:
            self.islands[i] = [Program(champ.code, champ.score, champ.diagnostics, champ.space)]


class Evolver:
    def __init__(self, data, splits, client, model=MODEL, n_islands=4, k_parents=2,
                 fit_budget=200, cost=0.0005, jobs=1, seed=0, log=print):
        self.data = data
        self.splits = splits
        self.client = client
        self.model = model
        self.k_parents = k_parents
        self.fit_budget = fit_budget
        self.cost = cost
        self.jobs = jobs
        self.seed_val = seed
        self.tools = alpha_tools_module()
        self.db = ProgramDatabase(n_islands=n_islands, seed=seed)
        self.log = log
        self.n_evaluated = 0
        self.n_rejected = 0

    def evaluate(self, code):
        """Compile, quick-check, then quarter-CCV score. None if invalid."""
        import pandas as pd
        try:
            strat, space = compile_strategy(code)
            sig = strat(self.data, self.tools, P.midpoint(space))   # cheap validity check
            if not isinstance(sig, pd.Series):
                raise TypeError("strategy did not return a pandas Series")
            diag = bt.ccv_median_oos(strat, space, self.data, self.tools, self.splits,
                                     self.fit_budget, self.cost, self.jobs, self.seed_val)
        except Exception:
            self.n_rejected += 1
            return None
        score = diag["median_oos"]
        if not np.isfinite(score):
            self.n_rejected += 1
            return None
        self.n_evaluated += 1
        return Program(code, float(score), diag, space)

    def seed(self, seed_code):
        prog = self.evaluate(seed_code)
        if prog is None:
            raise RuntimeError("seed strategy failed to evaluate — check the contract")
        for i in range(len(self.db.islands)):
            self.db.add(i, Program(prog.code, prog.score, prog.diagnostics, prog.space))
        self.log(f"seed median_oos={prog.score:.3f}")

    def _mutate(self, parents):
        kwargs = dict(
            model=self.model,
            max_tokens=1500,
            system=prompt_mod.SYSTEM,
            messages=[{"role": "user",
                       "content": prompt_mod.build_user_prompt([p.as_parent() for p in parents])}],
        )
        # `temperature` is removed on Opus 4.7/4.8 (400); keep it for models that accept
        # it (Haiku 4.5, Sonnet 4.6) to diversify mutations.
        if not self.model.startswith(("claude-opus-4-7", "claude-opus-4-8")):
            kwargs["temperature"] = 1.0
        msg = self.client.messages.create(**kwargs)
        text = next((b.text for b in msg.content if b.type == "text"), "")
        return extract_code(text)

    def step(self):
        island = self.db.pick_island()
        parents = self.db.sample_parents(island, k=self.k_parents)
        if not parents:
            return None
        try:
            code = self._mutate(parents)
        except Exception as e:
            self.log(f"  mutation API error: {e}")
            return None
        child = self.evaluate(code)
        if child is not None:
            self.db.add(island, child)
        return child

    def run(self, iterations, reset_every=50, log_every=10):
        for it in range(1, iterations + 1):
            child = self.step()
            if it % reset_every == 0:
                self.db.reset_weak_islands()
            if it % log_every == 0:
                best = self.db.best()
                b = best.score if best else float("nan")
                self.log(f"[{it:4d}] best_median_oos={b:.3f}  eval={self.n_evaluated} "
                         f"rej={self.n_rejected}"
                         + (f"  child={child.score:.3f}" if child else "  child=REJECT"))
        return self.db.best()
