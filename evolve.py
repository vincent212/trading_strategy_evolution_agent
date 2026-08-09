"""
The FunSearch evolutionary loop.

- ProgramDatabase holds several 'islands' (separate populations) to preserve
  diversity. Each entry is a scored strategy.
- Each step: pick an island, sample high-scoring parents, ask the LLM (Haiku by
  default) for an improved child, compile it, evaluate it on the TRAIN period via
  backtest.block_cv_score, and insert it back into that island.
- Islands are periodically reset: the weakest half are wiped and reseeded from the
  best survivors, which stops the search from stagnating.

SECURITY NOTE: this compiles and runs LLM-generated Python via exec(). It is intended
to run locally on your own machine on code you can inspect. The exec namespace is
restricted to a small builtin set, but that is a speed bump, not a sandbox. Do not
point this at an untrusted model or run it on shared infrastructure without a real
sandbox (subprocess + seccomp / container).
"""
from __future__ import annotations
import re
import math
import random
import numpy as np

import prompt as prompt_mod
import backtest as bt

MODEL = "claude-haiku-4-5"          # the mutation operator; change in run.py if desired
_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

# A deliberately small builtin surface for exec'd strategies.
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
    """exec a strategy code string and return the callable. Raises on failure."""
    import pandas as pd
    g = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np}
    exec(code, g)                                   # noqa: S102 (see SECURITY NOTE)
    fn = g.get("strategy")
    if not callable(fn):
        raise ValueError("no callable `strategy` defined")
    return fn


# ---------------------------------------------------------------------------

class Program:
    __slots__ = ("code", "score", "diagnostics", "signal")

    def __init__(self, code, score, diagnostics, signal):
        self.code = code
        self.score = score
        self.diagnostics = diagnostics
        self.signal = signal            # cached in-sample signal, for the null bar

    def as_parent(self):
        return {"code": self.code, "score": self.score, "diagnostics": self.diagnostics}


class ProgramDatabase:
    def __init__(self, n_islands: int = 4, temperature: float = 0.7):
        self.islands: list[list[Program]] = [[] for _ in range(n_islands)]
        self.temperature = temperature
        self.rng = random.Random(0)

    def add(self, island: int, prog: Program):
        self.islands[island].append(prog)

    def best(self) -> Program | None:
        allp = [p for isl in self.islands for p in isl]
        return max(allp, key=lambda p: p.score) if allp else None

    def all_programs(self) -> list[Program]:
        return [p for isl in self.islands for p in isl]

    def pick_island(self) -> int:
        return self.rng.randrange(len(self.islands))

    def sample_parents(self, island: int, k: int = 2) -> list[Program]:
        """Softmax sampling by score within an island (best first in the return)."""
        pop = self.islands[island]
        if not pop:
            return []
        scores = np.array([p.score for p in pop], dtype=float)
        w = np.exp((scores - scores.max()) / max(self.temperature, 1e-6))
        w = w / w.sum()
        k = min(k, len(pop))
        idx = list(np.random.choice(len(pop), size=k, replace=False, p=w))
        chosen = [pop[i] for i in idx]
        return sorted(chosen, key=lambda p: p.score, reverse=True)

    def reset_weak_islands(self, keep_fraction: float = 0.5):
        """Wipe the weakest islands; reseed each from the best program overall."""
        best_by_island = [(i, max((p.score for p in isl), default=-9.99))
                          for i, isl in enumerate(self.islands)]
        best_by_island.sort(key=lambda t: t[1])
        n_reset = int(len(self.islands) * (1 - keep_fraction))
        survivors = [p for isl in self.islands for p in isl]
        if not survivors:
            return
        seed = max(survivors, key=lambda p: p.score)
        for i, _ in best_by_island[:n_reset]:
            self.islands[i] = [Program(seed.code, seed.score, seed.diagnostics, seed.signal)]


# ---------------------------------------------------------------------------

class Evolver:
    def __init__(self, data_train, close_train, client, model=MODEL,
                 n_islands=4, k_parents=2, cv_kwargs=None, log=print):
        self.data = data_train
        self.close = close_train
        self.client = client
        self.model = model
        self.k_parents = k_parents
        self.db = ProgramDatabase(n_islands=n_islands)
        self.cv_kwargs = cv_kwargs or {}
        self.log = log
        self.n_evaluated = 0
        self.n_rejected = 0

    def evaluate(self, code: str) -> Program | None:
        """Compile + score a candidate on the TRAIN period. None if invalid/weak."""
        try:
            fn = compile_strategy(code)
            signal = fn(self.data, alpha_tools_module())
            import pandas as pd
            if not isinstance(signal, pd.Series):
                raise TypeError("strategy did not return a pandas Series")
            diag = bt.block_cv_score(signal, self.close, **self.cv_kwargs)
        except Exception as e:                      # any failure = rejected candidate
            self.n_rejected += 1
            return None
        self.n_evaluated += 1
        if diag["reason"] != "ok":
            return None
        return Program(code, diag["score"], diag, signal)

    def seed(self, seed_code: str):
        prog = self.evaluate(seed_code)
        if prog is None:
            raise RuntimeError("seed strategy failed to evaluate — check the contract")
        for i in range(len(self.db.islands)):
            self.db.add(i, Program(prog.code, prog.score, prog.diagnostics, prog.signal))
        self.log(f"seed score={prog.score:.3f}  {prog.diagnostics['block_sharpes']}")

    def _mutate(self, parents: list[Program]) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            temperature=1.0,                        # diversity across mutations (Haiku)
            system=prompt_mod.SYSTEM,
            messages=[{"role": "user",
                       "content": prompt_mod.build_user_prompt([p.as_parent() for p in parents])}],
        )
        text = next((b.text for b in msg.content if b.type == "text"), "")
        return extract_code(text)

    def step(self) -> Program | None:
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
        if child is not None and child.score > -9.0:
            self.db.add(island, child)
        return child

    def run(self, iterations: int, reset_every: int = 50, log_every: int = 10):
        for it in range(1, iterations + 1):
            child = self.step()
            if it % reset_every == 0:
                self.db.reset_weak_islands()
            if it % log_every == 0:
                best = self.db.best()
                b = best.score if best else float("nan")
                self.log(f"[{it:4d}] best={b:.3f}  eval={self.n_evaluated} "
                         f"rej={self.n_rejected}"
                         + (f"  child={child.score:.3f}" if child else "  child=REJECT"))
        return self.db.best()


def alpha_tools_module():
    import alpha_tools
    return alpha_tools
