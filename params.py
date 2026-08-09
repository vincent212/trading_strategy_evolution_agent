"""
Encode/decode an LLM-declared param_space() into the continuous box the optimizer
(SciPy differential evolution) searches over.

A param_space() returns a dict: name -> spec, where spec is one of
    ("float", lo, hi)
    ("int",   lo, hi)
    ("cat",   [choice0, choice1, ...])

Integers and categoricals are searched as continuous coordinates and snapped back
by flooring, so a single gradient-free optimizer handles the mixed space.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-9


def parse_space(space: dict):
    """Return (names, bounds, kinds, extra) for the optimizer.
    bounds is a list of (lo, hi) continuous ranges; kinds/extra decode them back."""
    names, bounds, kinds, extra = [], [], [], []
    for name, spec in space.items():
        kind = spec[0]
        if kind == "float":
            lo, hi = float(spec[1]), float(spec[2])
            if hi <= lo:
                hi = lo + EPS
            bounds.append((lo, hi)); extra.append(None)
        elif kind == "int":
            lo, hi = int(spec[1]), int(spec[2])
            if hi < lo:
                lo, hi = hi, lo
            bounds.append((float(lo), float(hi) + 1.0 - EPS)); extra.append(None)
        elif kind == "cat":
            choices = list(spec[1])
            if not choices:
                raise ValueError(f"empty choices for categorical param {name!r}")
            bounds.append((0.0, len(choices) - EPS)); extra.append(choices)
        else:
            raise ValueError(f"bad param spec for {name!r}: {spec!r}")
        names.append(name); kinds.append(kind)
    return names, bounds, kinds, extra


def decode(x, names, kinds, extra) -> dict:
    """Turn a continuous vector into a concrete params dict."""
    p = {}
    for xi, name, kind, ex in zip(x, names, kinds, extra):
        if kind == "float":
            p[name] = float(xi)
        elif kind == "int":
            p[name] = int(np.floor(xi))
        else:  # cat
            idx = int(np.floor(xi))
            idx = min(max(idx, 0), len(ex) - 1)
            p[name] = ex[idx]
    return p


def midpoint(space: dict) -> dict:
    """A cheap default parameter set, used for a quick validity check before fitting."""
    p = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "float":
            p[name] = (float(spec[1]) + float(spec[2])) / 2.0
        elif kind == "int":
            p[name] = int((int(spec[1]) + int(spec[2])) // 2)
        else:
            p[name] = list(spec[1])[0]
    return p
