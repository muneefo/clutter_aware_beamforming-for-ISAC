"""Check the rank-one pricing paths against the dense Hermitian oracle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import parte_teacher as teacher
import numpy as np


def run(args):
    os.sched_setaffinity(0, {args.cpu})
    scenes = teacher.Scenes(args.tutorial)
    fast = teacher.principal_prices
    audit = dict(calls=0, owners=0, dense_fallbacks=0, maximum_price_error=0.,
                 maximum_direction_objective_error=0., maximum_negative_upper_error=0.,
                 accelerated_seconds=0., dense_seconds=0.)

    def compare(base, h, strength):
        started = perf_counter()
        value, bound, vectors, fallback = fast(base, h, strength)
        audit["accelerated_seconds"] += perf_counter() - started
        h = h.transpose(0, 2, 1)
        user = base[:, None] + strength[..., None, None] * h[..., :, None] * h[..., None, :].conj()
        matrices = np.concatenate((user, base[:, None]), axis=1)
        started = perf_counter()
        exact = np.linalg.eigh(matrices)[0][..., -1]
        audit["dense_seconds"] += perf_counter() - started
        scores = np.einsum("nki,nkij,nkj->nk", vectors.conj(), matrices, vectors).real
        audit["calls"] += 1
        audit["owners"] += exact.size
        audit["dense_fallbacks"] += fallback
        errors = dict(maximum_price_error=np.max(abs(exact - value)),
                      maximum_direction_objective_error=np.max(abs(exact - scores)),
                      maximum_negative_upper_error=np.max(exact - bound))
        for key, error in errors.items():
            audit[key] = max(audit[key], float(error))
        return value, bound, vectors, fallback

    teacher.principal_prices = compare
    rows = []
    for sid in (*teacher.PILOT, 510036, 510144):
        x, key = scenes.make(sid)
        _, info = teacher.Teacher(x).label(80, 30)
        rows.append({**key, **info})
    if max(audit["maximum_price_error"], audit["maximum_direction_objective_error"]) > 1e-8:
        raise ValueError("Secular and dense oracle prices disagree")
    if audit["maximum_negative_upper_error"] > 1e-9:
        raise ValueError("Principal eigenvalue upper audit failed")
    with args.output.open("x") as f:
        json.dump(dict(numerical_agreement=audit, pilot=rows), f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tutorial", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--cpu", type=int, default=0)
    run(p.parse_args())
