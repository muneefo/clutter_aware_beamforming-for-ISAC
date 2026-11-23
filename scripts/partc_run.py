"""Execute the registered Part C cohorts with immutable inputs and fresh output."""
import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import highspy
import numpy as np
import scipy
from threadpoolctl import threadpool_info

from partb_run import load_case, scene, timed_ao, timed_lp

SEEDS = range(10001, 10033)
CONDITIONS = ((4, 8), (4, 12), (6, 8))
BUDGETS = ((1, 1), (2, 1), (3, 2), (5, 3), (8, 3))
CORE = ((1, 1), (3, 2), (5, 3))
METHODS = ("gaussian", "zf", "cooper", "tone", "lp")
ARM = {"gaussian": "B1", "zf": "B3", "cooper": "B4", "tone": "B5"}
QUALITY_DB = 0.8671285631736225


def run_solver(x, seed, method, outer, inner, checkpoints=None):
    if method == "lp":
        row, _ = timed_lp(x, seed, "B8", outer, inner)
        return row, []
    return timed_ao(x, seed, ARM[method], outer, inner, workers=1,
                    cold_retry=method == "tone", persistent=method == "cooper",
                    checkpoints=checkpoints, native_cone=False)[:2]


def attempt(x, seed, method, outer, inner, checkpoints=None):
    started = perf_counter()
    try:
        return run_solver(x, seed, method, outer, inner, checkpoints)
    except (TimeoutError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return dict(status="timeout" if isinstance(error, TimeoutError) else "error",
                    reason=str(error), seconds=perf_counter() - started), []


def output(stream, row, trace=(), **keys):
    print(json.dumps(dict(design={**row, **keys}, trace=trace)), file=stream, flush=True)


def static(x, args, stream):
    tasks = [(seed, users, gamma) for seed in ([9999] if args.pilot else SEEDS)
             for users, gamma in CONDITIONS]
    for seed, users, gamma in tasks[args.shard::args.shards]:
        x.K = users
        x.g = np.full((x.N, users), 10 ** (gamma / 10))
        scene(x, seed)
        shared = dict(seed=seed, users=users, gamma=gamma, warmup=seed == 9999)
        reference, trace = attempt(x, seed, "gaussian", 30, 30)
        output(stream, reference, trace, **shared, method="reference", outer=30, inner=30)
        budgets = BUDGETS if (users, gamma) == (4, 8) else CORE
        for outer, inner in budgets:
            row, trace = attempt(x, seed, "lp", outer, inner)
            output(stream, row, trace, **shared, method="lp", outer=outer, inner=inner)
        if (users, gamma) == (4, 8):
            for method in METHODS[:-1]:
                groups = [(1, BUDGETS)] if method == "cooper" else [
                    (r, [b for b in BUDGETS if b[1] == r]) for r in (1, 2, 3)]
                for inner, pairs in groups:
                    checkpoints = dict.fromkeys(b[0] for b in pairs[:-1])
                    latest, trace = attempt(x, seed, method, pairs[-1][0], inner, checkpoints)
                    for outer, cap in pairs:
                        row = checkpoints.get(outer) or latest
                        steps = row.get("sdp_solves", len(trace))
                        output(stream, row, trace[:steps], **shared, method=method, outer=outer,
                               inner=cap, effective_inner=inner, traversal_outer=pairs[-1][0])
        print(args.mode, seed, users, gamma, reference["status"], flush=True)


def manifest(tutorial, args):
    revision = subprocess.check_output(["git", "-C", tutorial, "rev-parse", "HEAD"], text=True).strip()
    with (tutorial / "data/covMat_BLP.mat").open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if revision != "0f864c45b4775fbe01789b313d81f22bebf65177" or checksum != "99744880b1a6fe5e5f8a0ef57c1738be79fb0ad28629e68db2bfe3a8df67472d":
        raise ValueError("Unexpected pinned tutorial or benchmark matrix")
    return dict(python=platform.python_version(), scipy=scipy.__version__, numpy=np.__version__,
                highs=highspy.Highs().version(), threadpools=threadpool_info(),
                cpu=next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")),
                cpu_affinity=sorted(os.sched_getaffinity(0)), os=platform.platform(),
                cpu_quota=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
                revision=revision, matrices_sha256=checksum, started_utc=datetime.now(timezone.utc).isoformat(),
                partc_source=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                mode=args.mode, pilot=args.pilot, shard=args.shard, shards=args.shards,
                lp_threads=1, tone_workers=1, cores_per_lane=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("static", "warm", "scaling"))
    p.add_argument("tutorial", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--cpu", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    data = manifest(args.tutorial, args)
    args.destination.mkdir(parents=True, exist_ok=False)
    (args.destination / "environment.json").write_text(json.dumps(data, indent=2) + "\n")
    x = load_case(args.tutorial)
    with (args.destination / "designs.jsonl").open("x") as stream:
        if args.mode == "static":
            static(x, args, stream)
        elif args.mode == "warm":
            from partc_warm import chains
            chains(x, args, stream)
        else:
            from partc_scaling import scale_sweep
            scale_sweep(x, args, stream)


if __name__ == "__main__":
    main()
