"""Run Part A using immutable inputs and a fresh output directory per task."""
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "first-isac-paper/experiments"))

import numpy as np
import highspy
import scs
import scipy
from scipy.optimize._highspy import _core
from threadpoolctl import threadpool_info

from parta_ao import beam_seed, covariance, run_ao
from verify_fast_beams import (
    MatlabTwister, audit, fixed_receiver, load_case, master, price_beams,
    recover, scene, seed_bank, true_score,
)

SEEDS = range(13001, 13033)
THRESHOLDS = (8, 12)
MARGINS = ("legacy", 0.0, 0.25, 0.5, 1.0)
ERROR_LEVELS = (0.01, 0.03, 0.05)
CASES = ("base", "cap_5x3", "outer_0.1", "inner_0.01", "redraw_distances",
         "range_100_500", "exponent_3.1", "power_cap_199.53")
ALT_POWER = 199.52623149688787


def setup_scene(x0, seed, gamma_db, variant):
    x = scene(copy.copy(x0), seed)
    x.g = np.full_like(x.g, 10 ** (gamma_db / 10))
    if variant in ("redraw_distances", "range_100_500", "exponent_3.1"):
        rng = MatlabTwister(seed)
        rng.complex((x.T, x.N, x.K))
        uniforms = np.array([rng.uniform() for _ in range(x.K)])
        reference = 50 + np.floor(351 * uniforms)
        if variant == "redraw_distances":
            alternate_rng = MatlabTwister(seed + 92000)
            distance = 50 + np.floor(351 * np.array([alternate_rng.uniform() for _ in range(x.K)]))
        elif variant == "range_100_500":
            distance = 100 + np.floor(401 * uniforms)
        else:
            distance = reference
        exponent = 3.1 if variant == "exponent_3.1" else 2.7
        x.H *= 10 ** ((27 * np.log10(reference) - 10 * exponent * np.log10(distance)) / 20)
    if variant == "power_cap_199.53":
        x.P = ALT_POWER
    return x


def a1_a2(x, mode, shard, shards, destination, backend):
    if mode == "startup":
        jobs = [(13999, 8, "startup", 0)]
    else:
        baseline = [(seed, g, "base", 0) for seed in SEEDS for g in THRESHOLDS]
        extra_starts = [(seed, 8, "base", i) for i in range(1, 20) for seed in list(SEEDS)[:3]]
        controls = [(seed, g, arm, 0) for arm in CASES[1:] for g in THRESHOLDS for seed in SEEDS]
        jobs = (baseline + extra_starts + controls)[shard::shards]
    options = {"cap_5x3": {"outer": 5, "inner": 3}, "outer_0.1": {"outer_tol": 0.1},
               "inner_0.01": {"inner_tol": 0.01}}
    with (destination / "ao.jsonl").open("w") as out:
        for seed, gamma, arm, init in jobs:
            y = setup_scene(x, seed, gamma, arm)
            init_seed = 240001 + 20 * (seed - 13001) + init if init else beam_seed(seed)
            row, trace = run_ao(y, init_seed, backend=backend, **options.get(arm, {}))
            row.update(seed=seed, gamma_db=gamma, arm=arm, start_index=init, init_seed=init_seed)
            if row["status"] == "solved":
                row["target_coefficient_only_true_cov_db"] = row["true_cov_db"] + 10 * np.log10(ALT_POWER / x.P)
            print(json.dumps({"design": row, "sdp_trace": trace}), file=out, flush=True)
            print(seed, gamma, arm, init, row["status"], round(row["seconds"], 2), flush=True)


def priced_lp(x, margin_setting):
    started = time.perf_counter()
    allocation_inputs = copy.copy(x)
    if margin_setting != "legacy":
        # The original helpers internally multiply their input thresholds by 1+2e-5.
        allocation_inputs.g = x.g * (10 ** (margin_setting / 10) / (1 + 2e-5))
    h, d, bank, owners, w = seed_bank(allocation_inputs)
    eta, a, c, hot, u = fixed_receiver(x, covariance(w), x.est)
    solves = 0
    for _ in range(5):
        old = eta
        best = eta
        for _ in range(3):
            p, base, mu, price, psi, gamma = master(allocation_inputs, h, d, bank, owners, a, c, hot)
            solves += 1
            candidate = recover(x, h, bank, owners, p)
            margin, power = audit(x, candidate)
            q = covariance(candidate)
            score = np.einsum("nij,nji->", a, q).real / (hot + np.einsum("nij,nji->", c, q).real)
            if margin >= -1e-4 and power <= x.P * (1 + 1e-7) and score >= best * (1 - 1e-10):
                w, best = candidate, score
            added, tags, residual = price_beams(x, h, mu, base, price, gamma)
            bank = [np.column_stack((v, more)) for v, more in zip(bank, added)]
            owners = [np.r_[initial, new] for initial, new in zip(owners, tags)]
            bound_db = 10 * np.log10(1 + residual / max(psi, 1e-16))
            if bound_db <= 0.002:
                break
        eta, a, c, hot, u = fixed_receiver(x, covariance(w), x.est)
        if 10 * np.log10(eta / old) < 0.002 and bound_db <= 0.002:
            break
    return w, u, solves, time.perf_counter() - started


def csi_margins(x, w, error):
    receive = np.einsum("rntk,nts->rnks", (x.H + error).conj(), w)
    gains = np.abs(receive) ** 2
    k = np.arange(x.K)
    desired = gains[..., k, k].copy()
    gains[..., k, k] = 0
    return 10 * np.log10(np.min(desired / ((gains.sum(axis=-1) + x.noise) * x.g), axis=(1, 2)))


def a4(x, destination, shard, shards):
    rows = [(seed, gamma) for seed in SEEDS for gamma in THRESHOLDS][shard::shards]
    result_path = destination / "csi-designs.jsonl"
    with result_path.open("w") as out, (destination / "csi-perturbations.csv").open("w", newline="") as raw:
        writer = csv.writer(raw)
        writer.writerow(("seed", "gamma_db", "delta_margin_db", "relative_error", "replicate", "min_sinr_margin_db"))
        for seed, gamma in rows:
            y = setup_scene(x, seed, gamma, "base")
            rng = np.random.default_rng(340000 + seed)
            shape = (100, *y.H.shape)
            standard = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2 * y.T)
            error = standard * np.linalg.norm(y.H, axis=1)[None, :, None, :]
            for setting in MARGINS:
                row = {"seed": seed, "gamma_db": gamma, "delta_margin_db": setting}
                try:
                    w, u, solves, seconds = priced_lp(y, setting)
                    margin, power = audit(y, w)
                    row.update(status="solved", design_db=true_score(y, covariance(w), u, y.est),
                               true_db=true_score(y, covariance(w), u, y.truth), seconds=seconds,
                               lp_solves=solves, min_sinr_margin_db=margin, power_w=power,
                               physically_feasible=bool(margin >= -1e-4 and power <= y.P * (1 + 1e-7)))
                    for i, level in enumerate(ERROR_LEVELS):
                        margins = csi_margins(y, w, level * error)
                        row[f"feasible_csi_{i}"] = int(np.count_nonzero(
                            (margins >= -1e-4) & (power <= y.P * (1 + 1e-7))))
                        writer.writerows((seed, gamma, setting, level, r, m) for r, m in enumerate(margins))
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    row.update(status="failed", reason=str(exc))
                print(json.dumps(row), file=out, flush=True)
                raw.flush()
            print(seed, gamma, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ao", "csi", "startup"))
    parser.add_argument("tutorial", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--backend", choices=("scs", "barrier", "eigencut", "incremental"), default="scs")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("Expected 0 <= shard < shards")
    data = args.tutorial / "data/covMat_BLP.mat"
    with data.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    revision = subprocess.check_output(["git", "-C", args.tutorial, "rev-parse", "HEAD"], text=True).strip()
    if digest != "99744880b1a6fe5e5f8a0ef57c1738be79fb0ad28629e68db2bfe3a8df67472d" or not revision.startswith("0f864c4"):
        parser.error("Unexpected tutorial revision or covariance file")
    args.destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(command=sys.argv, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    python=platform.python_version(), os=platform.platform(),
                    cpu_affinity=sorted(os.sched_getaffinity(0)), scs=scs.__version__,
                    highs=_core._Highs().version(), highspy_version=highspy.Highs().version(), scipy=scipy.__version__, numpy=np.__version__, backend=args.backend,
                    blas=threadpool_info(), tutorial_revision=revision, covariance_sha256=digest,
                    cpu_model=next(line.split(":", 1)[1].strip() for line in
                                   Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
                    cgroup_cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip(), gpu_used=False,
                    thread_limits=dict(blas=1, omp=1, mkl=1,
                                       highs_native=1 if args.backend == "incremental" else "auto"),
                    protocol_commit=subprocess.check_output(
                        ["git", "log", "-1", "--format=%H", "--", "PROTOCOL.md"], text=True).strip())
    (args.destination / "environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    x = load_case(args.tutorial)
    if args.mode == "csi":
        a4(x, args.destination, args.shard, args.shards)
    else:
        a1_a2(x, args.mode, args.shard, args.shards, args.destination, args.backend)


if __name__ == "__main__":
    main()
