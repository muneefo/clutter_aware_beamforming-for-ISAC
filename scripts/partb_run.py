"""Run the matched Part B Python ladder with exclusive timing by arm."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ARGS = argparse.ArgumentParser(description=__doc__)
ARGS.add_argument("tutorial", type=Path)
ARGS.add_argument("output", type=Path)
ARGS.add_argument("--threads", type=int, choices=(1, 2), default=1)
ARGS.add_argument("--pilot", action="store_true")
ARGS.add_argument("--recover-b5-from", type=Path)
ARGS.add_argument("--b4-persistent", action="store_true", help="Run the matched persistent-master B4 only")
if __name__ == "__main__":
    options = ARGS.parse_args()
    cores = sorted(os.sched_getaffinity(0))[:options.threads]
    if len(cores) != options.threads:
        ARGS.error("The requested CPU affinity is unavailable")
    os.sched_setaffinity(0, cores)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(options.threads)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "first-isac-paper/experiments"))

import highspy
import numpy as np
import scs
import scipy
from scipy.optimize import OptimizeWarning
from threadpoolctl import threadpool_info, threadpool_limits

from parta_ao import JointTransmitSDP, beam_seed, covariance, pair_sum, recover_streams
from parta_incremental import IncrementalCutSDP
from partb_sdp import CooperSDP, TonePriceSDP
from verify_fast_beams import MatlabTwister, audit, load_case, master, price_beams, recover, scene, seed_bank, true_score

warnings.filterwarnings("ignore", message=r"Unrecognized options detected: .*threads", category=OptimizeWarning)
METRICS = ("build", "solve", "price", "mvdr", "kernel", "recovery", "parallel_tone_wall")
ARMS = (
    ("B1", 30, 30), ("B2", 30, 30), ("B3", 5, 3), ("B3", 3, 2),
    ("B4", 5, 1), ("B4", 3, 1), ("B5", 3, 2),
    ("B6", 5, 3), ("B7", 5, 3), ("B8", 5, 3),
    ("B6", 3, 2), ("B7", 3, 2), ("B8", 3, 2),
)
SEEDS = range(16001, 16033)
WARMUP = 15999


def receiver(x, q, profile):
    """Same forward kernel, MVDR and adjoint as the archived solver."""
    start = perf_counter()
    noise = x.est.apply(q) + x.Reta
    profile["kernel"] += perf_counter() - start

    start = perf_counter()
    u = np.linalg.solve(noise, x.b[:, :, None])[:, :, 0]
    u /= np.einsum("ni,ni->n", x.b.conj(), u)[:, None]
    profile["mvdr"] += perf_counter() - start

    vv = u[:, :, None] * u[:, None, :].conj()
    d0 = np.einsum("nij,nji->", vv, x.Reta).real
    gain = x.sig * np.abs(np.einsum("ni,ni->n", u.conj(), x.b)) ** 2
    a = gain[:, None, None] * x.a[:, :, None] * x.a[:, None, :].conj()
    start = perf_counter()
    c = x.est.apply(vv, True)
    profile["kernel"] += perf_counter() - start
    return pair_sum(a, q) / (d0 + pair_sum(c, q)), a, c, d0, u


def score(x, q, w, u, row):
    stream_q = covariance(w)
    sinr, watts = audit(x, w)
    row.update(design_cov_db=true_score(x, q, u, x.est),
               design_streams_db=true_score(x, stream_q, u, x.est),
               true_cov_db=true_score(x, q, u, x.truth),
               true_streams_db=true_score(x, stream_q, u, x.truth),
               min_sinr_margin_db=sinr, power_w=watts,
               feasible=bool(sinr >= -1e-4 and watts <= x.P * (1 + 1e-7)))


def native_scs_check(problem, cut_solver, a, c, eta, sensing, users):
    """Call native SCS on the full PSD cones from a certified warm point."""
    numerical = a - eta * c
    matrix = numerical / np.max(np.abs(numerical))
    b = -np.r_[np.zeros(problem.cost.size), problem.real.pack(matrix).ravel(),
               np.repeat(problem.complex.pack(matrix), problem.k, axis=0).ravel()]
    q = sensing + users.sum(axis=1)
    actual = np.einsum("nkij,nji->nk", problem.grams, q).real
    desired = np.einsum("nkij,nkji->nk", problem.grams, users).real
    qos_slack = (1 + 1 / problem.gamma) * desired - actual - problem.noise
    power_slack = problem.n - np.trace(q, axis1=-1, axis2=-2).real.sum()
    y = np.r_[np.maximum(0, np.r_[power_slack, qos_slack.ravel()]),
              problem.real.pack(sensing).ravel(), problem.complex.pack(users).ravel()]
    z = cut_solver.last_dual
    model = scs.SCS({"A": problem.matrix, "b": b, "c": problem.cost}, problem.cone,
                    eps_abs=1e-6, eps_rel=1e-6, max_iters=12000, verbose=False)
    result = model.solve(warm_start=True, x=z, y=y, s=b - problem.matrix @ z)
    return result["info"]


def timed_ao(x, seed, arm, outer, inner, workers, check_native=False, cold_retry=False, persistent=False,
             checkpoints=None, native_cone=True):
    profile = dict.fromkeys(METRICS, 0.)
    start = perf_counter()
    build_at = perf_counter()
    if arm == "B1":
        raw = MatlabTwister(beam_seed(seed)).complex((x.T, x.S, x.N)).transpose(2, 0, 1)
        w0 = raw * np.sqrt(x.P) / np.linalg.norm(raw)
    else:
        *_, w0 = seed_bank(x)
    q = covariance(w0)
    problem = JointTransmitSDP(x, native_cone=native_cone)
    if arm == "B4":
        optimizer = CooperSDP(problem, persistent=persistent)
    elif arm == "B5":
        optimizer = TonePriceSDP(problem, workers, cold_retry)
    else:
        optimizer = IncrementalCutSDP(problem)
    profile["build"] += perf_counter() - build_at
    old, a, c, d0, u = receiver(x, q, profile)
    eta = old
    trace = []
    diagnostic = None
    excluded = 0.
    status = "solved"
    w = None
    try:
        for j in range(1, outer + 1):
            factor = problem.power_unit / d0
            aa, cc = a * factor, c * factor
            for k in range(1, inner + 1):
                sensing, users, info = optimizer.solve(aa, cc, eta)
                info.update(outer=j, inner=k)
                trace.append(info)
                if users is None:
                    status = info["status"]
                    break
                if check_native and diagnostic is None:
                    diagnostic = native_scs_check(problem, optimizer, aa, cc, eta, sensing, users)
                qp = sensing + users.sum(axis=1)
                updated = pair_sum(aa, qp) / (1 + pair_sum(cc, qp))
                relative_delta = abs(updated - eta) / max(abs(eta), 1e-30)
                eta = updated
                if relative_delta <= 0.001:
                    break
            if status != "solved":
                break
            q = qp * problem.power_unit
            if abs(10 * np.log10(eta / old)) < .01 or j == outer:
                break
            if checkpoints is not None and j in checkpoints:
                endpoint_at = perf_counter()
                _, endpoint_w, _, _ = recover_streams(x, q, users * problem.power_unit)
                endpoint = dict(status="solved", seconds=perf_counter() - start - excluded,
                                outer_steps=j, lp_solves=0, sdp_solves=len(trace),
                                price_evals=sum(r.get("price_evals", 0) for r in trace))
                score(x, q, endpoint_w, u, endpoint)
                checkpoints[j] = endpoint
                excluded += perf_counter() - endpoint_at
            old = eta
            _, a, c, d0, u = receiver(x, q, profile)
        if status == "solved":
            recover_at = perf_counter()
            _, w, _, _ = recover_streams(x, q, users * problem.power_unit)
            profile["recovery"] += perf_counter() - recover_at
    finally:
        if arm == "B5":
            optimizer.close()
    seconds = perf_counter() - start - excluded
    for key, value in optimizer.profile.items():
        profile[key] += value
    row = dict(seconds=seconds, status=status, outer_steps=j, sdp_solves=len(trace),
               dinkelbach_solves=0 if arm == "B4" else len(trace), lp_solves=0,
               price_evals=sum(r.get("price_evals", 0) for r in trace),
               bisection_steps=sum(r.get("bisection_steps", 0) for r in trace),
               local_masters=sum(r.get("local_masters", r["iter"]) for r in trace),
               timings={**profile, "other": max(0., seconds - sum(profile.values()))})
    if arm == "B5":
        row["tone_workers"] = workers
        row["cold_retries"] = optimizer.cold_retries
        row["tone_worker_seconds"] = optimizer.worker_profile
    if w is not None:
        score(x, q, w, u, row)
    return row, trace, diagnostic


def timed_lp(x, seed, arm, outer, inner, schedule=None, observer=None):
    profile = dict.fromkeys(METRICS, 0.)
    start = perf_counter()
    build_at = perf_counter()
    h, d, bank, owners, w = seed_bank(x)
    rng = np.random.default_rng(370000 + seed)
    profile["build"] += perf_counter() - build_at
    q = covariance(w)
    eta, a, c, d0, u = receiver(x, q, profile)
    plan = []
    residual = None
    for j in range(outer):
        best = eta
        for k in range(inner):
            lp_audit = {} if observer is not None else None
            power, base, mu, price, psi, gamma = master(
                x, h, d, bank, owners, a, c, d0, profile=profile, lp_audit=lp_audit)
            recover_at = perf_counter()
            candidate = recover(x, h, bank, owners, power)
            margin, p = audit(x, candidate)
            check = covariance(candidate)
            candidate_eta = pair_sum(a, check) / (d0 + pair_sum(c, check))
            if margin >= -1e-4 and p <= x.P * (1 + 1e-7) and candidate_eta >= best * (1 - 1e-10):
                q, w, best = check, candidate, candidate_eta
            profile["recovery"] += perf_counter() - recover_at

            last = j == outer - 1 and k == inner - 1
            if arm == "B8":
                price_at = perf_counter()
                new, tags, residual = price_beams(x, h, mu, base, price, gamma)
                profile["price"] += perf_counter() - price_at
                if observer is not None:
                    observer(j + 1, k + 1, last, u, a, c, d0, h, d, bank, owners, power,
                             mu, price, psi, gamma, residual, tags, lp_audit)
                if last:
                    continue
                plan.append([tags_n.tolist() for tags_n in tags])
            elif arm == "B7" and not last:
                tags = [np.asarray(labels, dtype=int) for labels in schedule[len(plan)]]
                new = []
                for ids in tags:
                    raw = rng.standard_normal((x.T, len(ids))) + 1j * rng.standard_normal((x.T, len(ids)))
                    new.append(raw / np.linalg.norm(raw, axis=0))
                plan.append([tags_n.tolist() for tags_n in tags])
            else:
                continue
            build_at = perf_counter()
            bank = [np.column_stack((v, add)) for v, add in zip(bank, new)]
            owners = [np.r_[ids, tag] for ids, tag in zip(owners, tags)]
            profile["build"] += perf_counter() - build_at
        eta, a, c, d0, u = receiver(x, q, profile)
    seconds = perf_counter() - start
    row = dict(status="solved", seconds=seconds, outer_steps=outer, lp_solves=outer * inner, sdp_solves=0,
               dinkelbach_solves=0, price_evals=0, local_masters=0, atoms=sum(map(len, owners)),
               terminal_bound_db=(float(10 * np.log10(1 + residual / psi)) if residual is not None and psi > 0 else None),
               timings={**profile, "other": max(0., seconds - sum(profile.values()))})
    score(x, q, w, u, row)
    return row, plan


def run(options):
    root = options.tutorial.resolve()
    revision = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    with (root / "data/covMat_BLP.mat").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if revision != "0f864c45b4775fbe01789b313d81f22bebf65177" or digest != "99744880b1a6fe5e5f8a0ef57c1738be79fb0ad28629e68db2bfe3a8df67472d":
        raise ValueError("Unexpected tutorial revision or benchmark matrices")
    options.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, highs=highspy.Highs().version(), scs=scs.__version__,
                    cpu=next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                             if l.startswith("model name")),
                    threads=options.threads, cores=sorted(os.sched_getaffinity(0)), threadpools=threadpool_info(),
                    quota=Path("/sys/fs/cgroup/cpu.max").read_text().strip(), revision=revision,
                    b4_master="persistent" if options.b4_persistent else "rebuilt",
                    tone_worker_blas_threads=1, highs_master_threads=1,
                    matrices_sha256=digest, command=sys.argv, git=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True).strip(), started_utc=datetime.now(timezone.utc).isoformat())
    (options.output / "environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    x = load_case(root)
    if options.recover_b5_from:
        entries = [json.loads(line)["design"] for line in options.recover_b5_from.read_text().splitlines()]
        seeds = [r["seed"] for r in entries if r["arm"] == "B5" and r["status"] != "solved" and not r["warmup"]]
        jobs = [("B5", 3, 2)]
    else:
        seeds = [WARMUP] if options.pilot else [WARMUP, *SEEDS]
        jobs = [("B4", 5, 1), ("B4", 3, 1)] if options.b4_persistent else ARMS
    with (options.output / "designs.jsonl").open("x") as out, (options.output / "schedules.jsonl").open("x") as plans:
        for seed in seeds:
            scene(x, seed)
            schedules = {}
            budgets = ((3, 2), (5, 3)) if not options.recover_b5_from and not options.b4_persistent else ()
            for outer, inner in budgets:
                t0 = perf_counter()
                _, schedule = timed_lp(x, seed, "B8", outer, inner)
                schedules[outer, inner] = schedule
                print(json.dumps(dict(seed=seed, outer=outer, inner=inner, labels=schedule,
                                      prep_seconds=perf_counter() - t0)), file=plans, flush=True)
            order = np.random.default_rng(400000 + seed).permutation(len(jobs))
            for index in order:
                arm, outer, inner = jobs[index]
                info, trace = None, []
                try:
                    if arm in ("B6", "B7", "B8"):
                        row, plan = timed_lp(x, seed, arm, outer, inner, schedules[outer, inner])
                        if arm in ("B7", "B8") and plan != schedules[outer, inner]:
                            raise RuntimeError("B7/B8 matched atom schedule differs")
                    elif arm == "B5":
                        with threadpool_limits(limits=1, user_api="blas"):
                            row, trace, info = timed_ao(
                                x, seed, arm, outer, inner, options.threads,
                                cold_retry=options.recover_b5_from is not None)
                        row["tone_worker_blas_threads"] = 1
                    else:
                        row, trace, info = timed_ao(
                            x, seed, arm, outer, inner, options.threads,
                            check_native=options.pilot and arm == "B1", persistent=options.b4_persistent)
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                    row = dict(status="error", exception=str(e))
                row.update(arm=arm, seed=seed, outer=outer, inner=inner, threads=options.threads,
                           warmup=seed == WARMUP)
                print(json.dumps(dict(design=row, solver_trace=trace, native_scs=info)), file=out, flush=True)
                print(seed, options.threads, arm, outer, inner, row["status"], row.get("seconds"), flush=True)


if __name__ == "__main__":
    run(options)
