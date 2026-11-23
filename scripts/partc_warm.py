"""Correlated channel trajectories with matched dictionary and simplex-basis reuse."""
from time import perf_counter
from types import SimpleNamespace

import highspy
import numpy as np

from parta_incremental import IncrementalCutSDP
from partb_run import METRICS, covariance, pair_sum, receiver, score
from partc_run import QUALITY_DB, attempt, output
from verify_fast_beams import audit, price_beams, recover, seed_bank


LOWER = highspy.HighsBasisStatus.kLower
BASIC = highspy.HighsBasisStatus.kBasic


def keep_basis(model):
    basis = model.model.getBasis()
    if not basis.valid:
        return None
    retained = []
    for bank, owners, indices in zip(model.bank, model.owners, model.indices):
        retained.append([(bank[j].copy(), owners[j]) for j, col in enumerate(indices)
                         if basis.col_status[col] == BASIC])
    return retained, basis.col_status[0], list(basis.row_status)


def transfer_basis(model, retained, t_status, row_status):
    basis = highspy.HighsBasis()
    basis.row_status = row_status
    columns = [t_status]
    for vectors, old in zip(model.bank, retained):
        columns += [BASIC] * len(old) + [LOWER] * (len(vectors) - len(old))
    basis.col_status = columns
    model.initial_basis = basis


def append_priced(x, h, model, a, c, objective_scale, gamma, result):
    mu = -result.ineqlin.marginals[:-1].reshape(x.N, x.K)
    price = -result.ineqlin.marginals[-1]
    psi = -result.fun
    base = a / objective_scale - psi * c - (h * mu[:, None, :]) @ h.swapaxes(1, 2).conj()
    new, tags, _ = price_beams(x, h, mu, base, price, gamma)
    for n, (vectors, owners) in enumerate(zip(new, tags)):
        model.bank[n].extend(vectors.T)
        model.owners[n].extend(((owners + 1) % (x.K + 1)).tolist())


def warm_lp(x, reference, prior=None, mode="cold"):
    start = perf_counter()
    h, d, vectors, owners, w = seed_bank(x)
    power_unit = x.P * (1 - 1e-8) / x.N
    gamma = x.g * (1 + 2e-5)
    hnt = h.transpose(0, 2, 1)
    problem = SimpleNamespace(n=x.N, k=x.K, t=x.T, normalized=hnt,
                              gamma=gamma, noise=d * x.P / power_unit)
    model = IncrementalCutSDP(problem, fractional=True)
    model.bank = [list(v.T) for v in vectors]
    model.owners = [((o + 1) % (x.K + 1)).tolist() for o in owners]
    retained = 0
    if prior is not None and mode != "cold":
        columns, t_status, row_status = prior
        retained = sum(map(len, columns))
        for n, saved in enumerate(columns):
            model.bank[n][:0] = [v for v, _ in saved]
            model.owners[n][:0] = [o for _, o in saved]
        if mode == "basis":
            transfer_basis(model, columns, t_status, row_status)
    q = covariance(w)
    profile = dict.fromkeys(METRICS, 0.)
    eta, a, c, d0, u = receiver(x, q, profile)
    hit = False
    solves = 0
    first_simplex = None
    status = "solved"
    for outer in range(1, 9):
        best = eta
        f = power_unit / d0
        a_scaled, c_scaled = a * f, c * f
        scale = np.max(np.abs(a_scaled))
        objective = a_scaled / scale
        for inner in range(3):
            result, widths = model.restricted(objective, c_scaled)
            solves += 1
            if solves == 1:
                first_simplex = model.model.getInfo().simplex_iteration_count
            if not result.success or result.x[-1] <= 0:
                status = f"master_{result.status}"
                break
            p = np.split(result.x[:-1] * power_unit / result.x[-1], np.cumsum(widths)[:-1])
            cand = recover(x, h, [np.column_stack(bank) for bank in model.bank],
                           [(np.asarray(tags) - 1) % (x.K + 1) for tags in model.owners], p)
            cand_q = covariance(cand)
            margin, power = audit(x, cand)
            objective_value = pair_sum(a, cand_q) / (d0 + pair_sum(c, cand_q))
            if margin >= -1e-4 and power <= x.P * (1 + 1e-7) and objective_value >= best * (1 - 1e-10):
                best, w, q = objective_value, cand, cand_q
            if inner < 2:
                append_priced(x, h, model, a_scaled, c_scaled, scale, gamma, result)
        if status != "solved":
            break
        eta, a, c, d0, u = receiver(x, q, profile)
        hit = reference is not None and reference - 10 * np.log10(eta) <= QUALITY_DB
        if hit or outer == 8:
            break
        append_priced(x, h, model, a_scaled, c_scaled, scale, gamma, result)
    seconds = perf_counter() - start
    save_at = perf_counter()
    saved = keep_basis(model) if status == "solved" and mode == "basis" else None
    saved_seconds = perf_counter() - save_at
    row = dict(status=status, seconds=seconds, reference_design_db=reference, hit=bool(hit),
               lp_solves=solves, outer_steps=outer, retained_columns=retained,
               basis_requested=mode == "basis" and prior is not None, basis_status=model.basis_status,
               basis_saved=saved is not None, basis_save_seconds=saved_seconds,
               first_simplex_iterations=first_simplex)
    if status == "solved":
        score(x, q, w, u, row)
        row["hit"] = row["hit"] and row["feasible"]
    return row, saved


def chains(x, args, stream):
    x.K = 4
    x.g = np.full((x.N, x.K), 10 ** 0.8)
    seeds = [21099] if args.pilot else list(range(21101, 21133))
    for seed in seeds[args.shard::args.shards]:
        rng = np.random.default_rng(seed)
        distances = rng.integers(50, 401, x.K)
        gain = 10 ** ((-30 - 27 * np.log10(distances)) / 20)
        shape = (x.N, x.T, x.K)
        draws = [(rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * gain / np.sqrt(2)
                 for _ in range(4)]
        x.H = draws[0]
        warmup = seed == 21099
        prep, initial = warm_lp(x, None, mode="basis")
        output(stream, prep, seed=seed, chain=seed, frame=0, rho=None, method="preparation",
               warmup=warmup, distances=distances.tolist())
        for rho in (.9, .99):
            history = initial
            x.H = draws[0]
            for frame, innovation in enumerate(draws[1:], 1):
                x.H = rho * x.H + np.sqrt(1 - rho ** 2) * innovation
                ref, trace = attempt(x, seed + 100000 * frame, "gaussian", 30, 30)
                output(stream, ref, trace, seed=seed, chain=seed, frame=frame, rho=rho, method="reference",
                       warmup=warmup)
                target = ref.get("design_streams_db") if ref.get("feasible") else None
                for mode in ("cold", "dictionary", "basis"):
                    row, next_history = warm_lp(x, target, history, mode)
                    output(stream, row, seed=seed, chain=seed, frame=frame, rho=rho, method=mode, warmup=warmup)
                    if mode == "basis" and next_history is not None:
                        history = next_history
        print(args.mode, seed, prep["status"], flush=True)
