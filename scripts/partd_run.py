"""Replay B8 and validate each terminal fixed-receiver certificate."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.optimize import linprog

from partb_run import METRICS, load_case, master, price_beams, scene, seed_bank, timed_lp
from partc_run import CONDITIONS, CORE, SEEDS, manifest

REFERENCE_DB = 1e-4
DBCHECK = 1e-7
ACTIVE = 1e-8
FLOOR = 64 * np.finfo(float).eps
ATOM_FIELDS = ("seed", "users", "gamma", "outer", "inner", "tone", "owner",
               "atom", "added_outer", "added_inner", "power_w", "active", "ever_active")


def db_ratio(upper, lower):
    if upper <= 0 or lower <= 0:
        raise ValueError("The fixed-receiver score must be positive")
    return float(10 * np.log10(upper / lower))


def append(bank, owners, new, labels):
    return [np.column_stack((a, b)) for a, b in zip(bank, new)], [np.r_[a, b] for a, b in zip(owners, labels)]


def feasible_witness(x, h, bank, gamma, lp):
    """Compute an independently checked, feasible normalized power allocation."""
    raw = lp["y_t"]
    if raw[-1] <= 0:
        raise ValueError("Charnes–Cooper scale is nonpositive")
    powers = np.maximum(0, raw[:-1] / raw[-1])
    inequalities = lp["inequalities"]
    slack = -(inequalities @ np.r_[powers, 1.])
    defects = slack < FLOOR
    mix = 0.
    if np.any(defects):
        anchor = np.zeros_like(powers)
        index = np.arange(x.K)
        offsets = np.r_[0, np.cumsum([v.shape[1] for v in bank])[:-1]]
        desired = np.abs(np.array([np.sum(hh.conj() * vv[:, :x.K], axis=0)
                                   for hh, vv in zip(h, bank)])) ** 2
        anchor[(offsets[:, None] + index).ravel()] = (
            2 * gamma * lp["normalized_noise"] / desired).ravel()
        anchor_slack = -(inequalities @ np.r_[anchor, 1.])
        if np.min(anchor_slack) <= FLOOR:
            raise ValueError("Strict ZF witness does not satisfy the tightened constraints")
        checked = slack
        for _ in range(8):
            defects = checked < FLOOR
            if not np.any(defects):
                break
            step = float(np.max((FLOOR - checked[defects]) /
                                (anchor_slack[defects] - checked[defects])))
            powers = (1 - step) * powers + step * anchor
            mix += (1 - mix) * step
            checked = -(inequalities @ np.r_[powers, 1.])
        if np.min(checked) < -FLOOR:
            raise ValueError(f"Numerical power witness fails with minimum scaled slack {checked.min():.3e}")
    lower = float(lp["target"] @ powers / (1 + lp["clutter"] @ powers))
    relative = -slack[:-1].reshape(lp["normalized_noise"].shape) / np.sqrt(lp["normalized_noise"])
    return lower, mix, float(max(0., relative.max())), float(max(0., -slack[-1]))


def spectral_data(c, d0, safe_power):
    c_scaled = (safe_power / d0) * c
    c_min = float(np.linalg.eigvalsh(c_scaled).min())
    if c_min <= -1:
        raise ValueError("Could not bound the Charnes–Cooper scale")
    return c_scaled, c_min


def certificate(x, h, bank, a, d0, mu, price, psi, gamma, epsilon, lp, spectral):
    feasible, mix, qos, power = feasible_witness(x, h, bank, gamma, lp)
    c_scaled, c_min = spectral
    dual_psi = lp["psi_dual"]
    corrected_mu = np.maximum(0., mu)
    corrected_price = max(0., price)
    normalized_reward = a * (lp["safe_power"] / d0 / lp["scale"])
    base = normalized_reward - dual_psi * c_scaled - (
        h * corrected_mu[:, None, :]) @ h.swapaxes(1, 2).conj()
    _, _, eps_dual = price_beams(x, h, corrected_mu, base, corrected_price, gamma)
    scale_violation = max(0., corrected_price - np.sum(corrected_mu * lp["normalized_noise"]) - dual_psi)
    t_max = 1 / (1 + min(0., c_min))
    padding = FLOOR * (1 + abs(dual_psi) + abs(corrected_price) + np.max(np.abs(corrected_mu)))
    upper = float(dual_psi + t_max * (eps_dual + scale_violation) + padding)
    return dict(psi=float(psi), psi_dual=dual_psi, epsilon=float(epsilon),
                corrected_epsilon=float(eps_dual), scale_rc_violation=float(scale_violation),
                nominal_bound_db=float(10 * np.log10(1 + epsilon / psi)), audited_upper=upper,
                witness_psi=feasible, audited_bound_db=db_ratio(upper, feasible),
                witness_mix=mix, decoded_relative_qos_shortfall=qos, decoded_power_violation=power,
                min_clutter_eigenvalue=c_min, objective_scale=float(lp["scale"]),
                multiplier_clip=float(max(0., -price, -np.min(mu))),
                t=float(lp["y_t"][-1]),
                **{key: lp[key] for key in ("eq_violation", "inequality_violation",
                                           "variable_violation", "dual_column_violation")})


class Observer:
    """Track terminal powers and a stable index within each tone's growing bank."""

    def __init__(self, x):
        self.x = x
        self.steps = []
        self.origins = []
        self.ever = []
        self.terminal = None

    def __call__(self, j, k, last, u, a, c, d0, h, d, bank, owners, power,
                 mu, price, psi, gamma, epsilon, tags, lp):
        if not self.steps:
            self.origins = [[(0, 0)] * len(o) for o in owners]
            self.ever = [np.zeros(len(o), dtype=bool) for o in owners]
        threshold = ACTIVE * lp["safe_power"]
        active = [p > threshold for p in power]
        for seen, now in zip(self.ever, active):
            seen |= now
        added = sum(map(len, tags))
        self.steps.append(dict(outer=j, inner=k, psi=float(psi), epsilon=float(epsilon),
                               scale=float(lp["scale"]), pricing_pass=len(self.steps) + 1,
                               atoms=sum(map(len, owners)), added=0 if last else added,
                               terminal_candidates=added if last else 0,
                               eq_violation=lp["eq_violation"], primal_residual=lp["inequality_violation"]))
        if last:
            self.terminal = dict(u=u, a=a, c=c, d0=d0, h=h, d=d, bank=bank, owners=owners,
                                 power=power, gamma=gamma, psi=psi, mu=mu, price=price,
                                 epsilon=epsilon, lp=lp, active=active)
        else:
            for n, ids in enumerate(tags):
                self.origins[n].extend([(j, k)] * len(ids))
                self.ever[n] = np.r_[self.ever[n], np.zeros(len(ids), dtype=bool)]

    def atom_rows(self):
        t = self.terminal
        for n in range(self.x.N):
            for i, owner in enumerate(t["owners"][n]):
                j, k = self.origins[n][i]
                yield dict(tone=n, owner=int(owner), atom=i, added_outer=j, added_inner=k,
                           power_w=float(t["power"][n][i]), active=bool(t["active"][n][i]),
                           ever_active=bool(self.ever[n][i]))


def solve_fixed(x, terminal, bank, owners, spectral, threshold=1e-11, price_master=True):
    t = terminal
    profile = dict.fromkeys(METRICS, 0.)
    lp = {}
    result = master(x, t["h"], t["d"], bank, owners, t["a"], t["c"], t["d0"], profile=profile,
                    lp_audit=lp, objective_scale=t["lp"]["scale"])
    _, base, mu, price, psi, gamma = result
    if not price_master:
        feasible, mix, qos, power = feasible_witness(x, t["h"], bank, gamma, lp)
        return dict(psi=float(psi), witness_psi=feasible, witness_mix=mix,
                    decoded_relative_qos_shortfall=qos, decoded_power_violation=power), None, None
    new, labels, eps = price_beams(x, t["h"], mu, base, price, gamma, threshold=threshold)
    audit = certificate(x, t["h"], bank, t["a"], t["d0"], mu, price, psi, gamma, eps, lp, spectral)
    return audit, new, labels


def reference_run(x, terminal, spectral):
    bank, owners = terminal["bank"], terminal["owners"]
    history = []
    status = "cap"
    for _ in range(80):
        audit, new, labels = solve_fixed(x, terminal, bank, owners, spectral)
        added = sum(map(len, labels))
        history.append(dict(atoms=sum(map(len, owners)), **audit, candidate_atoms=added))
        if audit["audited_bound_db"] <= REFERENCE_DB:
            status = "converged"
            break
        if not added:
            status = "stalled"
            break
        bank, owners = append(bank, owners, new, labels)
    return status, history


def controls(x, observer, spectral, seed):
    _, _, fixed, tags, _ = seed_bank(x)
    t = observer.terminal
    anchor, _, _ = solve_fixed(x, t, fixed, tags, spectral, price_master=False)
    rng = np.random.default_rng(596100 + seed)
    random, labels = [], []
    for original, all_owners in zip(tags, t["owners"]):
        label = all_owners[len(original):]
        raw = rng.standard_normal((x.T, len(label))) + 1j * rng.standard_normal((x.T, len(label)))
        random.append(raw / np.linalg.norm(raw, axis=0))
        labels.append(label)
    random, labels = append(fixed, tags, random, labels)
    randomized, _, _ = solve_fixed(x, t, random, labels, spectral, price_master=False)
    return anchor, randomized


def paired_audit(x, seed, outer, inner, archived, atoms, scene_key):
    observer = Observer(x)
    design, _ = timed_lp(x, seed, "B8", outer, inner, observer=observer)
    if archived is not None:
        if design["atoms"] != archived["atoms"]:
            raise ValueError("Part C terminal atom counts did not replay")
        for field in ("design_streams_db", "true_streams_db", "terminal_bound_db"):
            np.testing.assert_allclose(design[field], archived[field], atol=1e-9, rtol=0,
                                       err_msg=f"Part C {field} did not replay")
    t = observer.terminal
    c = spectral_data(t["c"], t["d0"], t["lp"]["safe_power"])
    terminal = certificate(x, t["h"], t["bank"], t["a"], t["d0"], t["mu"], t["price"],
                           t["psi"], t["gamma"], t["epsilon"], t["lp"], c)
    records = list(observer.atom_rows())
    atoms.writerows({**scene_key, **r} for r in records)
    priced = [r for r in records if r["added_outer"]]
    priced_power = sum(r["power_w"] for r in priced)
    total = sum(r["power_w"] for r in records)
    status, ref_history = reference_run(x, t, c)
    ref = ref_history[-1]
    fixed, random = controls(x, observer, c, seed)
    observed_gap = db_ratio(ref["witness_psi"], terminal["psi"])
    audited_gap = db_ratio(ref["witness_psi"], terminal["witness_psi"])
    row = dict(
        status=status, feasible=design["feasible"], lp_solves=design["lp_solves"],
        pricing_rounds=len(observer.steps), productive_rounds=sum(r["added"] > 0 for r in observer.steps),
        added_atoms=len(priced), terminal_candidates=observer.steps[-1]["terminal_candidates"],
        priced_active=sum(r["active"] for r in priced), priced_ever_active=sum(r["ever_active"] for r in priced),
        priced_power_w=priced_power, initial_power_w=total - priced_power, total_power_w=total,
        priced_power_share=priced_power / total,
        receiver_sha256=hashlib.sha256(np.ascontiguousarray(t["u"]).tobytes()).hexdigest(),
        reference_solves=len(ref_history), reference_residual_db=ref["audited_bound_db"],
        reference_witness_psi=ref["witness_psi"],
        observed_gap_db=observed_gap, audited_observed_gap_db=audited_gap,
        reference_gap_upper_db=db_ratio(ref["audited_upper"], terminal["psi"]),
        nominal_violation=observed_gap > terminal["nominal_bound_db"] + DBCHECK,
        audited_violation=audited_gap > terminal["audited_bound_db"] + DBCHECK,
        fixed_to_priced_db=db_ratio(terminal["witness_psi"], fixed["witness_psi"]),
        random_to_priced_db=db_ratio(terminal["witness_psi"], random["witness_psi"]),
        fixed_psi=fixed["witness_psi"], random_psi=random["witness_psi"], **terminal,
    )
    return row, observer.steps, ref_history


def recheck_restricted(x, seed, outer, inner, original):
    """Bracket the terminal restricted LP with a separate IPM solve."""
    observer = Observer(x)
    timed_lp(x, seed, "B8", outer, inner, observer=observer)
    t = observer.terminal
    primary = original["result"]
    receiver_hash = hashlib.sha256(np.ascontiguousarray(t["u"]).tobytes()).hexdigest()
    if receiver_hash != primary["receiver_sha256"] or t["psi"] != primary["psi"]:
        raise ValueError("The original terminal receiver or LP value did not replay")
    lp = t["lp"]
    reward = np.r_[lp["target"], 0.]
    equality = np.r_[lp["clutter"], 1.]
    inequalities = lp["inequalities"]
    result = linprog(-reward, A_ub=inequalities, b_ub=np.zeros(inequalities.shape[0]),
                     A_eq=equality[None, :], b_eq=[1.], method="highs-ipm",
                     options=dict(threads=1, primal_feasibility_tolerance=1e-10,
                                  dual_feasibility_tolerance=1e-10, small_matrix_value=1e-12))
    if not result.success:
        raise RuntimeError(result.message)
    lower, mix, qos, power = feasible_witness(x, t["h"], t["bank"], t["gamma"],
                                              {**lp, "y_t": result.x})
    dual = float(-result.eqlin.marginals[0])
    multipliers = np.maximum(0., -result.ineqlin.marginals)
    residual = reward - dual * equality - inequalities.T @ multipliers
    upper = dual + max(0., residual[:-1].max()) + max(0., residual[-1]) + FLOOR * (1 + abs(dual))
    width = db_ratio(upper, lower)
    if not -DBCHECK <= width <= DBCHECK:
        raise ValueError(f"IPM restricted LP bracket is {width:.3g} dB wide")
    ref = original["reference"][-1]
    gap_lower = db_ratio(ref["witness_psi"], upper)
    gap_upper = db_ratio(ref["audited_upper"], lower)
    nominal_upper = primary["psi"] + primary["epsilon"]
    return dict(status="rechecked", receiver_sha256=receiver_hash, psi_ipm=float(-result.fun),
                witness_psi=lower, restricted_upper=float(upper), bracket_db=width,
                decoded_relative_qos_shortfall=qos, decoded_power_violation=power, witness_mix=mix,
                raw_minus_ipm_db=db_ratio(primary["psi"], lower), gap_lower_db=gap_lower,
                gap_upper_db=gap_upper, nominal_against_ipm_db=db_ratio(nominal_upper, lower),
                nominal_violation=ref["witness_psi"] > nominal_upper * 10 ** (DBCHECK / 10))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tutorial", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--part-c", type=Path, default=Path("results/partc-2026-11-22"))
    p.add_argument("--cpu", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--recheck", type=Path, help="Completed bounds.jsonl for this shard")
    args = p.parse_args()
    args.mode = "recheck" if args.recheck else "certify"
    os.sched_setaffinity(0, {args.cpu})
    data = manifest(args.tutorial, args)
    args.destination.mkdir(parents=True, exist_ok=False)
    (args.destination / "environment.json").write_text(json.dumps(data, indent=2) + "\n")
    x = load_case(args.tutorial)
    if args.recheck:
        data["original_bounds_sha256"] = hashlib.sha256(args.recheck.read_bytes()).hexdigest()
        (args.destination / "environment.json").write_text(json.dumps(data, indent=2) + "\n")
        with args.recheck.open() as original, (args.destination / "restricted.jsonl").open("x") as out:
            for line in original:
                entry = json.loads(line)
                seed, users, gamma, outer, inner = (entry["result"][k] for k in ATOM_FIELDS[:5])
                x.K = users
                x.g = np.full((x.N, users), 10 ** (gamma / 10))
                scene(x, seed)
                check = recheck_restricted(x, seed, outer, inner, entry)
                keys = dict(zip(ATOM_FIELDS[:5], (seed, users, gamma, outer, inner)))
                out.write(json.dumps({**keys, **check}, allow_nan=False) + "\n")
                out.flush()
        return
    historical = {}
    for file in args.part_c.glob("static-*/designs.jsonl"):
        for line in file.read_text().splitlines():
            row = json.loads(line)["design"]
            if row["method"] == "lp":
                historical[row["seed"], row["users"], row["gamma"], row["outer"], row["inner"]] = row
    tasks = [(seed, users, gamma) for seed in ([9999] if args.pilot else SEEDS)
             for users, gamma in CONDITIONS]
    with (args.destination / "bounds.jsonl").open("x") as log, gzip.open(
            args.destination / "terminal-atoms.csv.gz", "xt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ATOM_FIELDS, lineterminator="\n")
        writer.writeheader()
        for seed, users, gamma in tasks[args.shard::args.shards]:
            x.K = users
            x.g = np.full((x.N, users), 10 ** (gamma / 10))
            scene(x, seed)
            for outer, inner in CORE:
                keys = dict(seed=seed, users=users, gamma=gamma, outer=outer, inner=inner)
                archived = None if args.pilot else historical[tuple(keys.values())]
                try:
                    row, steps, history = paired_audit(x, seed, outer, inner, archived, writer, keys)
                except (RuntimeError, ValueError, AssertionError, np.linalg.LinAlgError) as e:
                    row, steps, history = dict(status="error", reason=str(e)), [], []
                log.write(json.dumps(dict(result={**keys, **row}, trajectory=steps,
                                          reference=history), allow_nan=False) + "\n")
                log.flush()
                print(seed, users, gamma, outer, inner, row["status"], row.get("reference_solves"),
                      row.get("reason", ""), flush=True)


if __name__ == "__main__":
    main()
