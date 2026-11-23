"""Generate the Part E independent ideal-map scenes and AO teacher labels."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from types import SimpleNamespace

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

from parta_incremental import IncrementalCutSDP
from partb_run import METRICS, covariance, load_case, receiver, seed_bank
from partc_run import manifest
from partc_scaling import ScattererKernel

TRAIN = range(510000, 515000)
PILOT = range(509900, 509910)
HELD_OUT = range(520000, 520064)
ORIGINAL = range(530000, 530064)
GATE_DB = .01
ACTIVE = 1e-8
FLOOR = 64 * np.finfo(float).eps


def rng(stream, scene_id):
    return np.random.default_rng(np.random.SeedSequence([stream, scene_id]))


class Scenes:
    def __init__(self, tutorial):
        file = tutorial / "data/covMat_BLP.mat"
        data = loadmat(file, variable_names=["para", "theta_cc_all", "sigma2_cc_all"],
                       simplify_cells=True)
        p = data["para"]
        self.baseline = load_case(tutorial)
        self.power_profiles = data["sigma2_cc_all"].T
        self.freq = 1 + np.arange(self.baseline.N) * p["deltaf"] / p["fc"]

    def make(self, scene_id):
        scene = copy.copy(self.baseline)
        ch = rng(71001, scene_id)
        users, gamma = int(ch.integers(3, 7)), int(ch.integers(6, 15))
        original = scene_id in ORIGINAL
        if original:
            users, gamma = 4, 8
        scene.K = users
        scene.g = np.full((scene.N, users), 10 ** (gamma / 10))
        distances = ch.integers(50, 401, scene.K)
        gain = 10 ** ((-30 - 27 * np.log10(distances)) / 20)
        shape = (scene.N, scene.T, scene.K)
        scene.H = (ch.standard_normal(shape) + 1j * ch.standard_normal(shape)) * gain / np.sqrt(2)
        if not original:
            cr = rng(71002, scene_id)
            size = self.power_profiles.shape[1]
            angles = np.sin(np.deg2rad(cr.uniform(-90, 90, size)))
            power_indices = cr.integers(size, size=size)
            powers = self.power_profiles[:, power_indices] * cr.lognormal(-.35**2 / 2, .35, size)
            steering = np.exp(-1j * np.pi * self.freq[:, None, None] *
                              np.arange(scene.T)[None, :, None] * angles) / np.sqrt(scene.T)
            scene.est = scene.truth = ScattererKernel(steering, powers)
        return scene, dict(scene_id=scene_id, map_id="benchmark" if original else scene_id,
                           users=users, gamma_db=gamma, distances_m=distances.tolist())


def check_choi(x):
    vectors = x.est.vectors[0].T
    powers = x.est.powers[0]
    kraus = (vectors[:, :, None] * vectors[:, None, :].conj())
    factors = kraus.transpose(0, 2, 1).reshape(len(vectors), -1).T * np.sqrt(powers)
    choi = factors @ factors.conj().T
    e, u = np.linalg.eigh(choi)
    clip = (u * np.maximum(0, e)) @ u.conj().T
    h = x.H[0]
    q = h @ h.conj().T / np.linalg.norm(h)**2
    c_q = np.einsum("iajb,ij->ab", clip.reshape(x.T, x.R, x.T, x.R), q)
    v = x.est.vectors[0]
    target = (v * (powers * np.sum(v.conj() * (q @ v), axis=0).real)) @ v.conj().T
    rel = float(np.linalg.norm(c_q - target) / np.linalg.norm(target))
    if rel > 1e-10:
        raise ValueError(f"Choi PSD projection changes the ideal operator by {rel:.3g}")
    return dict(choi_min_eigenvalue=float(e.min()), choi_max_eigenvalue=float(e.max()),
                relative_projection_difference=rel)


def herm(a):
    return (a + a.swapaxes(-1, -2).conj()) / 2


def price_matrices(h, gamma, a, c, dual, mu):
    grams = h.transpose(0, 2, 1)
    grams = grams[..., :, None] * grams[..., None, :].conj()
    base = a - dual * c - (h * mu[:, None, :]) @ h.swapaxes(1, 2).conj()
    users = base[:, None] + (mu * (1 + 1 / gamma))[..., None, None] * grams
    return herm(np.concatenate((users, base[:, None]), axis=1))


def principal_prices(base, h, strength):
    """Principal directions of the shared base and its positive rank-one updates."""
    beta, modes = np.linalg.eigh(base)
    top = beta[:, -1:]
    largest = np.max(np.abs(beta), axis=1, keepdims=True)
    coords = (modes.swapaxes(1, 2).conj() @ h).transpose(0, 2, 1)
    weights = strength[..., None] * np.abs(coords)**2
    ceiling = weights.sum(axis=-1)
    lower = weights[..., -1]
    shift = lower.copy()
    drops = (top - beta)[:, None]
    active = ceiling > 1e-13
    pad = FLOOR * (1 + largest + ceiling)
    for _ in range(10):
        divisor = np.maximum(1e-75, drops + shift[..., None])
        q = weights / divisor
        gradient = (q / divisor).sum(axis=-1)
        update = np.divide(q.sum(axis=-1) - 1, gradient, out=np.zeros_like(shift), where=gradient > 0)
        shift = np.clip(shift + update, lower, ceiling)
    upper_shift = shift + pad
    verified = (weights / (drops + upper_shift[..., None])).sum(axis=-1) <= 1 - 8 * np.finfo(float).eps
    candidates = (modes[:, None] @ (coords / np.maximum(1e-75, drops + shift[..., None]))[..., None])[..., 0]
    top_modes = np.broadcast_to(modes[:, None, :, -1], candidates.shape)
    candidates = np.where(active[..., None], candidates, top_modes)
    directions = candidates / np.linalg.norm(candidates, axis=-1, keepdims=True)
    value = top + np.where(active, shift, 0.)
    upper = top + np.where(active, upper_shift, ceiling + pad)
    needs_dense = active & (~verified | (lower < 1e-13 * ceiling))
    if np.any(needs_dense):
        tone, owner = np.where(needs_dense)
        cols = h[tone, :, owner]
        updated = base[tone] + strength[tone, owner, None, None] * cols[..., None] * cols[:, None, :].conj()
        eig, vec = np.linalg.eigh(updated)
        directions[needs_dense] = vec[..., -1]
        value[needs_dense] = eig[..., -1]
        upper[needs_dense] = eig[..., -1] + pad[needs_dense]
    return (np.column_stack((value, top)), np.column_stack((upper, top + FLOOR * (1 + largest))),
            np.concatenate((directions, modes[:, None, :, -1]), axis=1), int(needs_dense.sum()))


def proposal_basis(x, h, bank):
    zf = np.stack([v[:, :x.K].T for v in bank])
    target = np.broadcast_to(x.a[:, None], zf.shape)
    perp = np.broadcast_to(np.stack([v[:, x.K] for v in bank])[:, None], zf.shape)
    return np.stack((zf, h.transpose(0, 2, 1), target, perp), axis=-1)


def project(basis, matrices):
    return herm(basis.swapaxes(-1, -2).conj() @ matrices @ basis)


class Teacher:
    def __init__(self, x):
        self.x = x
        self.h, noise, self.bank, self.owners, self.start = seed_bank(x)
        self.safe_power = x.P * (1 - 1e-8)
        self.gamma = x.g * (1 + 2e-5)
        self.noise = noise * x.P / self.safe_power
        self.row_scale = self.noise ** -.75
        self.problem = SimpleNamespace(n=x.N, k=x.K, t=x.T, power_cap=1.,
                                       normalized=self.h.transpose(0, 2, 1)
                                       * np.sqrt(self.row_scale)[..., None], gamma=self.gamma,
                                       noise=self.noise * self.row_scale)
        zf = np.stack([v[:, :x.K] for v in self.bank])
        self.anchor = 2 * self.gamma * self.noise / np.abs(np.sum(self.h.conj() * zf, axis=1)) ** 2

    def new_master(self, carried):
        model = IncrementalCutSDP(self.problem, fractional=True)
        for option, value in (("primal_feasibility_tolerance", 1e-10),
                              ("dual_feasibility_tolerance", 1e-10), ("small_matrix_value", 1e-12)):
            if int(model.model.setOptionValue(option, value)) != 0:
                raise ValueError(f"HiGHS rejected {option}")
        model.bank = [list(v.T) + [vv for vv, _ in p] for v, p in zip(self.bank, carried)]
        model.owners = [((o + 1) % (self.x.K + 1)).tolist() + [oo for _, oo in p]
                        for o, p in zip(self.owners, carried)]
        return model

    def witness(self, model, weights):
        """Audit the decoded original QoS and make a separate feasible witness."""
        slack, anchor_slack, vectors, gains = [], [], [], []
        for n, (b, o, p) in enumerate(zip(model.bank, model.owners, weights)):
            v = np.asarray(b)
            q = np.abs(self.h[n].conj().T @ v.T) ** 2 * model.signs[n, o].T
            slack.append(q @ p - self.noise[n])
            anchor_slack.append(q[:, :self.x.K] @ self.anchor[n] - self.noise[n])
            vectors.append(v)
            gains.append(q)
        slack, anchor_slack = np.asarray(slack), np.asarray(anchor_slack)
        budget = 1 - sum(float(p.sum()) for p in weights)
        anchor_budget = 1 - float(self.anchor.sum())
        if min(anchor_budget, float(anchor_slack.min())) <= 0:
            raise ValueError("Strict teacher ZF witness is infeasible")
        # Leave a margin for cancellation in the independently reevaluated QoS sums.
        floor = np.r_[1e-8 * self.noise.ravel(), FLOOR]
        s = np.r_[slack.ravel(), budget]
        z = np.r_[anchor_slack.ravel(), anchor_budget]
        mix = 0.
        for _ in range(8):
            bad = s < floor
            if np.any(bad):
                step = float(np.max((floor[bad] - s[bad]) / (z[bad] - s[bad])))
                mix += (1 - mix) * step
            audited_weights = [(1 - mix) * p for p in weights]
            for p, anchors in zip(audited_weights, self.anchor):
                p[:self.x.K] += mix * anchors
            checked_qos = np.stack([g @ p for g, p in zip(gains, audited_weights)]) - self.noise
            s = np.r_[checked_qos.ravel(), 1 - sum(p.sum() for p in audited_weights)]
            if np.all(s >= 0):
                break
        else:
            raise ValueError("Teacher witness fails the independent feasibility check")
        q = np.empty((self.x.N, self.x.T, self.x.T), complex)
        for n, (v, p) in enumerate(zip(vectors, audited_weights)):
            q[n] = (v.T * p) @ v.conj()
        return q, audited_weights, mix, float(max(0., (-slack / self.noise).max()))

    def retry_ipm(self, model):
        """Solve the same restricted rows from scratch, discarding the failed basis."""
        lp = model.model.getLp()
        cols = lp.a_matrix_
        matrix = csc_matrix((cols.value_, cols.index_, cols.start_),
                            shape=(len(lp.row_upper_), len(lp.col_cost_)))
        solved = linprog(lp.col_cost_, A_ub=matrix[:-1], b_ub=np.zeros(matrix.shape[0] - 1),
                         A_eq=matrix[-1:], b_eq=[1.], method="highs-ipm",
                         options=dict(threads=1, primal_feasibility_tolerance=1e-10,
                                      dual_feasibility_tolerance=1e-10, small_matrix_value=1e-12))
        if solved.success:
            solved.x = solved.x[sum(model.indices, []) + [0]]
        return solved

    def solve_block(self, a_raw, c_raw, d0, carried, cap):
        factor = self.safe_power / d0
        normalizer = float(np.max(self.x.sig) * factor)
        a, c = a_raw * (factor / normalizer), c_raw * factor
        cmin = float(np.linalg.eigvalsh(c).min())
        if cmin < -1e-10:
            raise ValueError("Synthetic ideal clutter operator has a negative response")
        cmin = max(0., cmin - FLOOR * (1 + np.max(np.abs(c))))
        model = self.new_master(carried)
        trace = []
        for j in range(cap):
            result, widths = model.restricted(a, c)
            retry = bool(not result.success or result.x[-1] <= 0)
            if retry:
                result = self.retry_ipm(model)
            if not result.success or result.x[-1] <= 0:
                raise RuntimeError(f"Teacher LP failed with HiGHS status {result.status}")
            dual = -float(result.eqlin.marginals[0])
            mu = np.maximum(0., -result.ineqlin.marginals[:-1].reshape(self.x.N, self.x.K)) * self.row_scale
            lam = max(0., -float(result.ineqlin.marginals[-1]))
            h = self.h
            base = herm(a - dual * c - (h * mu[:, None, :]) @ h.swapaxes(1, 2).conj())
            value, safe, vectors, fallback = principal_prices(base, h, mu * (1 + 1 / self.gamma))
            excess = value - lam
            eps = float(max(0., safe.max() - lam))
            scale_rc = max(0., lam - dual - float(np.sum(mu * self.noise)))
            upper = dual + eps / (1 + cmin) + scale_rc + FLOOR * (1 + abs(lam) + np.max(mu))
            chosen = excess > 1e-11
            gap_db = float(10 * np.log10(upper / -result.fun))
            lower, mix, qos = None, None, None
            if gap_db <= GATE_DB + 1e-7 or not np.any(chosen):
                raw_weights = np.split(np.maximum(0., result.x[:-1] / result.x[-1]), np.cumsum(widths)[:-1])
                q, weights, mix, qos = self.witness(model, raw_weights)
                clutter = float(np.einsum("nij,nji->", c, q).real)
                lower = float(np.einsum("nij,nji->", a, q).real / (1 + clutter))
                gap_db = float(10 * np.log10(upper / lower))
            trace.append((gap_db, mix, qos, fallback, retry))
            if lower is not None and -1e-7 <= gap_db <= GATE_DB:
                matrices = price_matrices(h, self.gamma, a, c, dual, mu)
                return SimpleNamespace(q=q * self.safe_power, bank=model.bank, owners=model.owners,
                                       powers=weights, matrices=matrices, scale=1 / (1 + clutter),
                                       lower=lower * normalizer, upper=upper * normalizer,
                                       trace=trace, gap_db=gap_db)
            if not np.any(chosen):
                raise RuntimeError(f"Teacher pricing stalled at {gap_db:.6g} dB")
            for n, owner in zip(*np.where(chosen)):
                model.bank[n].append(vectors[n, owner])
                model.owners[n].append((owner + 1) % (self.x.K + 1))
        raise RuntimeError(f"Teacher hit the {cap}-LP cap at {gap_db:.6g} dB")

    def label(self, cap, outer_cap):
        started = perf_counter()
        q = covariance(self.start)
        carried = [[] for _ in self.bank]
        rows = []
        for j in range(outer_cap):
            eta, a, c, d0, u = receiver(self.x, q, dict.fromkeys(METRICS, 0.))
            tx = self.solve_block(a, c, d0, carried, cap)
            gain_bound = float(10 * np.log10(tx.upper / eta))
            rows.append(dict(lp_solves=len(tx.trace), gap_db=tx.gap_db,
                             outer_upper_gain_db=gain_bound, lower_scnr_db=float(10 * np.log10(tx.lower)),
                             max_witness_mix=max(t[1] for t in tx.trace if t[1] is not None),
                             max_decoded_qos_shortfall=max(t[2] for t in tx.trace if t[2] is not None),
                             dense_prices=sum(t[3] for t in tx.trace), ipm_retries=sum(t[4] for t in tx.trace)))
            if j > 0 and -1e-7 <= gain_bound <= GATE_DB:
                labels = self.pack(tx, u)
                meta = dict(status="converged", seconds=perf_counter() - started,
                            outer_steps=j + 1, lp_solves=sum(r["lp_solves"] for r in rows),
                            rows=rows, active_atoms=len(labels["power"]),
                            receiver_sha256=hashlib.sha256(u.tobytes()).hexdigest())
                return labels, meta
            q = tx.q
            carried = [[(v, o) for v, o, p in zip(b[len(initial):], tags[len(initial):], w[len(initial):])
                        if p > 0] for b, tags, w, initial in zip(tx.bank, tx.owners, tx.powers, self.bank)]
        raise RuntimeError(f"Teacher hit the {outer_cap}-outer cap at {gain_bound:.6g} dB")

    def pack(self, tx, u):
        atoms, power, index = [], [], []
        for n, (b, o, w) in enumerate(zip(tx.bank, tx.owners, tx.powers)):
            for j in np.flatnonzero(w > ACTIVE):
                atoms.append(b[j])
                power.append(w[j])
                index.append((n, (o[j] - 1) % (self.x.K + 1)))
        # The teacher loss uses the sum of scaled primal masses per owner.
        mass = np.zeros((self.x.N, self.x.K + 1))
        np.add.at(mass, tuple(np.asarray(index).T), np.asarray(power) * tx.scale)
        basis = proposal_basis(self.x, self.h, self.bank)
        return dict(teacher_u=u, active_vectors=np.asarray(atoms), active_indices=np.asarray(index, np.int16),
                    power=np.asarray(power) * self.safe_power, mass=mass,
                    teacher_projection=project(basis, tx.matrices[:, :self.x.K]))


def run(args):
    os.sched_setaffinity(0, {args.cpu})
    scenes = Scenes(args.tutorial)
    args.mode, args.pilot = "part-e-teacher", args.split == "pilot"
    stamp = manifest(args.tutorial, args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "environment.json").write_text(json.dumps(stamp, indent=2) + "\n")
    identifiers = PILOT if args.pilot else TRAIN
    with (args.output / "teacher.jsonl").open("x") as log:
        for scene_id in identifiers[args.shard::args.shards]:
            x, row = scenes.make(scene_id)
            if scene_id == PILOT.start:
                row["choi_check"] = check_choi(x)
            teacher = Teacher(x)
            failures = []
            for inner, outer in ((80, 30), (160, 60)):
                try:
                    labels, info = teacher.label(inner, outer)
                    row.update(info)
                    with NamedTemporaryFile(dir=args.output, suffix=".npz", delete=False) as tmp:
                        np.savez(tmp, **labels)
                    Path(tmp.name).rename(args.output / f"{scene_id}.npz")
                    break
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as err:
                    failures.append(str(err))
            else:
                row["status"] = "failed"
            row["failures"] = failures
            print(json.dumps(row, allow_nan=False), file=log, flush=True)
            if args.pilot or scene_id % 50 in (0, 1):
                print(scene_id, row["status"], row.get("outer_steps"), row.get("lp_solves"),
                      row.get("seconds"), failures, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tutorial", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("pilot", "train"), required=True)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    run(parser.parse_args())
