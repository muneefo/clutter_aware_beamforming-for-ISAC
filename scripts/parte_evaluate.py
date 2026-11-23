"""Run matched Part E proposal ablations and audit the archived Draw 112 record."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from time import perf_counter
from pathlib import Path

from parte_teacher import ACTIVE, HELD_OUT, ORIGINAL, PILOT, Scenes, herm, proposal_basis, rng
from parte_policy import deployment_features, owner_matrices, predict
from partb_run import METRICS, covariance, master, pair_sum, receiver, recover, score, seed_bank
from partc_run import manifest
from verify_fast_beams import audit, scene

import numpy as np
from scipy.io import loadmat

ARMS = ("physics", "random", "gradient", "reduced-cost", "envelope")
BUDGETS = ((1, 1), (2, 1), (3, 2))


def unit(v):
    lengths = np.linalg.norm(v, axis=-1, keepdims=True)
    if lengths.min() <= np.finfo(float).tiny:
        raise ValueError("A proposal has zero norm")
    return v / lengths


def gradient_proposals(x, w, a, c, u, eta, depth, spsa=None, iteration=1):
    """Original normalized target, clutter and log-SINR proposal."""
    normals = x.H.swapaxes(1, 2).conj()
    g = normals @ w
    k = np.arange(x.K)
    desired = g[:, k, k].copy()
    g[:, k, k] = 0
    interference = np.sum(np.abs(g)**2, axis=-1) + x.noise
    ratios = np.abs(desired)**2 / (interference * x.g)
    qos = -x.H.transpose(0, 2, 1)[..., None] * (g / interference[..., None])[:, :, None]
    tiny = np.finfo(float).tiny
    qos[:, k, :, k] = (x.H.transpose(0, 2, 1) * (
        desired / np.maximum(np.abs(desired)**2, tiny))[..., None]
    ).swapaxes(0, 1)
    qos /= np.maximum(np.linalg.norm(qos, axis=(-2, -1), keepdims=True), tiny)
    qos = (qos / np.maximum(1., ratios)[..., None, None]**2).mean(axis=1)
    if spsa is None:
        logits = np.zeros((x.N, 4))
        step = .6
    else:
        raw, step = spsa
        q = covariance(w)
        hot = np.einsum("ni,nij,nj->n", u.conj(), x.Reta, u).real
        cold = np.einsum("nij,nji->n", q, c).real
        f = np.column_stack((cold / np.maximum(hot + cold, tiny),
                             np.tanh(np.log(np.maximum(ratios.min(axis=1), tiny))),
                             np.tanh(np.log(np.maximum(np.sum(abs(w)**2, axis=(1, 2))*x.N/x.P, tiny))),
                             np.full(x.N, iteration/depth)))
        hidden = np.tanh(f @ raw[:24].reshape((6, 4), order="F").T + raw[24:30])
        logits = np.tanh(hidden @ raw[30:54].reshape((4, 6), order="F").T + raw[54:])
    wgt = np.exp(logits)[..., None]
    target, clutter = a @ w, eta * (c @ w)
    denom = np.maximum(np.linalg.norm(target, axis=(-2, -1)) + np.linalg.norm(clutter, axis=(-2, -1)), tiny)
    direction = (wgt[:, 0, :, None] * target - wgt[:, 1, :, None] * clutter) / denom[:, None, None]
    direction += .15 * (1 + logits[:, 2, None, None]) * qos
    proposed = w + np.sqrt(x.P / x.N) * step * wgt[:, 3, :, None] * direction
    return unit(proposed[:, :, :x.K].transpose(0, 2, 1))


def eigprice(h, gamma, mu, base):
    matrices = herm(owner_matrices(h, gamma, mu, base, h.shape[-1]))
    eigenvalues, vectors = np.linalg.eigh(matrices)
    return eigenvalues[..., -1], vectors[..., -1], matrices


def design(x, scene_id, outer, inner, method, model=None):
    started = perf_counter()
    profile = dict.fromkeys(METRICS, 0.)
    h, noise, bank, owners, w = seed_bank(x)
    q = covariance(w)
    eta, a, c, d0, u = receiver(x, q, profile)
    proposal_counts = 0
    added = 0
    inference = 0.
    accepts = 0
    activated = np.zeros((x.N, x.K), dtype=bool)
    terminal_mass = 0.
    threshold = x.P * (1 - 1e-8) * ACTIVE
    first_columns = len(owners[0])
    for j in range(outer):
        best = eta
        for k in range(inner):
            power, base, mu, price, psi, gamma = master(x, h, noise, bank, owners, a, c, d0, profile)
            if added:
                p = np.stack([pn[first_columns:first_columns + x.K] for pn in power])
                activated |= p > threshold
                terminal_mass = float(p.sum())
            recovering = perf_counter()
            y = recover(x, h, bank, owners, power)
            margin, watts = audit(x, y)
            yq = covariance(y)
            objective = pair_sum(a, yq) / (d0 + pair_sum(c, yq))
            if margin >= -1e-4 and watts <= x.P * (1 + 1e-7) and objective >= best * (1 - 1e-10):
                best, w, q = objective, y, yq
                accepts += 1
            profile["recovery"] += perf_counter() - recovering
            terminal = j == outer - 1 and k == inner - 1
            first = j == 0 and k == 0
            price_started = perf_counter()
            if not first or terminal or method == "physics":
                eigen, directions, matrices = eigprice(h, gamma, mu, base)
                profile["price"] += perf_counter() - price_started
            else:
                matrices = herm(owner_matrices(h, gamma, mu, base, x.K))
                sensor = np.linalg.eigh(matrices[:, -1])[1][..., -1]
                profile["price"] += perf_counter() - price_started
            if terminal:
                bound_db = float(10 * np.log10(1 + max(0., float(eigen.max() - price)) / psi))
                continue
            if first:
                proposal_counts = x.N * x.K
                start_proposal = perf_counter()
                if method == "random":
                    r = rng(71005, scene_id)
                    shape = (x.N, x.K, x.T)
                    v = unit(r.standard_normal(shape) + 1j * r.standard_normal(shape))
                elif method in ("gradient", "spsa"):
                    v = gradient_proposals(x, w, a, c, u, best, outer,
                                           model if method == "spsa" else None)
                elif method != "physics":
                    basis = proposal_basis(x, h, bank)
                    features = deployment_features(x, h, noise, basis, matrices, price)[0]
                    v = predict(model, features, basis)
                    inference = perf_counter() - start_proposal
                if method != "physics":
                    directions = np.concatenate((v, sensor[:, None]), axis=1)
            bank = [np.column_stack((a_bank, new.T)) for a_bank, new in zip(bank, directions)]
            owners = [np.r_[label, np.arange(x.K + 1)] for label in owners]
            added += x.N * (x.K + 1)
        eta, a, c, d0, u = receiver(x, q, profile)
    seconds = perf_counter() - started
    design_row = dict(status="solved", seconds=seconds, inference_seconds=inference,
                      lp_solves=outer*inner, pricing_passes=outer*inner, accepted=accepts,
                      atoms=sum(map(len, owners)), atoms_added=added, first_user_proposals=proposal_counts,
                      first_user_proposals_ever_active=int(activated.sum()),
                      first_user_proposals_terminal_watts=terminal_mass, terminal_bound_db=bound_db)
    score(x, q, w, u, design_row)
    return design_row


def archived_diagnostic(x, checkpoint):
    """Replay the archived policy insertion schedule and log matched pricing."""
    h, noise, bank, owners, w = seed_bank(x)
    q = covariance(w)
    eta, a, c, d0, u = receiver(x, q, dict.fromkeys(METRICS, 0.))
    rows = []
    for outer in range(1, 4):
        entry_score = best = eta
        insert = gradient_proposals(x, w, a, c, u, eta, 3, checkpoint, outer)
        base_index = [len(o) for o in owners]
        bank = [np.column_stack((b, v.T)) for b, v in zip(bank, insert)]
        owners = [np.r_[o, np.arange(x.K)] for o in owners]
        for inner in range(1, 3):
            powers, base, mu, price, psi, gamma = master(
                x, h, noise, bank, owners, a, c, d0, dict.fromkeys(METRICS, 0.))
            eigenvalues, vectors, _ = eigprice(h, gamma, mu, base)
            for n in range(x.N):
                for owner in range(x.K):
                    atom = insert[n, owner]
                    p = float(powers[n][base_index[n] + owner])
                    p_eig = vectors[n, owner]
                    rows.append(dict(outer=outer, inner=inner, tone=n, owner=owner, watts=p,
                                     active=bool(p > ACTIVE * x.P * (1 - 1e-8)),
                                     cosine=float(np.clip(abs(np.vdot(atom, p_eig)), 0, 1)),
                                     priceable=bool(eigenvalues[n, owner] - price > 1e-7)))
            y = recover(x, h, bank, owners, powers)
            yq = covariance(y)
            margin, watts = audit(x, y)
            candidate = pair_sum(a, yq) / (d0 + pair_sum(c, yq))
            if margin >= -1e-4 and watts <= x.P * (1 + 1e-7) and candidate >= best * (1 - 1e-10):
                w, q, best = y, yq, candidate
            add = eigenvalues - price > 1e-7
            bank = [np.column_stack((b, v[mask].T)) for b, v, mask in zip(bank, vectors, add)]
            owners = [np.r_[o, np.flatnonzero(mask)] for o, mask in zip(owners, add)]
            if 10 * np.log10(1 + max(0., float(eigenvalues.max() - price)) / psi) <= .002:
                break
        eta, a, c, d0, u = receiver(x, q, dict.fromkeys(METRICS, 0.))
        if 10 * np.log10(eta / entry_score) < .002 and 10 * np.log10(
                1 + max(0., float(eigenvalues.max() - price)) / psi) <= .002:
            break
    return rows


def spsa_checkpoint(path):
    ckpt = loadmat(path, simplify_cells=True)
    model = ckpt.get("model")
    if model is None and "outWU" in ckpt:
        model = ckpt["outWU"]["model"]
    if model is None or np.asarray(model["W"]).size != 58:
        raise ValueError("Expected the original 58-parameter SPSA transmit model")
    w = np.asarray(model["W"], dtype=float).reshape(58)
    step = float(model["fixed"])
    if not np.all(np.isfinite(w)) or not np.isfinite(step) or step <= 0:
        raise ValueError("Original checkpoint must contain finite weights and a positive step")
    return w, step


def archive(args, scenes, checkpoint):
    path = Path("first-isac-paper/three-method-sinr/pilot_scnr.csv")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=rows)
    if checkpoint is None:
        out["status"] = "original_checkpoint_missing"
    else:
        out.update(status="measured", checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
        with (args.output / "spsa-atoms.jsonl").open("x") as f:
            for original in rows:
                draw = int(original["scene"])
                x = scenes.baseline
                scene(x, draw)
                for row in archived_diagnostic(x, checkpoint):
                    print(json.dumps(dict(scene=draw, **row), allow_nan=False), file=f)
    (args.output / "archive.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


def run(args):
    os.sched_setaffinity(0, {args.cpu})
    args.mode = "part-e-evaluate"
    scenes = Scenes(args.tutorial)
    stamp = manifest(args.tutorial, args)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = spsa_checkpoint(args.checkpoint) if args.checkpoint else None
    archive(args, scenes, checkpoint)
    weights = {}
    for name in ("reduced-cost", "envelope"):
        file = args.weights / f"{name}.npz"
        with np.load(file) as weights_file:
            weights[name] = [weights_file[f"w{j}"] for j in range(4)]
        stamp[name + "_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    if checkpoint is not None:
        weights["spsa"] = checkpoint
    arms = (*ARMS, "spsa") if checkpoint is not None else ARMS
    path = args.output / "environment.json"
    path.write_text(json.dumps(stamp, indent=2, allow_nan=False) + "\n")
    ids = [PILOT.start, HELD_OUT.start, ORIGINAL.start] if args.pilot else [*HELD_OUT, *ORIGINAL]
    with (args.output / "designs.jsonl").open("x") as f:
        for scene_id in ids[args.shard::args.shards]:
            x, key = scenes.make(scene_id)
            cases = [(outer, inner, method) for outer, inner in BUDGETS for method in arms]
            for index in rng(71006, scene_id).permutation(len(cases)):
                outer, inner, method = cases[index]
                try:
                    row = design(x, scene_id, outer, inner, method, weights.get(method))
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as err:
                    row = dict(status="failed", reason=str(err))
                print(json.dumps(dict(**key, outer=outer, inner=inner, arm=method, **row), allow_nan=False),
                      file=f, flush=True)
            if args.pilot or scene_id % 8 in (0, 1):
                print(scene_id, key["map_id"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tutorial", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    run(parser.parse_args())
