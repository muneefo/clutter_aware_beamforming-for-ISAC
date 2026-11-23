"""Permutation-equivariant proposal features, ablation losses and optimization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from parte_teacher import PILOT, TRAIN, Scenes, project, proposal_basis
from partb_run import METRICS, covariance, master, receiver, seed_bank

import numpy as np


def owner_matrices(h, gamma, mu, base, k):
    hc = h.transpose(0, 2, 1)
    owner_gram = hc[..., :, None] * hc[..., None, :].conj()
    users = base[:, None] + (mu * (1 + 1 / gamma))[..., None, None] * owner_gram
    return np.concatenate((users, base[:, None]), axis=1)


def normalized(a):
    scale = np.maximum(np.max(np.abs(a), axis=(-2, -1)), 1e-12)
    return a / scale[..., None, None], scale


def deployment_features(x, h, noise, basis, matrices, price):
    local, scale = normalized(project(basis, matrices[:, :x.K]))
    gram = basis.swapaxes(-1, -2).conj() @ basis
    cost = local.real
    diagonal = np.diagonal(cost, axis1=-2, axis2=-1)
    overlaps = np.abs(np.einsum("ntk,nktj->nkj", h.conj(), basis)) ** 2
    qos = np.stack((np.log10(x.g), np.log10(noise) / 20), axis=-1)
    pooled = np.stack((diagonal[..., 0].mean(axis=1),
                       overlaps[..., 0].mean(axis=1)), axis=-1)
    pooled = np.broadcast_to(pooled[:, None], (*noise.shape, 2))
    f = np.concatenate((diagonal, overlaps, qos, pooled), axis=-1)
    return f, gram.real, cost, price / scale


def initial_features(x):
    h, noise, bank, owners, w = seed_bank(x)
    _, a, c, d0, u = receiver(x, covariance(w), dict.fromkeys(METRICS, 0.))
    _, base, mu, price, _, gamma = master(x, h, noise, bank, owners, a, c, d0,
                                         profile=dict.fromkeys(METRICS, 0.))
    basis = proposal_basis(x, h, bank)
    return deployment_features(x, h, noise, basis, owner_matrices(h, gamma, mu, base, x.K), price), basis, u


def scene_training(x, path):
    (features, gram, reduced_cost, price), basis, u = initial_features(x)
    with np.load(path) as label:
        if "teacher_projection" in label:
            teacher = label["teacher_projection"]
        else:  # The first pilot saved full matrices.
            teacher = project(basis, label["teacher_matrices"][:, :x.K])
        teacher, _ = normalized(teacher)
        mass = label["mass"][:, :x.K]
        mass = mass / max(np.finfo(float).tiny, mass.sum())
        correlation = np.abs(np.vdot(u, label["teacher_u"])) / (
            np.linalg.norm(u) * np.linalg.norm(label["teacher_u"]))
        delta = np.rad2deg(np.arccos(np.clip(correlation, 0, 1)))
    return (features.reshape(-1, 12), gram.reshape(-1, 4, 4),
            reduced_cost.reshape(-1, 4, 4), price.ravel(),
            teacher.real.reshape(-1, 4, 4), mass.ravel(), np.full(price.size, 1 / price.size)), float(delta)


def initialization():
    rng = np.random.default_rng(71003)
    return [rng.normal(0, .15, (12, 8)), np.zeros(8),
            rng.normal(0, .15, (8, 4)), np.zeros(4)]


def forward(weights, f):
    w1, b1, w2, b2 = weights
    hidden = np.tanh(f @ w1 + b1)
    output = np.tanh(hidden @ w2 + b2)
    return hidden, output, output + np.array([0., 0., 1.25, 0.])


def predict(weights, f, basis):
    _, _, coefficients = forward(weights, f)
    v = (basis @ coefficients[..., None])[..., 0]
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Proposal is numerically zero in the four-direction span")
    return v / norm


def rayleigh(coefficients, gram, matrix, sqnorm):
    mc = np.einsum("...ij,...j->...i", matrix, coefficients)
    value = np.einsum("...i,...i->...", coefficients, mc) / sqnorm
    grad = 2 * (mc - value[..., None] * gram) / sqnorm[..., None]
    return value, grad


def loss(weights, prepared, envelope):
    f, gram, costs, price, teacher, mass, per_scene = prepared
    hidden, output, coefficients = forward(weights, f)
    gram_c = np.einsum("...ij,...j->...i", gram, coefficients)
    sqnorm = np.einsum("...i,...i->...", coefficients, gram_c)
    if sqnorm.min() <= 1e-20:
        raise ValueError("Singular normalized proposal in training")
    raw, grad = rayleigh(coefficients, gram_c, costs, sqnorm)
    teach, teach_grad = rayleigh(coefficients, gram_c, teacher, sqnorm)
    n = per_scene.sum()
    rc_loss = -float(np.dot(per_scene, raw - price)) / n
    envelope_loss = -float(np.dot(mass, teach)) / n
    g = -(per_scene[:, None] * grad + envelope * mass[:, None] * teach_grad) / n
    delta = g * (1 - output**2)
    dh = (delta @ weights[2].T) * (1 - hidden**2)
    grad = [f.T @ dh, dh.sum(axis=0), hidden.T @ delta, delta.sum(axis=0)]
    return rc_loss + envelope * envelope_loss, grad, (rc_loss, envelope_loss)


def checks(scenes, pilot):
    results = []
    for scene_id in list(PILOT)[:3]:
        x, _ = scenes.make(scene_id)
        data, angle = scene_training(x, pilot / f"{scene_id}.npz")
        w = initialization()
        for envelope in (0., 1.):
            actual, grads, _ = loss(w, data, envelope)
            maximum = 0.
            for block, grad in zip(w, grads):
                for index in np.ndindex(block.shape):
                    old = block[index]
                    step = 1e-5
                    block[index] = old + step
                    upper = loss(w, data, envelope)[0]
                    block[index] = old - step
                    lower = loss(w, data, envelope)[0]
                    block[index] = old
                    fd = (upper - lower) / (2 * step)
                    maximum = max(maximum, abs(grad[index] - fd) / max(1., abs(fd), abs(grad[index])))
            if maximum > 1e-6:
                raise ValueError(f"Finite-difference check failed at scene {scene_id}: {maximum}")
            results.append(dict(scene_id=scene_id, envelope=envelope, max_scaled_gradient_error=maximum,
                                loss=actual, teacher_deployment_u_angle_deg=angle))
        normal, b, _ = initial_features(x)
        permutation = np.random.default_rng(scene_id).permutation(x.K)
        x.H = x.H[..., permutation]
        x.g = x.g[..., permutation]
        permuted, pb, _ = initial_features(x)
        pv = predict(w, permuted[0], pb)
        v = predict(w, normal[0], b)[:, permutation]
        cosine = np.abs(np.sum(pv * v.conj(), axis=-1))
        delta = max(np.max(np.abs(permuted[0] - normal[0][:, permutation])),
                    np.max(np.abs(1 - cosine)))
        if delta > 1e-6:
            raise ValueError(f"User permutation changes features or proposals by {delta:.3g}")
        results[-1]["permutation_max_error"] = float(delta)
    return results


def prepared_dataset(scenes, training, output, pilot=False):
    roots = [training] if pilot else sorted(training.glob("parte-train-*"))
    paths = {int(p.stem): p for r in roots for p in r.glob("*.npz")}
    expected = set(PILOT if pilot else TRAIN)
    if set(paths.keys()) != expected:
        missing = sorted(expected - paths.keys())
        raise ValueError(f"Teacher set has {len(paths)} scenes. Missing {len(missing)}; first IDs are {missing[:8]}")
    all_rows = []
    examples = []
    for scene_id, p in sorted(paths.items()):
        x, _ = scenes.make(scene_id)
        data, angle = scene_training(x, p)
        examples.append(data)
        all_rows.append(dict(scene_id=scene_id, teacher_deployment_u_angle_deg=angle,
                             label_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    offsets = np.cumsum([0] + [r[0].shape[0] for r in examples])
    keys = ("features", "gram", "costs", "price", "teacher", "mass", "per_scene")
    compact = {k: np.concatenate([r[j] for r in examples], axis=0)
               for j, k in enumerate(keys)}
    np.savez(output / "training-data.npz", offsets=offsets, **compact)
    (output / "training-labels.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in all_rows))


def train_arrays(dataset, envelope, epochs=5):
    keys = ("features", "gram", "costs", "price", "teacher", "mass", "per_scene")
    with np.load(dataset) as prepared:
        offsets = prepared["offsets"]
        arrays = [prepared[k] for k in keys]
    w = initialization()
    rng = np.random.default_rng(71004)
    m, v = [[np.zeros_like(a) for a in w] for _ in range(2)]
    traces = []
    step = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(offsets) - 1)
        moments = np.zeros(3)
        for start in range(0, len(order), 32):
            scenes = order[start:start+32]
            selection = np.concatenate([np.arange(offsets[j], offsets[j+1]) for j in scenes])
            val, grad, separate = loss(w, [a[selection] for a in arrays], envelope)
            moments += len(scenes) * np.array([val, *separate])
            step += 1
            for j, g in enumerate(grad):
                m[j] = .9 * m[j] + .1 * g
                v[j] = .999 * v[j] + .001 * g**2
                w[j] -= .003 * (m[j] / (1 - .9**step)) / (np.sqrt(v[j] / (1 - .999**step)) + 1e-8)
        val, rc, env = (moments / len(order)).tolist()
        traces.append(dict(epoch=epoch, loss=val, reduced_cost_loss=rc, envelope_loss=env,
                           scenes=len(order), steps=step))
    return w, traces


def run(args):
    scenes = Scenes(args.tutorial)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.pilot:
        check = checks(scenes, args.teacher)
        (args.output / "gradient-checks.json").write_text(json.dumps(check, indent=2, allow_nan=False) + "\n")
    prepared_dataset(scenes, args.teacher, args.output, args.pilot)
    for arm, envelope in (("reduced-cost", 0.), ("envelope", 1.)):
        weights, trace = train_arrays(args.output / "training-data.npz", envelope)
        np.savez(args.output / f"{arm}.npz", **{f"w{j}": a for j, a in enumerate(weights)})
        (args.output / f"{arm}.json").write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tutorial", type=Path)
    p.add_argument("teacher", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--pilot", action="store_true")
    run(p.parse_args())
