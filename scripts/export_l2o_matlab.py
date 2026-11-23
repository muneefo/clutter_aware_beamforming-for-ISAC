"""Export the two frozen Part E policies and a deterministic parity fixture."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

from parte_policy import deployment_features, predict
import numpy as np
from scipy.io import savemat


def export(checkpoints, destination):
    rng = np.random.default_rng(531241)
    nt, k = 16, 4
    b = rng.normal(size=(1, k, nt, 4)) + 1j * rng.normal(size=(1, k, nt, 4))
    b /= np.linalg.norm(b, axis=-2, keepdims=True)
    h = b[..., 1].transpose(0, 2, 1)
    a = rng.normal(size=(1, k, nt, nt)) + 1j * rng.normal(size=(1, k, nt, nt))
    m = (a + a.swapaxes(-1, -2).conj()) / 2
    gamma = 10 ** .8
    noise = 10 ** rng.uniform(-9, -6, size=(1, k))
    features = deployment_features(SimpleNamespace(K=k, g=np.full((1, k), gamma)),
                                   h, noise, b, m, .17)[0]
    fixture = dict(basis=b[0].transpose(1, 2, 0), matrices=m[0].transpose(1, 2, 0),
                   channels=h[0], gamma=gamma, noiseRatio=noise[0], features=features[0])
    bundle = {"fixture": fixture}
    for key, source in (("reducedCost", "reduced-cost"), ("envelope", "envelope")):
        path = checkpoints / f"{source}.npz"
        with np.load(path) as checkpoint:
            weights = [checkpoint[f"w{i}"] for i in range(4)]
        bundle[key] = {f"w{i}": a for i, a in enumerate(weights)}
        bundle[key]["sourceSHA256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        fixture[key] = predict(weights, features, b)[0].T
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as out:
        savemat(out, bundle, do_compression=True, oned_as="row")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    export(args.checkpoints, args.destination)
