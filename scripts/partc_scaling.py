"""Array-size control using the archived scatterers as a factorized operator."""
import copy
import signal
from contextlib import contextmanager

import numpy as np
from scipy.io import loadmat

from partb_run import MatlabTwister
from partc_run import attempt, output


class ScattererKernel:
    """Sum of nonnegative rank-one covariance responses at each tone."""

    def __init__(self, vectors, powers):
        self.vectors = vectors
        self.powers = powers

    def apply(self, covariance, back=False):
        # Tx and Rx use the same unit-norm ULA manifold in this scaling control.
        v = self.vectors
        vv = v.swapaxes(1, 2).conj()
        gains = np.sum(v.conj() * (covariance @ v), axis=1).real
        mapped = (v * (self.powers * gains)[:, None, :]) @ vv
        return (mapped + mapped.swapaxes(1, 2).conj()) / 2


def rooms(base, file):
    data = loadmat(file, variable_names=["para", "theta_cc_all", "sigma2_cc_all"], simplify_cells=True)
    freq = 1. + np.arange(base.N) * data["para"]["deltaf"] / data["para"]["fc"]
    angles = np.sin(np.deg2rad(data["theta_cc_all"]))
    powers = data["sigma2_cc_all"].T
    level = np.trace(base.Reta, axis1=1, axis2=2).real / base.R
    for m in (8, 16, 32, 64):
        x = copy.copy(base)
        x.T = x.R = x.S = m
        x.K = 4
        x.g = np.full((x.N, x.K), 10 ** 0.8)
        steering = np.exp(-1j * np.pi * freq[:, None, None] *
                          np.arange(m)[None, :, None] * angles[None, None, :]) / np.sqrt(m)
        target = np.exp(-1j * np.pi * freq[:, None] *
                        np.arange(m)[None] * np.sin(np.deg2rad(-10))) / np.sqrt(m)
        x.a = x.b = target
        x.est = x.truth = ScattererKernel(steering, powers)
        x.Reta = level[:, None, None] * np.eye(m)
        yield m, x


def channels(x, seed):
    rng = MatlabTwister(seed)
    big = rng.complex((64, x.N, x.K)).transpose(1, 0, 2)
    distances = np.array([50 + int(351 * rng.uniform()) for _ in range(x.K)])
    x.H = np.ascontiguousarray(big[:, :x.T] * 10 ** ((-30 - 27 * np.log10(distances)) / 20))
    return distances


@contextmanager
def deadline(seconds):
    def expire(_signum, _frame):
        raise TimeoutError(f"Exceeded {seconds} s scaling wall-time cap")
    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def scale_sweep(base, args, stream):
    path = args.tutorial / "data" / "covMat_BLP.mat"
    for m, x in rooms(base, path):
        blocked = set()
        for seed in ([22099] if args.pilot else [22099, *range(22101, 22133)]):
            distances = channels(x, seed)
            methods = ("zf", "tone", "lp")
            for method in methods:
                if method in blocked:
                    continue
                with deadline(30):
                    row, trace = attempt(x, seed, method, 1, 1)
                output(stream, row, trace, method=method, seed=seed, array=m, users=4, gamma=8,
                       outer=1, inner=1, warmup=seed == 22099, distances=distances.tolist())
                if seed == 22099 and method == "zf" and row["status"] in ("timeout", "cut_limit"):
                    blocked.add(method)
            print(args.mode, m, seed, flush=True)
