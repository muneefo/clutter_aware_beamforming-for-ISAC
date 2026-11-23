"""Check B4 and both B5 execution paths against a two-tone analytic optimum."""
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from parta_ao import JointTransmitSDP, pair_sum
from partb_sdp import CooperSDP, TonePriceSDP


def main():
    h = np.array([1, 1j]) / np.sqrt(2)
    g = np.array([1, -1j]) / np.sqrt(2)
    htones = np.broadcast_to(h[None, :, None], (2, 2, 1))
    x = SimpleNamespace(N=2, K=1, T=2, P=2, H=htones, noise=.1, g=np.full((2, 1), 2.))
    matrices = JointTransmitSDP(x)
    signal = np.array([.75, .50])[:, None, None] * np.outer(g, g.conj())
    clutter = np.broadcast_to(.25 * np.eye(2), signal.shape)
    # Each user needs 0.2 power. The remaining 1.6 goes to the higher-reward tone.
    optimum = .75 * 1.6
    for persistent in (False, True):
        solver = CooperSDP(matrices, persistent=persistent)
        for penalty in (clutter, 2 * clutter):
            sensing, users, info = solver.solve(signal, penalty, 0.)
            assert info["status"] == "solved", info
            total = sensing + users.sum(axis=1)
            response = pair_sum(signal, total) / (1 + pair_sum(penalty, total))
            expected = optimum / (1 + 2 * penalty[0, 0, 0])
            assert abs(response - expected) < 1e-5
    for count in (1, 2):
        solver = TonePriceSDP(matrices, count)
        try:
            sensing, users, info = solver.solve(signal, np.zeros_like(clutter), 0.)
        finally:
            solver.close()
        assert info["status"] == "solved" and info["price_evals"] > 1, info
        total = sensing + users.sum(axis=1)
        assert abs(pair_sum(signal, total) - optimum) < 1e-5
        assert min(matrices.residuals(sensing, users, total).values()) >= -1e-6
    print("The fractional SDP and both power-price execution paths match the analytic optimum.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
