"""Joint AO–SDP replay of the pinned tutorial, with native SCS PSD cones."""
import time

import numpy as np
import scs
from scipy import sparse

from parta_barrier import BarrierSDP
from parta_eigencut import EigenCutSDP
from parta_incremental import IncrementalCutSDP
from verify_fast_beams import MatlabTwister, audit, fixed_receiver, true_score


def covariance(w):
    return w @ w.swapaxes(-1, -2).conj()


def pair_sum(a, b):
    return np.einsum("nij,nji->", a, b).real


class ConeCoordinates:
    """SCS lower-triangular, inner-product-preserving PSD coordinates."""

    def __init__(self, dim, complex_cone):
        coordinates = []
        for col in range(dim):
            for row in range(col, dim):
                coordinates.append((row, col, False))
                if complex_cone and row != col:
                    coordinates.append((row, col, True))
        self.dim = dim
        self.row, self.col, self.imag = np.array(coordinates).T
        self.imag = self.imag.astype(bool)
        self.real = ~self.imag
        self.weight = np.where(self.row == self.col, 1.0, np.sqrt(2))

    def pack(self, matrix):
        entries = matrix[..., self.row, self.col]
        return np.where(self.imag, entries.imag, entries.real) * self.weight

    def unpack(self, vector):
        values = vector / self.weight
        result = np.zeros((*values.shape[:-1], self.dim, self.dim), complex)
        result[..., self.row[self.real], self.col[self.real]] = values[..., self.real]
        result[..., self.row[self.imag], self.col[self.imag]] += 1j * values[..., self.imag]
        result[..., self.col, self.row] = result[..., self.row, self.col].conj()
        return result


class JointTransmitSDP:
    """Conic dual with global power price and per-tone user prices."""

    def __init__(self, x):
        self.n, self.k, self.t = x.N, x.K, x.T
        self.real = ConeCoordinates(x.T, False)
        self.complex = ConeCoordinates(x.T, True)
        self.power_unit = x.P / x.N
        lengths = np.linalg.norm(x.H, axis=1)
        normalized = (x.H / lengths[:, None, :]).transpose(0, 2, 1)
        channel_grams = normalized[..., None] * normalized[..., None, :].conj()
        noise = x.noise / lengths**2 / self.power_unit
        self.normalized = normalized
        self.grams = channel_grams
        self.gamma = x.g
        self.noise = noise
        self.cost = np.r_[x.N, -noise.ravel()]
        coeff = np.broadcast_to(-np.ones((x.N, x.K, x.K)), (x.N, x.K, x.K)).copy()
        idx = np.arange(x.K)
        coeff[:, idx, idx] = 1 / x.g
        ranges = np.arange(x.N * x.K).reshape(x.N, x.K) + 1
        matrices = [-sparse.eye(1 + x.N * x.K, format="csc")]
        for coords, gram, count in [
            (self.real, -self.real.pack(channel_grams), 1),
            (self.complex, self.complex.pack(channel_grams)[:, None] * coeff[..., None], x.K),
        ]:
            size = coords.weight.size
            coeff_by_block = gram.reshape(x.N, count, x.K, size).transpose(0, 1, 3, 2)
            row = np.arange(x.N * count * size).reshape(x.N, count, size, 1)
            col = ranges[:, None, None, :]
            graph = sparse.coo_matrix(
                (coeff_by_block.ravel(), (np.broadcast_to(row, coeff_by_block.shape).ravel(),
                                            np.broadcast_to(col, coeff_by_block.shape).ravel())),
                shape=(x.N * count * size, self.cost.size),
            ).tocsc()
            price = np.tile(-coords.pack(np.eye(x.T)), x.N * count)
            graph = graph + sparse.csc_matrix((price, (row.ravel(), np.zeros(price.size, dtype=int))), shape=graph.shape)
            matrices.append(graph)
        self.matrix = sparse.vstack(matrices, format="csc")
        self.cone = {"l": self.cost.size, "s": [x.T] * x.N, "cs": [x.T] * (x.N * x.K)}
        self.model = None
        self.built_seconds = 0.0

    def solve(self, reward, penalty, eta):
        a = reward - eta * penalty
        scale = max(np.max(np.abs(a)), np.finfo(float).tiny)
        b = -np.r_[np.zeros(self.cost.size), self.real.pack(a).ravel(),
                   np.repeat(self.complex.pack(a), self.k, axis=0).ravel()] / scale
        if self.model is None:
            started = time.perf_counter()
            self.model = scs.SCS({"A": self.matrix, "b": b, "c": self.cost}, self.cone,
                                 eps_abs=1e-6, eps_rel=1e-6, max_iters=12000, verbose=False)
            self.built_seconds = time.perf_counter() - started
            sol = self.model.solve(warm_start=False)
        else:
            self.model.update(b=b)
            sol = self.model.solve(warm_start=True)
        info = sol["info"]
        if info["status_val"] != 1:
            return None, None, info
        real_end = self.cost.size + self.n * self.real.weight.size
        sensing = self.real.unpack(sol["y"][self.cost.size:real_end].reshape(self.n, -1))
        users = self.complex.unpack(sol["y"][real_end:].reshape(self.n, self.k, -1))
        return sensing, users, info

    def residuals(self, sensing, users, q):
        total = np.einsum("nkij,nji->nk", self.grams, q).real
        desired = np.einsum("nkij,nkji->nk", self.grams, users).real
        slack = (1 + 1 / self.gamma) * desired - total - self.noise
        return {
            "normalized_qos_slack": float(np.min(slack)),
            "normalized_power_slack": float(self.n - np.trace(q, axis1=-2, axis2=-1).real.sum()),
            "sensing_min_eigenvalue": float(np.min(np.linalg.eigvalsh(sensing))),
            "user_min_eigenvalue": float(np.min(np.linalg.eigvalsh(users))),
        }


def recover_streams(x, q, users):
    channels = x.H.transpose(0, 2, 1)
    responses = np.einsum("nkij,nkj->nki", users, channels)
    denom = np.einsum("nki,nki->nk", channels.conj(), responses).real
    # The tutorial replaces nonpositive denominators with 1e-12.
    w = (responses / np.sqrt(np.where(denom > 0, denom, 1e-12))[..., None]).transpose(0, 2, 1)
    residual = q - covariance(w)
    residual = (residual + residual.swapaxes(-1, -2).conj()) / 2
    eig, vec = np.linalg.eigh(residual)
    positive = np.maximum(eig, 0)
    rank = (eig > 1e-7 * positive[:, -1:]).sum(axis=1)
    full = np.concatenate([w, vec * np.sqrt(positive)[:, None, :]], axis=-1)
    discard = x.T - (x.S - x.K)
    truncated = np.concatenate([w, vec[:, :, discard:] * np.sqrt(positive[:, None, discard:])], axis=-1)
    return full, truncated, rank, float(positive[:, :discard].sum())


def beam_seed(channel_seed):
    return (17389 + 1664525 * channel_seed) % (2**32)


def run_ao(x, start_seed, outer=30, inner=30, outer_tol=0.01, inner_tol=0.001, backend="scs"):
    started = time.perf_counter()
    raw = MatlabTwister(start_seed).complex((x.T, x.S, x.N)).transpose(2, 0, 1)
    q = covariance(raw * np.sqrt(x.P) / np.linalg.norm(raw))
    old, _, _, _, _ = fixed_receiver(x, q, x.est)
    eta = old
    problem = JointTransmitSDP(x)
    solver = {"scs": lambda p: p, "barrier": BarrierSDP, "eigencut": EigenCutSDP,
              "incremental": IncrementalCutSDP}[backend](problem)
    trace = []
    for j in range(1, outer + 1):
        _, a, c, hot, u = fixed_receiver(x, q, x.est)
        a = a * problem.power_unit / hot
        c = c * problem.power_unit / hot
        for k in range(1, inner + 1):
            sensing, users, info = solver.solve(a, c, eta)
            info = {name: info[name] for name in (
                "iter", "status", "pobj", "dobj", "res_pri", "res_dual", "gap",
                "setup_time", "solve_time",
            )}
            info.update(outer=j, inner=k)
            trace.append(info)
            if users is None:
                return {"status": info["status"], "seconds": time.perf_counter() - started}, trace
            q = sensing + users.sum(axis=1)
            updated = pair_sum(a, q) / (1 + pair_sum(c, q))
            info.update(eta=updated)
            rel = abs(updated - eta) / max(abs(eta), 1e-30)
            eta = updated
            if rel <= inner_tol:
                break
        q *= problem.power_unit
        if abs(10 * np.log10(eta / old)) < outer_tol:
            break
        old = eta
    full, w, rank, discarded = recover_streams(x, q, users * problem.power_unit)
    margin, power = audit(x, w)
    # Audit the raw blocks at the solver's scaling, before recovery or clipping.
    metrics = problem.residuals(sensing, users, q / problem.power_unit)
    metrics.update(
        status="solved", seconds=time.perf_counter() - started, outer_steps=j, sdp_solves=len(trace),
        max_residual_rank=int(rank.max()), rank_excess_tones=int(np.count_nonzero(rank > x.S - x.K)),
        omitted_power_W=discarded, min_sinr_margin_db=margin, power_W=power,
        physically_feasible=bool(margin >= -1e-4 and power <= x.P * (1 + 1e-7)),
    )
    for name, power_matrix in (("cov", q), ("full", covariance(full)), ("streams", covariance(w))):
        metrics[f"design_{name}_db"] = true_score(x, power_matrix, u, x.est)
        metrics[f"true_{name}_db"] = true_score(x, power_matrix, u, x.truth)
    return metrics, trace
