"""Solve the joint SDP via primal eigenvector cuts and a full dual bound."""
import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


class EigenCutSDP:
    """Maintain PSD eigenvector atoms, with an independently feasible dual price."""

    def __init__(self, problem):
        self.problem = problem
        self.n, self.k, self.t = problem.n, problem.k, problem.t
        self.channels = problem.normalized
        directions = np.linalg.pinv(self.channels.conj()).transpose(0, 2, 1)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        self.bank = [list(directions[tone]) for tone in range(self.n)]
        self.owners = [list(range(1, self.k + 1)) for _ in range(self.n)]
        self.signs = -np.ones((self.n, self.k + 1, self.k))
        owners = np.arange(self.k)
        self.signs[:, owners + 1, owners] = 1 / problem.gamma

    def restricted(self, matrix):
        widths = [len(bank) for bank in self.bank]
        gains, rewards = [], []
        for n, (bank, owners) in enumerate(zip(self.bank, self.owners)):
            directions = np.column_stack(bank)
            access = np.abs(self.channels[n].conj() @ directions) ** 2
            gains.append(access * self.signs[n, owners].T)
            rewards.append(np.sum((directions.conj() * (matrix[n] @ directions)).real, axis=0))
        reward = np.concatenate(rewards)
        constraints = sparse.vstack([
            -sparse.block_diag(tuple(gains)),
            sparse.csr_matrix(np.ones((1, reward.size))),
        ], format="csc")
        result = linprog(
            -reward, A_ub=constraints, b_ub=np.r_[-self.problem.noise.ravel(), self.n],
            bounds=(0, None), method="highs",
            options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
        )
        return result, widths

    def solve(self, reward, penalty, eta):
        started = time.perf_counter()
        a = reward - eta * penalty
        matrix = a / max(np.max(np.abs(a)), np.finfo(float).tiny)
        upper = lower = gap = violation = np.nan
        status = "cut_limit"
        for step in range(1, 151):
            result, widths = self.restricted(matrix)
            if not result.success:
                status = "restricted_" + str(result.status)
                break
            lower = -result.fun
            mu = np.maximum(0, -result.ineqlin.marginals[:-1]).reshape(self.n, self.k)
            dual = matrix[:, None] + np.einsum("nok,nk,nkij->noij", self.signs, mu, self.problem.grams)
            vals_s, vec_s = np.linalg.eigh(dual[:, 0].real)
            vals_u, vec_u = np.linalg.eigh(dual[:, 1:])
            principal = np.column_stack((vals_s[:, -1], vals_u[..., -1]))
            v = np.concatenate((vec_s[:, None, :, -1], vec_u[..., -1]), axis=1)
            price = max(0, -result.ineqlin.marginals[-1])
            global_price = max(price, float(principal.max()))
            upper = global_price * self.n - (mu * self.problem.noise).sum()
            gap = upper - lower
            violation = max(0, -result.ineqlin.residual.min())
            gate = 1e-5 * max(1, abs(upper))
            if abs(gap) <= gate and violation <= 1e-5:
                status = "solved"
                break
            if gap < -gate:
                status = "dual_below_primal"
                break
            for n, owner in zip(*np.where(principal > price + gate / (4 * self.n))):
                self.bank[n].append(v[n, owner])
                self.owners[n].append(owner)
        info = dict(iter=step, status=status, pobj=float(upper), dobj=float(lower), res_pri=0.0,
                    res_dual=float(violation), gap=float(gap), setup_time=0.0,
                    solve_time=(time.perf_counter() - started) * 1e3)
        if status != "solved":
            return None, None, info
        weights = np.split(result.x, np.cumsum(widths)[:-1])
        blocks = np.zeros((self.n, self.k + 1, self.t, self.t), complex)
        for n, (bank, tags, powers) in enumerate(zip(self.bank, self.owners, weights)):
            a = np.column_stack(bank)
            for owner in range(self.k + 1):
                keep = np.array(tags) == owner
                v = a[:, keep]
                blocks[n, owner] = (v * powers[keep]) @ v.conj().T
        return blocks[:, 0], blocks[:, 1:], info
