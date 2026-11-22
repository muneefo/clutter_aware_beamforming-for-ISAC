"""Structured log-det barrier solver for the joint transmit SDP dual."""
import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


class BarrierSDP:
    """Use block inverse Hessians and a Schur complement for global power."""

    def __init__(self, scs_problem):
        self.problem = scs_problem
        n, k, t = scs_problem.n, scs_problem.k, scs_problem.t
        self.identity = np.eye(t)
        # Blocks are sensing, then the K user owners. Local prices follow lambda.
        self.derivatives = np.zeros((n, k + 1, k + 1, t, t), complex)
        self.derivatives[:, :, 0] = self.identity
        self.derivatives[:, :, 1:] = scs_problem.grams[:, None]
        self.derivatives[:, 0, 1:] = scs_problem.grams.real
        users = np.arange(k)
        self.derivatives[:, users + 1, users + 1] *= -1 / scs_problem.gamma[..., None, None]
        self.cost = np.column_stack((np.ones(n), -scs_problem.noise))

    def slacks(self, z, m):
        return np.einsum("nv,nbvij->nbij", z, self.derivatives, optimize=False) - m

    def log_barrier(self, z, slack):
        if np.min(z) <= 0:
            return -np.inf
        try:
            chol = np.linalg.cholesky(slack)
        except np.linalg.LinAlgError:
            return -np.inf
        diag = chol.diagonal(axis1=-1, axis2=-2).real
        return 2 * np.log(diag).sum() + np.log(z[:, 1:]).sum() + np.log(z[0, 0])

    def solve(self, reward, penalty, eta):
        clock = time.perf_counter()
        objective = reward - eta * penalty
        scale = max(np.max(np.abs(objective)), np.finfo(float).tiny)
        n, k = self.problem.n, self.problem.k
        m = np.repeat((objective / scale)[:, None], k + 1, axis=1)
        m[:, 0] = m[:, 0].real
        z = np.ones((n, k + 1))
        z[:, 0] = 0.0
        z[:, 0] = max(0.0, np.linalg.eigvalsh(-self.slacks(z, m)).max()) + 1.0
        steps = 0
        for mu in 10.0 ** (-np.arange(2, 10)):
            for _ in range(40):
                steps += 1
                slack = self.slacks(z, m)
                inv = np.linalg.solve(slack, np.broadcast_to(self.identity, slack.shape))
                product = inv[:, :, None] @ self.derivatives
                grad = self.cost - mu * product.trace(axis1=-1, axis2=-2).real.sum(axis=1)
                diag = mu / z ** 2
                grad[:, 1:] -= mu / z[:, 1:]
                grad[0, 0] -= mu / z[0, 0]
                hess = mu * np.einsum("nbuij,nbvji->nuv", product, product).real
                variables = np.arange(1, k + 1)
                hess[:, variables, variables] += diag[:, 1:]
                hess[0, 0, 0] += diag[0, 0]
                cross = hess[:, 1:, 0]
                local = np.linalg.solve(hess[:, 1:, 1:], np.stack((grad[:, 1:], cross), axis=-1))
                dlam = -(grad[:, 0].sum() - (cross * local[:, :, 0]).sum()) / (
                    hess[:, 0, 0].sum() - (cross * local[:, :, 1]).sum()
                )
                delta = np.column_stack((np.full(n, dlam), -local[:, :, 0] - dlam * local[:, :, 1]))
                gradient_step = (grad * delta).sum()
                if -gradient_step < max(1e-26, 1e-16 * mu):
                    break
                old = self.log_barrier(z, slack)
                cost_step = (delta * self.cost).sum()
                d_slack = self.slacks(delta, 0)
                step = 1 / (1 + np.sqrt(max(0., -gradient_step / mu)))
                for _ in range(60):
                    proposal = z + step * delta
                    new = self.log_barrier(proposal, slack + step * d_slack)
                    change = step * cost_step - mu * (new - old)
                    if change <= 0.01 * step * gradient_step:
                        z = proposal
                        break
                    step *= 0.5
                else:
                    break
        slack = self.slacks(z, m)
        primal = self.recover_primal(slack, m)
        sensing, users = primal[:, 0], primal[:, 1:]
        q = sensing + users.sum(axis=1)
        checks = self.problem.residuals(sensing, users, q)
        primal_value = np.einsum("nbij,nbji->", m, primal).real
        dual_value = (z * self.cost).sum()
        gap = abs(dual_value - primal_value)
        violation = max(0.0, -min(checks.values()))
        dual_violation = max(0.0, -float(np.linalg.eigvalsh(slack).min()))
        valid = (violation <= 1e-5 and dual_violation <= 1e-5
                 and gap <= 1e-5 * max(1, abs(dual_value)))
        info = dict(iter=steps, status="solved" if valid else "verification_failed",
                    pobj=float(dual_value), dobj=float(primal_value), res_pri=dual_violation,
                    res_dual=violation, gap=gap, setup_time=0.0,
                    solve_time=(time.perf_counter() - clock) * 1e3)
        if not valid:
            return None, None, info
        return sensing, users, info

    def recover_primal(self, slack, objective):
        # All PSD blocks use real weights. Sensing eigenvectors are real.
        n, k, t = self.problem.n, self.problem.k, self.problem.t
        _, sensed = np.linalg.eigh(slack[:, 0].real)
        _, communication = np.linalg.eigh(slack[:, 1:])
        directions = np.concatenate((sensed[:, None], communication), axis=1)
        channels = self.problem.normalized
        zf = np.linalg.pinv(channels.conj()).transpose(0, 2, 1)[..., None]
        zf /= np.linalg.norm(zf, axis=-2, keepdims=True)
        pad = np.concatenate((sensed[:, None, :, :1], zf), axis=1)
        directions = np.concatenate((directions, pad), axis=-1)
        gain = np.abs(channels[:, None].conj() @ directions) ** 2
        reward = np.sum((directions.conj() * (objective @ directions)).real, axis=-2)
        signs = -np.ones((n, k + 1, k))
        owners = np.arange(k)
        signs[:, owners + 1, owners] = 1 / self.problem.gamma
        coefficients = (gain * signs[..., None]).transpose(0, 2, 1, 3).reshape(n, k, -1)
        constraints = sparse.vstack([
            -sparse.block_diag(tuple(coefficients)),
            sparse.csr_matrix(np.ones((1, reward.size))),
        ], format="csc")
        result = linprog(-reward.ravel(), A_ub=constraints, b_ub=np.r_[-self.problem.noise.ravel(), n],
                         bounds=(0, None), method="highs",
                         options={"primal_feasibility_tolerance": 1e-9,
                                  "dual_feasibility_tolerance": 1e-9})
        if not result.success:
            return np.zeros((n, k + 1, t, t))
        allocation = result.x.reshape(n, k + 1, 1, t + 1)
        return (directions * allocation) @ directions.swapaxes(-1, -2).conj()
