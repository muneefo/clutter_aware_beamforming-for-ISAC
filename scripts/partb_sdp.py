"""Certified Charnes–Cooper and per-tone power-price SDP variants."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from parta_ao import pair_sum
from parta_eigencut import EigenCutSDP
from parta_incremental import IncrementalCutSDP


def blocks_from_bank(bank, owners, sizes, weights, users, antennas):
    blocks = np.zeros((len(bank), users + 1, antennas, antennas), complex)
    for n, (directions, tags, powers) in enumerate(zip(bank, owners, np.split(weights, np.cumsum(sizes)[:-1]))):
        v = np.column_stack(directions)
        for owner in range(users + 1):
            take = np.asarray(tags) == owner
            u = v[:, take]
            blocks[n, owner] = (u * powers[take]) @ u.conj().T
    return blocks


def principal_modes(matrix):
    vals_s, vec_s = np.linalg.eigh(matrix[:, 0].real)
    vals_u, vec_u = np.linalg.eigh(matrix[:, 1:])
    return (np.column_stack((vals_s[:, -1], vals_u[..., -1])),
            np.concatenate((vec_s[:, None, :, -1], vec_u[..., -1]), axis=1))


class CooperSDP(IncrementalCutSDP):
    """One fractional SDP, solved with cuts in the full PSD cones."""

    def __init__(self, problem, persistent=False):
        super().__init__(problem, fractional=persistent)

    def restricted_cooper(self, target, penalty):
        if self.fractional:
            return self.restricted(target, penalty)
        start = perf_counter()
        widths = list(map(len, self.bank))
        constraints, rewards, penalties = [], [], []
        for n, (bank, owners) in enumerate(zip(self.bank, self.owners)):
            v = np.column_stack(bank)
            gains = np.abs(self.channels[n].conj() @ v) ** 2
            constraints.append(-gains * self.signs[n, owners].T)
            rewards.append(np.sum((v.conj() * (target[n] @ v)).real, axis=0))
            penalties.append(np.sum((v.conj() * (penalty[n] @ v)).real, axis=0))
        a, c = np.concatenate(rewards), np.concatenate(penalties)
        qos = sparse.hstack([sparse.block_diag(tuple(constraints)),
                             sparse.csc_matrix(self.problem.noise.reshape(-1, 1))])
        a_ub = sparse.vstack([qos, sparse.csc_matrix(np.r_[np.ones(a.size), -self.power_cap][None])],
                             format="csc")
        self.profile["build"] += perf_counter() - start
        start = perf_counter()
        result = linprog(-np.r_[a, 0.], A_ub=a_ub, b_ub=np.zeros(self.n * self.k + 1),
                         A_eq=np.r_[c, 1.][None], b_eq=[1.], bounds=(0, None), method="highs",
                         options={"primal_feasibility_tolerance": 1e-9,
                                  "dual_feasibility_tolerance": 1e-9, "threads": 1})
        self.profile["solve"] += perf_counter() - start
        return result, widths

    def solve(self, reward, penalty, _eta):
        scale = float(np.max(np.abs(reward)))
        target = reward / scale
        min_penalty = min(0., float(np.linalg.eigvalsh(penalty).min()))
        t_max = 1 / (1 + self.power_cap * min_penalty)
        lower = upper = violation = gap = np.nan
        status = "cut_limit"
        for step in range(1, 151):
            result, widths = self.restricted_cooper(target, penalty)
            if not result.success:
                status = f"restricted_{result.status}"
                break
            start = perf_counter()
            lower = -result.fun
            mu = np.maximum(0., -result.ineqlin.marginals[:-1]).reshape(self.n, self.k)
            price = max(0., -result.ineqlin.marginals[-1])
            psi = -result.eqlin.marginals[0]
            base = (target - psi * penalty)[:, None]
            dual = base + np.einsum("nok,nk,nkij->noij", self.signs, mu, self.problem.grams)
            vals, modes = principal_modes(dual)
            excess = vals - price
            scale_violation = max(0., self.power_cap * price - np.sum(mu * self.problem.noise) - psi)
            upper = psi + (self.power_cap * max(0., float(excess.max())) + scale_violation) * t_max
            gap = upper - lower
            violation = max(0., -result.ineqlin.residual.min(), np.max(np.abs(result.eqlin.residual)))
            self.profile["price"] += perf_counter() - start
            gate = 1e-5 * max(1., abs(upper))
            if abs(gap) <= gate and violation <= 1e-5 and result.x[-1] > 0:
                status = "solved"
                break
            if gap < -gate or result.x[-1] <= 0:
                status = "verification_failed"
                break
            for n, owner in zip(*np.where(excess > gate / (4 * self.power_cap * t_max))):
                self.bank[n].append(modes[n, owner])
                self.owners[n].append(owner)
        info = dict(iter=step, status=status, pobj=float(upper), dobj=float(lower), gap=float(gap),
                    res_dual=float(violation), price_evals=0, local_masters=step)
        if status != "solved":
            return None, None, info
        blocks = blocks_from_bank(self.bank, self.owners, widths, result.x[:-1] / result.x[-1], self.k, self.t)
        return blocks[:, 0], blocks[:, 1:], info


class TonePriceSDP:
    """Solve a Dinkelbach SDP by bisection on the global power multiplier."""

    def __init__(self, problem, workers, cold_retry=False):
        self.problem = problem
        self.bounds = [IncrementalCutSDP(SimpleNamespace(
            n=1, k=problem.k, t=problem.t, power_cap=problem.n,
            grams=problem.grams[i:i + 1], normalized=problem.normalized[i:i + 1],
            gamma=problem.gamma[i:i + 1], noise=problem.noise[i:i + 1],
        )) for i in range(problem.n)]
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self.workers = workers
        self.cold_retry = cold_retry
        self.cold_retries = 0
        self.profile = dict(build=0., solve=0., price=0., parallel_tone_wall=0.)
        self.worker_profile = dict(build=0., solve=0., price=0.)
        self.identity = np.eye(problem.t)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()

    def at_price(self, matrix, price):
        def solve_tone(i):
            model = self.bounds[i]
            local = matrix[i:i + 1] - price * self.identity
            sensing, users, info = model.solve_matrix(local, absolute_gate=1e-7)
            if self.cold_retry and info["status"] != "solved":
                retry = EigenCutSDP(model.problem)
                retry.master_threads = 1
                retry.bank, retry.owners = model.bank, model.owners
                sense, use, new = retry.solve_matrix(local, absolute_gate=1e-7)
                new["cold_retry_from"] = info["status"]
                new["iter"] += info["iter"]
                return sense, use, new
            return sensing, users, info

        started = perf_counter()
        if self.pool is None:
            rows = [solve_tone(i) for i in range(len(self.bounds))]
        else:
            rows = list(self.pool.map(solve_tone, range(len(self.bounds))))
            self.profile["parallel_tone_wall"] += perf_counter() - started
        for key in ("build", "solve", "price"):
            subtotal = sum(p.profile[key] for p in self.bounds)
            self.worker_profile[key] = subtotal
            if self.pool is None:
                self.profile[key] = subtotal
        metadata = [row[2] for row in rows]
        masters = sum(row["iter"] for row in metadata)
        self.cold_retries += sum("cold_retry_from" in row for row in metadata)
        failure = next((row["status"] for row in metadata if row["status"] != "solved"), None)
        if failure is not None:
            return None, None, None, masters, failure
        sensing = np.concatenate([row[0] for row in rows])
        users = np.concatenate([row[1] for row in rows])
        q = sensing + users.sum(axis=1)
        power = np.trace(q, axis1=-1, axis2=-2).real.sum()
        upper = price * len(rows) + sum(row["pobj"] for row in metadata)
        return (power, sensing, users), upper, max(row["res_dual"] for row in metadata), masters, None

    def solve(self, reward, penalty, eta):
        objective = reward - eta * penalty
        matrix = objective / np.max(np.abs(objective))
        cap = self.problem.n
        low_price = 0.
        high_price = max(1., float(np.linalg.eigvalsh(matrix).max()))
        price = low_price
        low, high = None, None
        best_bound = np.inf
        value = violation = gap = np.nan
        total_masters = 0
        status = "price_limit"
        bisections = 0
        is_midpoint = False
        for count in range(1, 65):
            bisections += int(is_midpoint)
            result, bound, residual, masters, failure = self.at_price(matrix, price)
            total_masters += masters
            if failure is not None:
                status = f"tone_{failure}"
                break
            best_bound = min(best_bound, bound)
            if result[0] <= cap:
                high, high_price = result, price
            else:
                low, low_price = result, price
            if high is not None:
                sensing, users = high[1:]
                if low is not None:
                    mix = (cap - high[0]) / (low[0] - high[0])
                    sensing = (1 - mix) * sensing + mix * low[1]
                    users = (1 - mix) * users + mix * low[2]
                q = sensing + users.sum(axis=1)
                value = pair_sum(matrix, q)
                gap = best_bound - value
                physical = self.problem.residuals(sensing, users, q)
                violation = max(0., residual, -min(physical.values()))
                if abs(gap) <= 1e-5 * max(1, abs(best_bound)) and violation <= 1e-5:
                    status = "solved"
                    break
            if high is None:
                if price >= high_price:
                    high_price *= 2
                price = high_price
                is_midpoint = False
            else:
                price = (low_price + high_price) / 2
                is_midpoint = True
        info = dict(iter=count, status=status, pobj=float(best_bound), dobj=float(value), gap=float(gap),
                    res_dual=float(violation), price_evals=count, bisection_steps=bisections, local_masters=total_masters)
        if status != "solved":
            return None, None, info
        return sensing, users, info
