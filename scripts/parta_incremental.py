"""Persistent HiGHS master for the certified SDP eigencut solver."""
import time
from types import SimpleNamespace

import highspy
import numpy as np

from parta_eigencut import EigenCutSDP


class IncrementalCutSDP(EigenCutSDP):
    def __init__(self, problem, fractional=False):
        super().__init__(problem)
        self.fractional = fractional
        self.model = highspy.Highs()
        for option, value in (
            ("output_flag", False),
            ("threads", 1),
            ("solver", "simplex"),
            ("presolve", "off"),
            ("primal_feasibility_tolerance", 1e-9),
            ("dual_feasibility_tolerance", 1e-9),
        ):
            self.model.setOptionValue(option, value)
        if fractional:
            self.limit = np.r_[np.zeros(self.n * self.k + 1), 1.]
            lower = np.r_[np.full(self.n * self.k + 1, -np.inf), 1.]
        else:
            self.limit = np.r_[-self.problem.noise.ravel(), self.power_cap]
            lower = np.full(self.limit.size, -np.inf)
        self.model.addRows(self.limit.size, lower, self.limit,
                           0, np.zeros(1, np.int32), np.empty(0, np.int32), np.empty(0))
        if fractional:
            rows = self.limit.size
            self.model.addCols(1, np.zeros(1), np.zeros(1), np.full(1, np.inf), rows,
                               np.array([0, rows], np.int32), np.arange(rows, dtype=np.int32),
                               np.r_[self.problem.noise.ravel(), -self.power_cap, 1.])
        self.indices = [[] for _ in range(self.n)]
        self.previous_matrix = None
        self.initial_basis = None
        self.basis_status = None

    def restricted(self, matrix, penalty=None):
        build_at = time.perf_counter()
        changing_objective = matrix is not self.previous_matrix
        self.previous_matrix = matrix
        costs, rows, values = [], [], []
        next_column = self.model.getNumCol()
        for n, (bank, owners, ids) in enumerate(zip(self.bank, self.owners, self.indices)):
            cached = len(ids)
            vectors = np.asarray(bank).T
            if changing_objective and cached:
                old = vectors[:, :cached]
                cost = -np.sum((old.conj() * (matrix[n] @ old)).real, axis=0)
                self.model.changeColsCost(cached, np.asarray(ids, np.int32), cost)
                if self.fractional:
                    coeff = np.sum((old.conj() * (penalty[n] @ old)).real, axis=0)
                    for col, value in zip(ids, coeff):
                        self.model.changeCoeff(self.limit.size - 1, col, float(value))
            if cached < len(bank):
                new = vectors[:, cached:]
                fresh = len(bank) - cached
                gain = np.abs(self.channels[n].conj() @ new) ** 2
                coeff = -(gain * self.signs[n, owners[cached:]].T).T
                row_ids = np.r_[n * self.k + np.arange(self.k), self.n * self.k]
                coeff = np.column_stack((coeff, np.ones(fresh)))
                if self.fractional:
                    row_ids = np.r_[row_ids, self.limit.size - 1]
                    c = np.sum((new.conj() * (penalty[n] @ new)).real, axis=0)
                    coeff = np.column_stack((coeff, c))
                values.append(coeff.ravel())
                rows.append(np.tile(row_ids, fresh))
                costs.append(-np.sum((new.conj() * (matrix[n] @ new)).real, axis=0))
                ids.extend(range(next_column, next_column + fresh))
                next_column += fresh
        if costs:
            new_costs = np.concatenate(costs)
            count = new_costs.size
            coeff = np.concatenate(values)
            stride = self.k + 1 + int(self.fractional)
            state = self.model.addCols(
                count, new_costs, np.zeros(count), np.full(count, np.inf), coeff.size,
                np.arange(count + 1, dtype=np.int32) * stride,
                np.concatenate(rows).astype(np.int32), coeff,
            )
            if state == highspy.HighsStatus.kError:
                raise RuntimeError("HiGHS rejected the new master columns")
        if self.initial_basis is not None:
            self.basis_status = int(self.model.setBasis(self.initial_basis))
            self.initial_basis = None
        self.profile["build"] += time.perf_counter() - build_at
        solve_at = time.perf_counter()
        self.model.run()
        self.profile["solve"] += time.perf_counter() - solve_at
        state = self.model.getModelStatus()
        result = SimpleNamespace(success=state == highspy.HighsModelStatus.kOptimal, status=int(state))
        if result.success:
            solution = self.model.getSolution()
            columns = sum(self.indices, [])
            result.x = np.asarray(solution.col_value)[columns + [0] if self.fractional else columns]
            result.fun = self.model.getObjectiveValue()
            marginals = np.asarray(solution.row_dual)
            residual = self.limit - np.asarray(solution.row_value)
            if self.fractional:
                result.eqlin = SimpleNamespace(marginals=marginals[-1:], residual=residual[-1:])
                marginals, residual = marginals[:-1], residual[:-1]
            result.ineqlin = SimpleNamespace(marginals=marginals, residual=residual)
        return result, list(map(len, self.indices))
