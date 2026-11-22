"""Persistent HiGHS master for the certified SDP eigencut solver."""
from types import SimpleNamespace

import highspy
import numpy as np

from parta_eigencut import EigenCutSDP


class IncrementalCutSDP(EigenCutSDP):
    def __init__(self, problem):
        super().__init__(problem)
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
        self.limit = np.r_[-self.problem.noise.ravel(), self.n]
        self.model.addRows(self.limit.size, np.full(self.limit.size, -np.inf), self.limit,
                           0, np.zeros(1, np.int32), np.empty(0, np.int32), np.empty(0))
        self.indices = [[] for _ in range(self.n)]
        self.previous_matrix = None

    def restricted(self, matrix):
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
            if cached < len(bank):
                new = vectors[:, cached:]
                fresh = len(bank) - cached
                gain = np.abs(self.channels[n].conj() @ new) ** 2
                coeff = -(gain * self.signs[n, owners[cached:]].T).T
                values.append(np.column_stack((coeff, np.ones(fresh))).ravel())
                rows.append(np.tile(np.r_[n * self.k + np.arange(self.k), self.n * self.k], fresh))
                costs.append(-np.sum((new.conj() * (matrix[n] @ new)).real, axis=0))
                ids.extend(range(next_column, next_column + fresh))
                next_column += fresh
        if costs:
            new_costs = np.concatenate(costs)
            count = new_costs.size
            coeff = np.concatenate(values)
            state = self.model.addCols(
                count, new_costs, np.zeros(count), np.full(count, np.inf), coeff.size,
                np.arange(count + 1, dtype=np.int32) * (self.k + 1),
                np.concatenate(rows).astype(np.int32), coeff,
            )
            if state == highspy.HighsStatus.kError:
                raise RuntimeError("HiGHS rejected the new master columns")
        self.model.run()
        state = self.model.getModelStatus()
        result = SimpleNamespace(success=state == highspy.HighsModelStatus.kOptimal, status=int(state))
        if result.success:
            solution = self.model.getSolution()
            result.x = np.asarray(solution.col_value)[sum(self.indices, [])]
            result.fun = self.model.getObjectiveValue()
            result.ineqlin = SimpleNamespace(
                marginals=np.asarray(solution.row_dual),
                residual=self.limit - np.asarray(solution.row_value),
            )
        return result, list(map(len, self.indices))
