"""Weighted implicit ALS: normal equations, no external recommender library."""
import numpy as np
from scipy import sparse

class IALS:
    def __init__(self, factors=32, reg=1.0, alpha=20., iterations=8, seed=42):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    def targets(self, data):
        return (data.frame.rating.to_numpy() >= 4).astype(float)

    def fit(self, data):
        f = data.frame
        shape = data.positive.shape
        # P retains explicit zeros: confidence is identical for every observed pair.
        p = sparse.csr_matrix((self.targets(data), (f.u, f.i)), shape=shape)
        c = sparse.csr_matrix((np.ones(len(f)), (f.u, f.i)), shape=shape)
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, .03, (shape[0], self.factors))
        self.item_factors = rng.normal(0, .03, (shape[1], self.factors))
        def update(obs, targets, fixed):
            result = np.zeros((obs.shape[0], self.factors))
            base = fixed.T @ fixed + self.reg * np.eye(self.factors)
            for row in range(obs.shape[0]):
                lo, hi = obs.indptr[row:row+2]
                idx = obs.indices[lo:hi]
                if not len(idx):
                    continue
                y = fixed[idx]
                a = base + self.alpha * (y.T @ y)
                b = (1+self.alpha) * y.T @ targets.data[lo:hi]
                result[row] = np.linalg.solve(a, b)
            return result
        assert np.array_equal(c.indices, p.indices)
        for _ in range(self.iterations):
            self.user_factors = update(c, p, self.item_factors)
            self.item_factors = update(c.T.tocsr(), p.T.tocsr(), self.user_factors)
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
