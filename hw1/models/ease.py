"""EASE closed-form ridge solution with zero diagonal."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

class EASE:
    def __init__(self, reg=200.):
        self.reg = reg

    def fit(self, data):
        self.x = data.positive
        gram = (self.x.T @ self.x).toarray().astype(np.float64)
        gram.flat[::gram.shape[0]+1] += self.reg
        precision = cho_solve(cho_factor(gram, overwrite_a=True), np.eye(gram.shape[0]))
        self.weights = (-precision / np.diag(precision)[None, :]).astype(np.float32)
        np.fill_diagonal(self.weights, 0.)
        return self

    def score(self, users):
        return np.asarray(self.x[users] @ self.weights)
