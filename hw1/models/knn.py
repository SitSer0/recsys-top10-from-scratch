"""Item cosine kNN, implemented with sparse matrix operations."""
import numpy as np
from scipy import sparse

class ItemKNN:
    def __init__(self, neighbors=50, shrink=10):
        self.neighbors, self.shrink = neighbors, shrink

    def fit(self, data):
        self.x = data.positive
        gram = (self.x.T @ self.x).tocsr()
        norm = np.sqrt(self.x.power(2).sum(axis=0)).A1
        rows, cols = gram.nonzero()
        gram.data /= norm[rows] * norm[cols] + self.shrink
        gram.setdiag(0)
        gram.eliminate_zeros()
        # Keep neighbors per destination item; score_j = sum_i X_ui S_ij.
        gram = gram.tocsc()
        for j in range(gram.shape[1]):
            lo, hi = gram.indptr[j:j+2]
            if hi-lo > self.neighbors:
                ids = np.lexsort((gram.indices[lo:hi], -gram.data[lo:hi]))
                gram.data[lo:hi][ids[self.neighbors:]] = 0
        gram.eliminate_zeros()
        self.similarity = gram.tocsr()
        return self

    def score(self, users):
        return (self.x[users] @ self.similarity).toarray()
