"""Second-order FM with BCE, train-history metadata, own negative sampler."""
import numpy as np
import torch
from torch import nn

class FM:
    def __init__(self, factors=16, lr=.01, reg=1e-5, epochs=8, negatives=2,
                 metadata=True, seed=42, batch_size=4096):
        self.factors, self.lr, self.reg = factors, lr, reg
        self.epochs, self.negatives, self.metadata = epochs, negatives, metadata
        self.seed, self.batch_size = seed, batch_size

    def fit(self, data):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.data = data
        self.n_items = data.positive.shape[1]
        self.linear = nn.Embedding(data.n_features, 1)
        self.embedding = nn.Embedding(data.n_features, self.factors)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=.02)
        params = [self.bias, self.linear.weight, self.embedding.weight]
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.reg)
        f = data.frame
        observed_u, observed_i = f.u.to_numpy(), f.i.to_numpy()
        labels = (f.rating.to_numpy() >= 4).astype(np.float32)
        pos_u = observed_u[labels == 1]
        self.losses = []
        for _ in range(self.epochs):
            nu = np.repeat(pos_u, self.negatives)
            ni = rng.integers(self.n_items, size=len(nu))
            # Reject every observed training pair, including low ratings.
            invalid = np.asarray(data.seen[nu, ni]).ravel() > 0
            while invalid.any():
                ni[invalid] = rng.integers(self.n_items, size=invalid.sum())
                invalid = np.asarray(data.seen[nu, ni]).ravel() > 0
            us = np.r_[observed_u, nu]; it = np.r_[observed_i, ni]
            ys = np.r_[labels, np.zeros(len(nu), np.float32)]
            order = rng.permutation(len(us))
            total = 0.
            for start in range(0, len(us), self.batch_size):
                b = order[start:start+self.batch_size]
                feats = torch.as_tensor(data.features(us[b], it[b], self.metadata))
                pred = self.forward(feats)
                loss = nn.functional.binary_cross_entropy_with_logits(pred, torch.from_numpy(ys[b]))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += loss.item() * len(b)
            self.losses.append(total / len(us))
        return self

    def forward(self, features):
        v = self.embedding(features)
        interactions = .5 * (v.sum(1).square() - v.square().sum(1)).sum(1)
        return self.bias + self.linear(features).sum((1, 2)) + interactions

    def score(self, users):
        users = np.asarray(users)
        result = np.empty((len(users), self.n_items), np.float32)
        with torch.no_grad():
            for row, u in enumerate(users):
                for start in range(0, self.n_items, self.batch_size):
                    items = np.arange(start, min(start+self.batch_size, self.n_items))
                    feats = self.data.features(np.full(len(items), u), items, self.metadata)
                    result[row, start:start+len(items)] = self.forward(torch.as_tensor(feats)).numpy()
        return result
