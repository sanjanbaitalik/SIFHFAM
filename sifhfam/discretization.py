"""Train-only discretization (prompt: never fit on the test split)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Discretizer:
    """Column-wise 1-D discretizer fit on the training fold only."""

    method: str = "quantile"   # quantile (equal-frequency) | uniform (equal-width)
    bins: int = 10
    edges_: Optional[list] = None  # list of per-column edge arrays (variable length)

    def fit(self, X: np.ndarray) -> "Discretizer":
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        b = max(2, int(self.bins))
        qs = np.linspace(0, 100, b + 1)[1:-1]
        edges: list = []
        if self.method == "quantile":
            # vectorized percentile computation, per-column unique afterwards
            q_edges = np.percentile(X, qs, axis=0)  # (b-1, d)
            for j in range(d):
                col = X[:, j]
                if np.allclose(col, col[0]):
                    edges.append(np.array([col[0]], dtype=float))
                else:
                    edges.append(np.unique(q_edges[:, j].astype(float)))
        elif self.method == "uniform":
            col_min = X.min(axis=0)
            col_max = X.max(axis=0)
            u_edges = np.linspace(0, 1, b + 1)[1:-1]
            for j in range(d):
                if np.allclose(X[:, j], X[0, j]):
                    edges.append(np.array([X[0, j]], dtype=float))
                else:
                    span = col_max[j] - col_min[j]
                    edges.append(np.unique(col_min[j] + u_edges * span))
        else:
            raise ValueError(f"Unknown discretizer: {self.method}")
        self.edges_ = edges
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.edges_ is None:
            raise RuntimeError("Discretizer not fitted")
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if d != len(self.edges_):
            raise ValueError(f"Expected {len(self.edges_)} columns, got {d}")
        out = np.zeros((n, d), dtype=np.int32)
        for j in range(d):
            e = self.edges_[j]
            out[:, j] = np.searchsorted(e, X[:, j], side="right").astype(np.int32)
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def discretize(X: np.ndarray, method: str = "quantile", bins: int = 10) -> np.ndarray:
    return Discretizer(method=method, bins=bins).fit_transform(X)
