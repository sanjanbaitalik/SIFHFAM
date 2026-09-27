"""Discrete information-theoretic quantities (prompt sections 6, 7).

Entropy base: natural log (nats) throughout.  Default estimator: ordinary
plug-in (maximum-likelihood) discrete entropy.  An optional Miller-Madow
corrected plug-in estimator is provided for sensitivity only.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np

EPS = 1e-12
ENTROPY_BASE = "nats (natural log)"


def _counts(arr: np.ndarray) -> np.ndarray:
    """Counts of unique rows (or elements for 1-D input)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        _, counts = np.unique(arr, return_counts=True)
        return counts.astype(np.float64)
    _, counts = np.unique(arr, axis=0, return_counts=True)
    return counts.astype(np.float64)


def entropy_from_counts(counts: np.ndarray, estimator: str = "plugin",
                        n_samples: int | None = None) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    h = float(-np.sum(p * np.log(p)))
    if estimator == "plugin":
        return h
    if estimator == "miller_madow":
        # Miller-Madow: H_MM = H_plugin + (K - 1) / (2n)  [nats]
        k = int(p.size)
        n = float(total if n_samples is None else n_samples)
        return float(h + (k - 1) / (2.0 * max(n, 1.0)))
    raise ValueError(f"Unknown estimator: {estimator}")


def entropy(x: np.ndarray, estimator: str = "plugin") -> float:
    return entropy_from_counts(_counts(x), estimator=estimator)


def joint_entropy(cols: np.ndarray, estimator: str = "plugin") -> float:
    arr = np.asarray(cols)
    if arr.ndim == 1:
        return entropy(arr, estimator=estimator)
    return entropy_from_counts(_counts(arr), estimator=estimator)


def mutual_information(x: np.ndarray, y: np.ndarray, estimator: str = "plugin") -> float:
    """I(X;Y) = H(X) + H(Y) - H(X,Y), clamped at >= 0."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    stacked = np.stack([x, y], axis=1)
    mi = entropy(x, estimator) + entropy(y, estimator) - joint_entropy(stacked, estimator)
    return float(max(0.0, mi))


def nmi(x: np.ndarray, y: np.ndarray, estimator: str = "plugin") -> float:
    """NMI(X;Y) = I(X;Y) / min(H(X), H(Y)); 0 when the denominator is ~0."""
    hx = entropy(np.asarray(x).ravel(), estimator)
    hy = entropy(np.asarray(y).ravel(), estimator)
    denom = min(hx, hy)
    if denom <= EPS:
        return 0.0
    return float(min(1.0, mutual_information(x, y, estimator) / denom))


def feature_nmi_scores(X_disc: np.ndarray, y: np.ndarray,
                       estimator: str = "plugin") -> np.ndarray:
    """Fast NMI(X_j; y) for integer-coded X_disc and integer labels y."""
    Xd = np.asarray(X_disc)
    y = np.asarray(y).ravel().astype(np.int64)
    if Xd.dtype.kind not in "iub":
        raise ValueError("feature_nmi_scores requires integer codes")
    n, d = Xd.shape
    Xd = Xd.astype(np.int64, copy=False)
    xo = int(min(0, int(Xd.min()), int(y.min())))
    if xo != 0:
        Xd = Xd - xo
        y = y - xo
    Bx = int(max(Xd.max(), y.max())) + 1
    y_off = int(Xd.max()) + 1
    hy = _entropy_from_codes(y, n, estimator)
    out = np.zeros(d, dtype=float)
    if hy <= EPS:
        return out
    for j in range(d):
        hx = _entropy_from_codes(Xd[:, j], n, estimator)
        counts = np.bincount(Xd[:, j] * y_off + y)
        counts = counts[counts > 0].astype(np.float64)
        p = counts / counts.sum()
        hxy = float(-np.sum(p * np.log(p)))
        if estimator == "miller_madow":
            hxy += (int(counts.size) - 1) / (2.0 * max(n, 1.0))
        mi = max(0.0, hx + hy - hxy)
        denom = min(hx, hy)
        out[j] = min(1.0, mi / denom) if denom > EPS else 0.0
    return out


def nmi_multivariate(cols: np.ndarray, y: np.ndarray, estimator: str = "plugin") -> float:
    """NMI(X_e; Y) with X_e a (n, k) block of columns."""
    cols = np.asarray(cols)
    if cols.ndim == 1:
        cols = cols.reshape(-1, 1)
    y = np.asarray(y).ravel()
    stacked = np.hstack([cols, y.reshape(-1, 1)])
    hx = joint_entropy(cols, estimator)
    hy = entropy(y, estimator)
    denom = min(hx, hy)
    if denom <= EPS:
        return 0.0
    hxy = joint_entropy(stacked, estimator)
    mi = max(0.0, hx + hy - hxy)
    return float(min(1.0, mi / denom))


def total_correlation(cols: np.ndarray, estimator: str = "plugin",
                      marginals: Sequence[float] | None = None) -> float:
    """TC(e) = sum_j H(X_j) - H(X_e), clamped at >= 0 (nats)."""
    cols = np.asarray(cols)
    if cols.ndim == 1:
        cols = cols.reshape(-1, 1)
    if marginals is None:
        marginals = [entropy(cols[:, j], estimator) for j in range(cols.shape[1])]
    tc = float(sum(marginals) - joint_entropy(cols, estimator))
    return max(0.0, tc)


def group_redundancy(cols: np.ndarray, estimator: str = "plugin",
                     marginals: Sequence[float] | None = None) -> Tuple[float, float]:
    """Bounded group redundancy R_e = TC(e) / (sum_j H(X_j) + eps).

    Returns (R_e, TC_e).  Safe when the denominator is zero (constant
    features): returns (0.0, 0.0).
    """
    cols = np.asarray(cols)
    if cols.ndim == 1:
        cols = cols.reshape(-1, 1)
    if marginals is None:
        marginals = [entropy(cols[:, j], estimator) for j in range(cols.shape[1])]
    denom = float(sum(marginals))
    tc = total_correlation(cols, estimator=estimator, marginals=marginals)
    if denom <= EPS:
        return 0.0, 0.0
    r = tc / (denom + EPS)
    # numerical safety: TC <= sum H up to floating point
    r = float(min(1.0, max(0.0, r)))
    return r, tc


def _entropy_from_codes(codes: np.ndarray, n: int, estimator: str = "plugin") -> float:
    counts = np.bincount(codes)
    counts = counts[counts > 0].astype(np.float64)
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    h = float(-np.sum(p * np.log(p)))
    if estimator == "miller_madow":
        return h + (int(counts.size) - 1) / (2.0 * max(n, 1.0))
    if estimator == "plugin":
        return h
    raise ValueError(f"Unknown estimator: {estimator}")


def pairwise_nmi_mean(X_disc: np.ndarray, estimator: str = "plugin") -> np.ndarray:
    """Fast mean-NMI redundancy for non-negative integer codes.

    r_j = mean_{i != j} NMI(X_j; X_i), computed with combined-key bincount
    histograms (equivalent to the generic ``nmi`` on the same inputs).
    """
    Xd = np.asarray(X_disc)
    if Xd.dtype.kind not in "iub":
        raise ValueError("pairwise_nmi_mean requires integer codes")
    n, f = Xd.shape
    if f <= 1:
        return np.zeros(f, dtype=float)
    Xd = Xd.astype(np.int64, copy=False)
    offset = int(min(0, Xd.min()))
    if offset != 0:
        Xd = Xd - offset  # ensure non-negative
    B = int(Xd.max()) + 1
    H = np.array([_entropy_from_codes(Xd[:, j], n, estimator) for j in range(f)])
    out = np.zeros(f, dtype=float)
    for i in range(f):
        ci = Xd[:, i]
        Hi = H[i]
        acc = 0.0
        for j in range(f):
            if i == j:
                continue
            counts = np.bincount(ci * B + Xd[:, j])
            counts = counts[counts > 0].astype(np.float64)
            p = counts / counts.sum()
            h_ij = float(-np.sum(p * np.log(p)))
            if estimator == "miller_madow":
                h_ij += (int(counts.size) - 1) / (2.0 * max(n, 1.0))
            mi = max(0.0, Hi + H[j] - h_ij)
            denom = min(Hi, H[j])
            acc += mi / denom if denom > EPS else 0.0
        out[i] = acc / (f - 1)
    return out

