"""Controlled synthetic datasets with known relevant features (prompt section 5).

Ground truth is stored explicitly.  XOR/parity generators are constructed so
that each interacting feature has (by construction) weak marginal association
with the label; a test asserts this property.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SyntheticDataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    support: np.ndarray            # ground-truth relevant feature indices
    interaction_support: np.ndarray  # subset of support that is interaction-only
    n_redundant_copies: int = 0
    noise_feature_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    meta: Dict[str, object] = field(default_factory=dict)


def _add_noise(X: np.ndarray, n_noise: int, rng: np.random.Generator) -> np.ndarray:
    if n_noise <= 0:
        return X
    noise = rng.normal(size=(X.shape[0], n_noise))
    return np.hstack([X, noise])


def gen_main_effect(n: int, n_signal: int, n_noise: int, seed: int) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n, n_signal))
    logits = 1.5 * signal.sum(axis=1) / max(1, np.sqrt(n_signal))
    y = (logits + 0.5 * rng.normal(size=n) > 0).astype(int)
    X = _add_noise(signal, n_noise, rng)
    support = np.arange(n_signal, dtype=int)
    return SyntheticDataset("main_effect", X, y, support, np.array([], dtype=int),
                            meta={"n_signal": n_signal, "n_noise": n_noise, "seed": seed})


def gen_redundant_copy(n: int, n_signal: int, n_copies: int, n_noise: int,
                       seed: int) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    core = rng.normal(size=(n, n_signal))
    logits = 1.5 * core.sum(axis=1) / max(1, np.sqrt(n_signal))
    y = (logits + 0.5 * rng.normal(size=n) > 0).astype(int)
    copies = []
    for _ in range(n_copies):
        # correlated copy + small noise
        j = int(rng.integers(0, n_signal))
        copies.append(0.9 * core[:, j] + 0.35 * rng.normal(size=n))
    X_sig = np.hstack([core] + [c.reshape(-1, 1) for c in copies]) if copies else core
    X = _add_noise(X_sig, n_noise, rng)
    support = np.arange(n_signal, dtype=int)
    # redundant copies are *not* counted as independent ground-truth targets
    return SyntheticDataset("redundant_copy", X, y, support, np.array([], dtype=int),
                            n_redundant_copies=n_copies,
                            meta={"n_signal": n_signal, "n_copies": n_copies,
                                  "n_noise": n_noise, "seed": seed})


def gen_xor2(n: int, n_noise: int, seed: int) -> SyntheticDataset:
    """2-way XOR: y = x0 XOR x1 with weak marginals by construction."""
    rng = np.random.default_rng(seed)
    a = (rng.random(n) < 0.5).astype(float)
    b = (rng.random(n) < 0.5).astype(float)
    y = np.bitwise_xor(a.astype(int), b.astype(int))
    signal = np.column_stack([a + 0.05 * rng.normal(size=n),
                              b + 0.05 * rng.normal(size=n)])
    X = _add_noise(signal, n_noise, rng)
    support = np.array([0, 1], dtype=int)
    return SyntheticDataset("xor2", X, y, support, support.copy(),
                            meta={"n_noise": n_noise, "seed": seed})


def gen_parity3(n: int, n_noise: int, seed: int) -> SyntheticDataset:
    """3-way parity: y = x0 XOR x1 XOR x2."""
    rng = np.random.default_rng(seed)
    a = (rng.random(n) < 0.5).astype(int)
    b = (rng.random(n) < 0.5).astype(int)
    c = (rng.random(n) < 0.5).astype(int)
    y = np.bitwise_xor(np.bitwise_xor(a, b), c)
    signal = np.column_stack([a + 0.05 * rng.normal(size=n),
                              b + 0.05 * rng.normal(size=n),
                              c + 0.05 * rng.normal(size=n)])
    X = _add_noise(signal, n_noise, rng)
    support = np.array([0, 1, 2], dtype=int)
    return SyntheticDataset("parity3", X, y, support, support.copy(),
                            meta={"n_noise": n_noise, "seed": seed})


def gen_mixed(n: int, n_main: int, n_noise: int, seed: int) -> SyntheticDataset:
    """Main effects (0..n_main-1) plus a 2-way XOR block (n_main, n_main+1)."""
    rng = np.random.default_rng(seed)
    main = rng.normal(size=(n, n_main))
    a = (rng.random(n) < 0.5).astype(float)
    b = (rng.random(n) < 0.5).astype(float)
    xor_y = np.bitwise_xor(a.astype(int), b.astype(int))
    logits = 1.2 * main.sum(axis=1) / max(1, np.sqrt(n_main)) + 0.0
    base = (logits + 0.5 * rng.normal(size=n) > 0).astype(int)
    y = np.bitwise_xor(base, xor_y)
    xor_block = np.column_stack([a + 0.05 * rng.normal(size=n),
                                 b + 0.05 * rng.normal(size=n)])
    X = np.hstack([main, xor_block])
    X = _add_noise(X, n_noise, rng)
    support = np.arange(n_main + 2, dtype=int)
    inter = np.array([n_main, n_main + 1], dtype=int)
    return SyntheticDataset("mixed_main_interaction", X, y, support, inter,
                            meta={"n_main": n_main, "n_noise": n_noise, "seed": seed})


def gen_interaction_noise(n: int, n_noise: int, seed: int,
                          n_main: int = 2) -> SyntheticDataset:
    """XOR block plus main effects plus many noise features."""
    rng = np.random.default_rng(seed)
    main = rng.normal(size=(n, n_main))
    a = (rng.random(n) < 0.5).astype(float)
    b = (rng.random(n) < 0.5).astype(float)
    xor_y = np.bitwise_xor(a.astype(int), b.astype(int))
    logits = 1.2 * main.sum(axis=1) / max(1, np.sqrt(n_main))
    base = (logits + 0.5 * rng.normal(size=n) > 0).astype(int)
    y = np.bitwise_xor(base, xor_y)
    xor_block = np.column_stack([a + 0.05 * rng.normal(size=n),
                                 b + 0.05 * rng.normal(size=n)])
    X = np.hstack([main, xor_block])
    X = _add_noise(X, n_noise, rng)
    support = np.arange(n_main + 2, dtype=int)
    inter = np.array([n_main, n_main + 1], dtype=int)
    return SyntheticDataset("interaction_many_noise", X, y, support, inter,
                            meta={"n_main": n_main, "n_noise": n_noise, "seed": seed})


def gen_correlated_noise(n: int, n_signal: int, n_noise: int, seed: int,
                         rho: float = 0.8) -> SyntheticDataset:
    """Main effects + a block of strongly correlated noise features."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n, n_signal))
    logits = 1.5 * signal.sum(axis=1) / max(1, np.sqrt(n_signal))
    y = (logits + 0.5 * rng.normal(size=n) > 0).astype(int)
    # correlated noise block via AR(1)-like construction
    z1 = rng.normal(size=n)
    cols = [z1]
    for _ in range(max(0, n_noise - 1)):
        cols.append(rho * cols[-1] + np.sqrt(1 - rho ** 2) * rng.normal(size=n))
    cnoise = np.column_stack(cols) if cols else np.zeros((n, 0))
    X = np.hstack([signal, cnoise])
    support = np.arange(n_signal, dtype=int)
    return SyntheticDataset("correlated_noise", X, y, support, np.array([], dtype=int),
                            meta={"n_signal": n_signal, "n_noise": n_noise,
                                  "rho": rho, "seed": seed})


GENERATORS = {
    "main_effect": gen_main_effect,
    "redundant_copy": gen_redundant_copy,
    "xor2": gen_xor2,
    "parity3": gen_parity3,
    "mixed": gen_mixed,
    "interaction_noise": gen_interaction_noise,
    "correlated_noise": gen_correlated_noise,
}


def generate(name: str, n: int, seed: int, **kwargs) -> SyntheticDataset:
    fn = GENERATORS[name]
    return fn(n=n, seed=seed, **kwargs)


def recovery_metrics(selected: Sequence[int] | np.ndarray, support: np.ndarray) -> Dict[str, float]:
    S = set(int(i) for i in np.asarray(selected, dtype=int).ravel())
    T = set(int(i) for i in np.asarray(support, dtype=int).ravel())
    tp = len(S & T)
    prec = tp / len(S) if S else 0.0
    rec = tp / len(T) if T else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    jac = tp / len(S | T) if (S | T) else 1.0
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "jaccard": float(jac),
        "all_selected": float(T <= S) if T else 1.0,
        "n_selected": float(len(S)),
        "n_support": float(len(T)),
    }


def marginal_association_check(X_col: np.ndarray, y: np.ndarray, bins: int = 5) -> float:
    """Discretized NMI between one feature and y (for XOR weakness assertion)."""
    from .discretization import Discretizer
    from .information import nmi
    Xd = Discretizer(method="quantile", bins=bins).fit_transform(
        np.asarray(X_col, dtype=float).reshape(-1, 1))
    return float(nmi(Xd[:, 0], y))


def export_interaction_example_table(path: Path, seed: int = 0) -> Path:
    """Deterministic small numerical feature-interaction example -> CSV."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    n = 40
    x1 = (rng.random(n) < 0.5).astype(int)
    x2 = (rng.random(n) < 0.5).astype(int)
    y = np.bitwise_xor(x1, x2)
    x3 = (x1 + x2 + rng.integers(0, 2, n)) % 2  # dependent helper
    df = pd.DataFrame({
        "row": np.arange(n),
        "x1_binary": x1,
        "x2_binary": x2,
        "y_xor": y,
        "x3_noisy_sum": x3,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
