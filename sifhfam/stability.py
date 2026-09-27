"""Stability measures and instability decomposition (prompt section 14)."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    A, B = set(map(int, a)), set(map(int, b))
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))


def kuncheva(a: Sequence[int], b: Sequence[int], d: int) -> float:
    A, B = set(map(int, a)), set(map(int, b))
    k, l = len(A), len(B)
    if k == 0 or l == 0 or d <= 1:
        return float("nan")
    r = len(A & B)
    expected = k * l / d
    denom = min(k, l) - expected
    if abs(denom) < 1e-12:
        return float("nan")
    return float((r - expected) / denom)


def nogueira_stability(selections: Sequence[Sequence[int]], d: int) -> float:
    """Nogueira et al. (2018) stability estimator.

    S = 1 - ( (1/d) * sum_j p_j (1 - p_j) ) / ( (K/d) * (1 - K/d) )
    where p_j is the selection frequency of feature j across runs and K is the
    mean number of selected features.  Returns NaN when undefined
    (K in {0, d}).
    """
    if not selections or d <= 0:
        return float("nan")
    m = len(selections)
    counts = np.zeros(d, dtype=float)
    sizes = []
    for s in selections:
        idx = np.asarray(list(s), dtype=int)
        idx = idx[(idx >= 0) & (idx < d)]
        counts[np.unique(idx)] += 1
        sizes.append(len(np.unique(idx)))
    p = counts / m
    K = float(np.mean(sizes))
    kd = K / d
    if kd <= 0 or kd >= 1:
        return float("nan")
    denom = kd * (1 - kd)
    numer = (1.0 / d) * float(np.sum(p * (1 - p)))
    return float(1.0 - numer / denom)


def pairwise_summary(selections: Sequence[Sequence[int]], d: int) -> Dict[str, float]:
    rows = list(combinations(range(len(selections)), 2))
    if not rows:
        return {"n_pairs": 0, "jaccard_mean": float("nan"), "jaccard_std": float("nan"),
                "kuncheva_mean": float("nan"), "kuncheva_std": float("nan"),
                "nogueira": nogueira_stability(selections, d)}
    jac = [jaccard(selections[i], selections[j]) for i, j in rows]
    kun = [kuncheva(selections[i], selections[j], d) for i, j in rows]
    return {
        "n_pairs": len(rows),
        "jaccard_mean": float(np.mean(jac)),
        "jaccard_std": float(np.std(jac, ddof=1)) if len(jac) > 1 else 0.0,
        "kuncheva_mean": float(np.nanmean(kun)) if len(kun) else float("nan"),
        "kuncheva_std": float(np.nanstd(kun, ddof=1)) if len(kun) > 1 else 0.0,
        "nogueira": nogueira_stability(selections, d),
    }


def consensus_select(selections: Sequence[Sequence[int]], m: int,
                     d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Training-only consensus: top-m features by selection frequency.

    Returns (selected_indices, frequencies).  Ties broken by smaller index.
    """
    counts = np.zeros(d, dtype=float)
    for s in selections:
        idx = np.unique(np.asarray(list(s), dtype=int))
        idx = idx[(idx >= 0) & (idx < d)]
        counts[idx] += 1.0
    freq = counts / max(1, len(selections))
    order = np.lexsort((np.arange(d), -freq))  # sort by -freq, then index
    selected = order[: int(m)]
    return selected.astype(int), freq
