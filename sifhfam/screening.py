"""Training-only screening strategies (prompt sections 5 and 12).

Every strategy fits on the training fold only.  The default canonical
strategy is univariate information-theoretic relevance (discretized NMI),
consistent with the manuscript's information-theoretic description; the
legacy ANOVA-F shortcut is available only as an explicit non-default option
and is labelled as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .discretization import Discretizer
from .information import nmi


@dataclass
class ScreeningResult:
    strategy: str
    scores: np.ndarray            # full-d vector (0 for unselected when applicable)
    screened_idx: np.ndarray      # ascending original indices
    fit_seconds: float
    sort_seconds: float
    notes: str = ""


def _fit_scores(strategy: str, X: np.ndarray, y: np.ndarray, bins: int,
                discretizer: str, seed: int) -> Tuple[np.ndarray, str]:
    d = X.shape[1]
    if strategy == "mi":
        disc = Discretizer(method=discretizer, bins=bins).fit(X)
        Xd = disc.transform(X)
        from .information import feature_nmi_scores
        s = feature_nmi_scores(Xd, y)
        return s, "univariate discretized NMI(X_j; y), train-only"
    if strategy == "anova_f":
        from sklearn.feature_selection import f_classif
        try:
            s, _ = f_classif(X, y)
        except Exception:
            s = np.var(X, axis=0)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        return s, "LEGACY-COMPATIBLE ANOVA-F (not information-theoretic)"
    if strategy == "relieff":
        s = _relieff_scores(X, y, seed)
        return s, "ReliefF (skrebate) importances as screening scores"
    if strategy == "none":
        return np.ones(d, dtype=float), "no screening (f = d)"
    raise ValueError(f"Unknown screening strategy: {strategy}")


def _relieff_scores(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    import sys
    from pathlib import Path
    bundle = Path(__file__).resolve().parents[2] / "FHFAM bundle" / "ReliefF" / "scikit-rebate"
    if str(bundle) not in sys.path:
        sys.path.insert(0, str(bundle))
    from skrebate import ReliefF
    n = X.shape[0]
    k = int(min(10, max(2, n // 10)))
    clf = ReliefF(n_features_to_select=min(X.shape[1], 10), n_neighbors=k,
                  n_jobs=1, verbose=False)
    clf.fit(X, y)
    return np.nan_to_num(np.asarray(clf.feature_importances_, dtype=float),
                         nan=0.0, posinf=0.0, neginf=0.0)


def interaction_union_scores(X: np.ndarray, y: np.ndarray, f_budget: int, bins: int,
                             discretizer: str, seed: int,
                             union_share: float = 0.5) -> np.ndarray:
    """Ablation: deterministic union of top univariate NMI and top ReliefF.

    Produces scores whose top-f set equals the union of the top-(share*f) MI
    features and top-((1-share)*f) ReliefF features (ties broken by index).
    Not the default.
    """
    d = X.shape[1]
    disc = Discretizer(method=discretizer, bins=bins).fit(X)
    Xd = disc.transform(X)
    mi_scores = np.array([nmi(Xd[:, j], y) for j in range(d)], dtype=float)
    rf_scores = _relieff_scores(X, y, seed)
    n_mi = max(1, int(round(union_share * f_budget)))
    n_rf = max(1, f_budget - n_mi)
    mi_top = set(int(i) for i in np.argsort(-mi_scores, kind="stable")[:n_mi])
    rf_order = np.argsort(-rf_scores, kind="stable")
    rf_top = set()
    for i in rf_order:
        if int(i) not in mi_top or len(rf_top) < n_rf:
            if int(i) not in rf_top and (int(i) not in mi_top or len(rf_top) < n_rf):
                rf_top.add(int(i))
        if len(mi_top) + len(rf_top) >= f_budget and len(rf_top) >= n_rf:
            break
    union = mi_top | rf_top
    # combined score: rank-based so both lists contribute; unselected get 0
    out = np.zeros(d, dtype=float)
    mi_rank = {i: r for r, i in enumerate(np.argsort(-mi_scores, kind="stable"))}
    rf_rank = {int(i): r for r, i in enumerate(rf_order)}
    for j in range(d):
        if j in union:
            out[j] = 2.0 * d - min(mi_rank.get(j, d), rf_rank.get(j, d))
    return out


def screen(X: np.ndarray, y: np.ndarray, f_budget: int, strategy: str = "mi",
           bins: int = 10, discretizer: str = "quantile", seed: int = 42,
           union_share: float = 0.5) -> ScreeningResult:
    import time
    d = X.shape[1]
    if strategy == "none":
        f_budget = d
    f_budget = int(max(1, min(f_budget, d)))

    t0 = time.perf_counter()
    if strategy == "interaction_union":
        scores = interaction_union_scores(X, y, f_budget, bins, discretizer, seed, union_share)
        notes = "union of top univariate NMI and top ReliefF (ablation, non-default)"
    else:
        scores, notes = _fit_scores(strategy, X, y, bins, discretizer, seed)
    t1 = time.perf_counter()
    if strategy == "none":
        idx = np.arange(d, dtype=int)
    else:
        order = np.argsort(-scores, kind="stable")
        idx = np.sort(order[:f_budget].astype(int))
    t2 = time.perf_counter()
    full_scores = np.zeros(d, dtype=float)
    full_scores[idx] = scores[idx]
    return ScreeningResult(
        strategy=strategy,
        scores=full_scores,
        screened_idx=idx,
        fit_seconds=float(t1 - t0),
        sort_seconds=float(t2 - t1),
        notes=notes,
    )
