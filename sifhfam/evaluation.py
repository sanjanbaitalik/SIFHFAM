"""Downstream evaluation with fixed classifiers (prompt section 10).

Feature selection is performed ONCE on the training split; the same selected
features are then evaluated with several fixed downstream classifiers.
No classifier hyperparameter is tuned per dataset or per selector.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


CLASSIFIER_NAMES = ("RF", "SVM", "KNN", "LR")


def build_classifier(name: str, seed: int) -> Pipeline:
    """Fixed, globally chosen configurations (no per-dataset tuning)."""
    if name == "RF":
        return Pipeline([("clf", RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=1))])
    if name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=1.0, kernel="rbf", random_state=seed)),
        ])
    if name == "KNN":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ])
    if name == "LR":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        ])
    raise ValueError(f"Unknown classifier: {name}")


@dataclass
class EvalResult:
    classifier: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    eval_seconds: float
    n_selected: int
    error: Optional[str] = None


def evaluate_selected(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    selected: Sequence[int],
    classifier: str = "RF",
    seed: int = 42,
) -> EvalResult:
    idx = np.asarray(list(selected), dtype=int)
    n = int(idx.size)
    if n == 0:
        return EvalResult(classifier, 0.0, 0.0, 0.0, 0.0, 0)
    try:
        clf = build_classifier(classifier, seed)
        t0 = time.perf_counter()
        clf.fit(X_train[:, idx], y_train)
        pred = clf.predict(X_test[:, idx])
        dt = time.perf_counter() - t0
        return EvalResult(
            classifier=classifier,
            accuracy=float(accuracy_score(y_test, pred)),
            balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
            macro_f1=float(f1_score(y_test, pred, average="macro", zero_division=0)),
            eval_seconds=float(dt),
            n_selected=n,
        )
    except Exception as exc:
        return EvalResult(classifier, float("nan"), float("nan"), float("nan"),
                          0.0, n, error=f"{type(exc).__name__}: {exc}")


def evaluate_all_classifiers(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    selected: Sequence[int],
    classifiers: Sequence[str] = CLASSIFIER_NAMES,
    seed: int = 42,
) -> List[EvalResult]:
    return [evaluate_selected(X_train, y_train, X_test, y_test, selected,
                              classifier=c, seed=seed) for c in classifiers]
