"""Dataset loading and leakage-safe split protocol (prompt: split indices saved)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_dir(data_dir: Optional[str | Path] = None) -> Path:
    if data_dir is None:
        return project_root() / "Updated Dataset"
    return Path(data_dir).expanduser().resolve()


def load_dataset(stem: str, data_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    data_dir = data_dir or resolve_data_dir()
    X_path = data_dir / f"{stem}_X.csv"
    y_path = data_dir / f"{stem}_Y.csv"
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing files for dataset '{stem}': {X_path}, {y_path}. "
            "Datasets are external inputs: provide headerless CSV pairs "
            "<data_dir>/<name>_X.csv (features) and <data_dir>/<name>_Y.csv "
            "(labels). The default <data_dir> is '<project_root>/Updated Dataset'; "
            "pass an explicit data_dir to load_dataset(...) to override it. "
            "Without these files the dataset-dependent tests and experiments "
            "cannot run (self-contained tests are unaffected)."
        )
    X_raw = pd.read_csv(X_path, header=None).to_numpy(dtype=float)
    y_raw = pd.read_csv(y_path, header=None).to_numpy().ravel()

    _, y_codes = np.unique(y_raw, return_inverse=True)
    vals, counts = np.unique(y_codes, return_counts=True)
    keep = vals[counts >= 2]
    mask = np.isin(y_codes, keep)
    removed = int((~mask).sum())
    X = X_raw[mask]
    y = y_codes[mask]
    _, y = np.unique(y, return_inverse=True)
    meta = {
        "dataset": stem,
        "original_n_samples": int(X_raw.shape[0]),
        "effective_n_samples": int(X.shape[0]),
        "n_total_features": int(X.shape[1]),
        "removed_singleton_label_rows": removed,
        "original_label_counts": {str(k): int(v) for k, v in zip(*np.unique(y_raw, return_counts=True))},
        "effective_label_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }
    return X, y.astype(int), meta


@dataclass
class Split:
    dataset: str
    run: int
    seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset,
            "run": self.run,
            "seed": self.seed,
            "train_idx": self.train_idx.astype(int).tolist(),
            "test_idx": self.test_idx.astype(int).tolist(),
        }


def make_split(dataset: str, n_samples: int, y: np.ndarray, run: int,
               test_size: float, seed: int) -> Split:
    """Deterministic stratified split.  Identical seed => identical indices."""
    idx = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y,
    )
    return Split(dataset=dataset, run=run, seed=seed,
                 train_idx=np.sort(train_idx), test_idx=np.sort(test_idx))


def run_seed(base_seed: int, dataset: str, run: int) -> int:
    """Stable per-(dataset, run) seed derived from the base seed."""
    # Python's hash() is salted per-process; use a fixed string hash instead.
    h = 0
    for ch in f"{base_seed}|{dataset}|{run}":
        h = (h * 131 + ord(ch)) % (2 ** 31 - 1)
    return int(h)


def save_splits(path: Path, splits: List[Split]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [s.to_dict() for s in splits]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def load_splits(path: Path) -> List[Split]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [
        Split(dataset=p["dataset"], run=int(p["run"]), seed=int(p["seed"]),
              train_idx=np.asarray(p["train_idx"], dtype=int),
              test_idx=np.asarray(p["test_idx"], dtype=int))
        for p in payload
    ]
