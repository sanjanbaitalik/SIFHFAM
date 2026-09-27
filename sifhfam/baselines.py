"""Authentic baseline registry and wrappers (prompt section 9).

Integrity rules enforced here:
- No proxy implementation may be exported under a published method name.
- Methods without an authentic local implementation are raised as
  ``UnavailableBaseline`` and recorded UNAVAILABLE/NOT VERIFIED.
- Every runnable baseline is tied to a registry row with exact source path.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .discretization import Discretizer

BUNDLE_ROOT = Path(__file__).resolve().parents[2] / "FHFAM bundle"


class UnavailableBaseline(RuntimeError):
    """Raised when a baseline cannot be run authentically in this workspace."""


PROXY_BANNED = {
    "HIFS", "UDFS", "NDFS", "HSIC-Lasso",
    "hifs", "udfs", "ndfs", "hsic_lasso",
    "HIFS_proxy", "UDFS_proxy", "NDFS_proxy", "ReliefF_proxy", "HSIC-Lasso_proxy",
}


@dataclass
class BaselineOutcome:
    method: str
    status: str                      # OK | UNAVAILABLE | FAILED
    ranking: Optional[np.ndarray]    # full/partial ranking (original indices) or None
    native_subset: Optional[np.ndarray]
    n_selected: int
    seconds: float
    notes: str = ""
    error: Optional[str] = None
    registry_key: Optional[str] = None


# ------------------------------- registry -------------------------------- #

def _git_commit(path: Path) -> str:
    g = path / ".git"
    if not g.exists():
        return ""
    try:
        import subprocess
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def baseline_registry() -> List[Dict[str, object]]:
    """Mandatory baseline manifest rows (prompt section 9 columns)."""
    rows: List[Dict[str, object]] = []

    def add(**kw):
        rows.append(kw)

    add(**{
        "method_name": "mRMR",
        "status": "authentic",
        "exact_source_path": str(BUNDLE_ROOT / "CMIM/scikit-feature/skfeature/function/information_theory_based/MRMR.py"),
        "package_repository_name": "scikit-feature (skfeature)",
        "version_or_commit": _git_commit(BUNDLE_ROOT / "CMIM/scikit-feature"),
        "supervised_unsupervised": "supervised",
        "output_type": "ranking",
        "parameters_used": "discretized (train-only quantile, cfg.bins); n_selected_features = max budget; if d > 1000: MI-prefilter pool = top 1000 univariate-MI features (documented computational protocol), authentic skfeature mRMR inside pool",
        "random_seed_behavior": "deterministic (no RNG)",
        "preprocessing_required": "discrete input (train-only quantile discretization)",
        "task_compatibility": "single-label classification: compatible",
        "primary_paper_metadata_found_locally": "Brown et al., Conditional Likelihood Maximisation, JMLR 2012 (LCSI/MRMR path); Peng et al. mRMR criterion family",
        "notes": "MI-based mRMR via LCSI gamma=0 function_name='MRMR'. The pypi `mrmr` package in the bundle defaults to F-stat relevance + correlation redundancy and is NOT used under the mRMR name.",
    })
    add(**{
        "method_name": "ReliefF",
        "status": "authentic",
        "exact_source_path": str(BUNDLE_ROOT / "ReliefF/scikit-rebate/skrebate/relieff.py"),
        "package_repository_name": "scikit-rebate (skrebate)",
        "version_or_commit": _git_commit(BUNDLE_ROOT / "ReliefF/scikit-rebate"),
        "supervised_unsupervised": "supervised",
        "output_type": "ranking",
        "parameters_used": "n_features_to_select = max budget; n_neighbors = min(10, max(2, n//10)); n_jobs=1",
        "random_seed_behavior": "deterministic given data (all training instances used)",
        "preprocessing_required": "none (continuous features OK)",
        "task_compatibility": "single-label classification: compatible",
        "primary_paper_metadata_found_locally": "Kira, Rendell etc. ReliefF as implemented in Urbanowicz/Olson scikit-rebate (MIT license, cited in repo README)",
        "notes": "Replaces the legacy ANOVA-F proxy that was previously exported as 'ReliefF'.",
    })
    add(**{
        "method_name": "CMIM",
        "status": "authentic",
        "exact_source_path": str(BUNDLE_ROOT / "CMIM/scikit-feature/skfeature/function/information_theoretical_based/CMIM.py"),
        "package_repository_name": "scikit-feature (skfeature)",
        "version_or_commit": _git_commit(BUNDLE_ROOT / "CMIM/scikit-feature"),
        "supervised_unsupervised": "supervised",
        "output_type": "ranking",
        "parameters_used": "discrete input; n_selected_features = max budget; if d > 1000: MI-prefilter pool = top 1000 univariate-MI features (documented computational protocol), authentic skfeature CMIM inside pool",
        "random_seed_behavior": "deterministic (no RNG)",
        "preprocessing_required": "discrete input (train-only quantile discretization)",
        "task_compatibility": "single-label classification: compatible",
        "primary_paper_metadata_found_locally": "Brown et al., Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection, JMLR 2012 (docstring)",
        "notes": "",
    })
    add(**{
        "method_name": "JMIM",
        "status": "authentic",
        "exact_source_path": str(BUNDLE_ROOT / "JMIM/mifs/mifs/mifs.py"),
        "package_repository_name": "mifs (MutualInformationFeatureSelector)",
        "version_or_commit": _git_commit(BUNDLE_ROOT / "JMIM"),
        "supervised_unsupervised": "supervised",
        "output_type": "ranking",
        "parameters_used": "method='JMIM', k=min(5, smallest_train_class-1), n_features = max budget, categorical=True, n_jobs=1; if d > 200: candidate pool = top 200 univariate-MI features (documented kNN-MI computational protocol)",
        "random_seed_behavior": "deterministic given data (kNN MI, no RNG)",
        "preprocessing_required": "none (continuous features OK; kNN MI)",
        "task_compatibility": "single-label classification: compatible",
        "primary_paper_metadata_found_locally": "docstring: JMI/JMIM/MRMR references [1][2][3] inside mifs.py",
        "notes": "Bundle folder is named JMIM; the authentic JMIM implementation is method='JMIM' inside the mifs package.",
    })
    add(**{
        "method_name": "FCBF",
        "status": "authentic",
        "exact_source_path": str(BUNDLE_ROOT / "FCFB/FCBF_module/FCBF_module.py"),
        "package_repository_name": "FCBF_module (Senliol/Gulgezen variant; Yu & Liu ICML 2003 algorithm)",
        "version_or_commit": _git_commit(BUNDLE_ROOT / "FCFB"),
        "supervised_unsupervised": "supervised",
        "output_type": "subset",
        "parameters_used": "th = 0.01 (default); discrete input",
        "random_seed_behavior": "deterministic (no RNG)",
        "preprocessing_required": "discrete input (train-only quantile discretization)",
        "task_compatibility": "single-label classification: compatible",
        "primary_paper_metadata_found_locally": "Yu & Liu ICML 2003 citation in module docstring; Senliol et al. ISCIS 2008 variant note",
        "notes": "Intrinsic output is a subset; reported in the native-cardinality table (no fabricated ranking).",
    })
    add(**{
        "method_name": "FRFS",
        "status": "authentic",
        "exact_source_path": str(BUNDLE_ROOT / "FRFS/fuzzy-rough-learn/frlearn/neighbours/feature_preprocessors.py"),
        "package_repository_name": "fuzzy-rough-learn (frlearn)",
        "version_or_commit": _git_commit(BUNDLE_ROOT / "FRFS/fuzzy-rough-learn"),
        "supervised_unsupervised": "supervised",
        "output_type": "subset",
        "parameters_used": "FRFS() defaults; greedy positive-region growth (Cornelis/Verbiest/Jensen OWA-FRFS)",
        "random_seed_behavior": "deterministic (no RNG)",
        "preprocessing_required": "internal RangeStandardiser",
        "task_compatibility": "single-label classification: compatible",
        "primary_paper_metadata_found_locally": "Cornelis C., Verbiest N., Jensen R., OWA Based Fuzzy Rough Sets, RSKT 2010 (docstring); package 2022-era fuzzy-rough-learn",
        "notes": "Recent fuzzy-rough comparator requested by reviewers; native subset.",
    })
    add(**{
        "method_name": "PPFS",
        "status": "authentic",
        "exact_source_path": str(BUNDLE_ROOT / "PPFS/PyImpetus/PyImpetus.py"),
        "package_repository_name": "PyImpetus 4.1.1 (PPIMBC)",
        "version_or_commit": _git_commit(BUNDLE_ROOT / "PPFS"),
        "supervised_unsupervised": "supervised",
        "output_type": "subset",
        "parameters_used": "PPIMBC(cv=0, num_simul=5, simul_type=0, simul_size=0.2, sig_test_type='non-parametric', random_state=<seed>, p_val_thresh=0.05, verbose=0)",
        "random_seed_behavior": "random_state parameter exposed; fixed per run",
        "preprocessing_required": "pandas DataFrame input; no scaling required",
        "task_compatibility": "single-label classification: compatible (Markov blanket selection)",
        "primary_paper_metadata_found_locally": "PyImpetus README/package: PPIMBC (pattern-recognition-era Markov blanket FS)",
        "notes": "May return an empty subset when no feature passes the significance threshold; reported honestly.",
    })
    add(**{
        "method_name": "QuickSelection",
        "status": "authentic_local_source",
        "exact_source_path": str(BUNDLE_ROOT / "QuickSelection/repo/QuickSelection/Sparse_DAE.py"),
        "package_repository_name": "QuickSelection (Atashgahi et al., arXiv:2012.00560 / ML journal-track family)",
        "version_or_commit": _git_commit(BUNDLE_ROOT / "QuickSelection"),
        "supervised_unsupervised": "unsupervised feature scoring via sparse AE (label-free encoder), graph ranking",
        "output_type": "ranking",
        "parameters_used": "Sparse_DAE((d, 1000, d), (Sigmoid, Linear|tanh), epsilon=13); epochs per profile; lr=0.01, momentum=0.9, weight_decay=1e-5, zeta=0.2, dropout=0.2, batch=100, noise=0.2; fs_utils.feature_selection method='Node Strength(in)'",
        "random_seed_behavior": "global numpy seed set per run (Erdos-Renyi mask + SET evolution use global RNG)",
        "preprocessing_required": "continuous features; train fold only",
        "task_compatibility": "single-label classification: compatible for ranking extraction",
        "primary_paper_metadata_found_locally": "QuickSelection.py header citation: Atashgahi et al. 2020 arXiv:2012.00560",
        "notes": "Driver scripts use module-level argparse; we call Sparse_DAE + fs_utils directly, mirroring the driver's W1/W2 usage. Two documented runtime adaptations: batch_size = min(100, n_train) (driver's 100 yields zero batches for n<100), and a numpy>=2 behavior-identical alias np.in1d = raveling np.isin required by the bundled Sparse_DAE.",
    })
    add(**{
        "method_name": "ATR",
        "status": "excluded_task_incompatible",
        "exact_source_path": str(BUNDLE_ROOT / "ATR/ATR_code/classes/atr.py"),
        "package_repository_name": "ATR (Adaptive and Transformed Relevance)",
        "version_or_commit": _git_commit(BUNDLE_ROOT / "ATR"),
        "supervised_unsupervised": "supervised",
        "output_type": "subset",
        "parameters_used": "n/a - not run",
        "random_seed_behavior": "n/a",
        "preprocessing_required": "n/a",
        "task_compatibility": "INCOMPATIBLE: multi-label feature selection (README: 'Multi-Label Feature Selection Using Adaptive and Transformed Relevance') vs this paper's single-label setting",
        "primary_paper_metadata_found_locally": "ATR_code/README.md title",
        "notes": "Not used. No compatible single-label mode is asserted without evidence.",
    })
    for name, why in [
        ("HIFS", "No authentic local implementation found in the workspace bundle."),
        ("UDFS", "No authentic local implementation found (only explicit proxies in legacy code)."),
        ("NDFS", "No authentic local implementation found (only explicit proxies in legacy code)."),
        ("HSIC-Lasso", "No authentic local implementation found (only explicit proxies in legacy code)."),
    ]:
        add(**{
            "method_name": name,
            "status": "UNAVAILABLE/NOT VERIFIED",
            "exact_source_path": "",
            "package_repository_name": "",
            "version_or_commit": "",
            "supervised_unsupervised": "n/a",
            "output_type": "n/a",
            "parameters_used": "n/a - not run",
            "random_seed_behavior": "n/a",
            "preprocessing_required": "n/a",
            "task_compatibility": "unknown without authentic implementation",
            "primary_paper_metadata_found_locally": "none",
            "notes": why + " Legacy numbers under this name are frozen artifacts only and are NOT verified for revised evidence.",
        })
    add(**{
        "method_name": "mRMR_pypi_package",
        "status": "available_but_not_published_mrmr",
        "exact_source_path": str(BUNDLE_ROOT / "mRMR/mrmr/mrmr/pandas.py"),
        "package_repository_name": "mrmr (pypi-style package)",
        "version_or_commit": _git_commit(BUNDLE_ROOT / "mRMR"),
        "supervised_unsupervised": "supervised",
        "output_type": "ranking",
        "parameters_used": "n/a - not used under the mRMR name",
        "random_seed_behavior": "n/a",
        "preprocessing_required": "n/a",
        "task_compatibility": "compatible, but relevance default is F-statistic, not MI",
        "primary_paper_metadata_found_locally": "setup.py / README",
        "notes": "Using this package under the published name 'mRMR' would misrepresent Peng's MI-based mRMR; excluded from the mRMR label.",
    })
    return rows


# ------------------------------ loaders ---------------------------------- #

def _ensure_path(p: Path) -> None:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise UnavailableBaseline(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------ runners ---------------------------------- #

def _mi_prefilter_pool(X: np.ndarray, y: np.ndarray, pool: int) -> np.ndarray:
    """Top-univariate-MI candidate pool for high-d inputs of discrete-MI
    baselines (documented computational protocol; see baseline manifest)."""
    d = X.shape[1]
    if d <= pool:
        return np.arange(d, dtype=int)
    from .discretization import Discretizer
    from .information import feature_nmi_scores
    Xd = Discretizer(method="quantile", bins=10).fit_transform(X)
    mi = feature_nmi_scores(Xd, y)
    order = np.argsort(-mi, kind="stable")
    return np.sort(order[:pool].astype(int))


def _select_mrmr(X: np.ndarray, y: np.ndarray, max_budget: int, seed: int,
                 pool: int = 1000) -> BaselineOutcome:
    t0 = time.perf_counter()
    try:
        _ensure_path(BUNDLE_ROOT / "CMIM/scikit-feature")
        from skfeature.function.information_theoretical_based.MRMR import mrmr as sk_mrmr
        pool_idx = _mi_prefilter_pool(X, y, pool)
        Xd = Discretizer(method="quantile", bins=10).fit_transform(X[:, pool_idx])
        F, _, _ = sk_mrmr(Xd, y, n_selected_features=int(min(max_budget, len(pool_idx))))
        lead = pool_idx[np.asarray(F, dtype=int)]
        if len(lead) < X.shape[1]:
            tail = np.setdiff1d(np.arange(X.shape[1], dtype=int), lead, assume_unique=False)
            ranking = np.concatenate([lead, tail]).astype(int)
        else:
            ranking = lead.astype(int)
        note = ("full-space" if X.shape[1] <= pool else
                f"MI-prefilter pool={pool} of d={X.shape[1]} (top univariate MI); "
                "skfeature mRMR authentic inside pool")
        return BaselineOutcome("mRMR", "OK", ranking, ranking[:max_budget],
                               min(max_budget, len(ranking)),
                               time.perf_counter() - t0, notes=note,
                               registry_key="mRMR")
    except Exception as exc:
        return BaselineOutcome("mRMR", "FAILED", None, None, 0,
                               time.perf_counter() - t0,
                               error=f"{type(exc).__name__}: {exc}",
                               registry_key="mRMR")


def _select_relieff(X: np.ndarray, y: np.ndarray, max_budget: int, seed: int) -> BaselineOutcome:
    t0 = time.perf_counter()
    try:
        _ensure_path(BUNDLE_ROOT / "ReliefF/scikit-rebate")
        from skrebate import ReliefF
        n = X.shape[0]
        k = int(min(10, max(2, n // 10)))
        clf = ReliefF(n_features_to_select=int(min(max_budget, X.shape[1])),
                      n_neighbors=k, n_jobs=1, verbose=False)
        clf.fit(X, y)
        imp = np.nan_to_num(np.asarray(clf.feature_importances_, dtype=float),
                            nan=0.0, posinf=0.0, neginf=0.0)
        ranking = np.argsort(-imp, kind="stable").astype(int)
        return BaselineOutcome("ReliefF", "OK", ranking, ranking[:max_budget],
                               int(min(max_budget, len(ranking))),
                               time.perf_counter() - t0, registry_key="ReliefF")
    except Exception as exc:
        return BaselineOutcome("ReliefF", "FAILED", None, None, 0,
                               time.perf_counter() - t0,
                               error=f"{type(exc).__name__}: {exc}", registry_key="ReliefF")


def _select_cmim(X: np.ndarray, y: np.ndarray, max_budget: int, seed: int,
                 pool: int = 1000) -> BaselineOutcome:
    t0 = time.perf_counter()
    try:
        _ensure_path(BUNDLE_ROOT / "CMIM/scikit-feature")
        from skfeature.function.information_theoretical_based.CMIM import cmim
        pool_idx = _mi_prefilter_pool(X, y, pool)
        Xd = Discretizer(method="quantile", bins=10).fit_transform(X[:, pool_idx])
        F, _, _ = cmim(Xd, y, n_selected_features=int(min(max_budget, len(pool_idx))))
        lead = pool_idx[np.asarray(F, dtype=int)]
        if len(lead) < X.shape[1]:
            tail = np.setdiff1d(np.arange(X.shape[1], dtype=int), lead, assume_unique=False)
            ranking = np.concatenate([lead, tail]).astype(int)
        else:
            ranking = lead.astype(int)
        note = ("full-space" if X.shape[1] <= pool else
                f"MI-prefilter pool={pool} of d={X.shape[1]} (top univariate MI); "
                "skfeature CMIM authentic inside pool")
        return BaselineOutcome("CMIM", "OK", ranking, ranking[:max_budget],
                               min(max_budget, len(ranking)),
                               time.perf_counter() - t0, notes=note,
                               registry_key="CMIM")
    except Exception as exc:
        return BaselineOutcome("CMIM", "FAILED", None, None, 0,
                               time.perf_counter() - t0,
                               error=f"{type(exc).__name__}: {exc}",
                               registry_key="CMIM")


def _select_jmim(X: np.ndarray, y: np.ndarray, max_budget: int, seed: int,
                 pool: int = 200) -> BaselineOutcome:
    t0 = time.perf_counter()
    try:
        _ensure_path(BUNDLE_ROOT / "JMIM/mifs")
        import mifs
        X = np.asarray(X, dtype=np.float64)  # _entropy uses np.finfo -> must be float
        y = np.asarray(y).ravel()
        # Constant/near-constant columns make the kNN MI estimator NaN;
        # exclude them explicitly.  High-d inputs are restricted to a
        # top-univariate-MI candidate pool (documented computational protocol;
        # kNN-MI over all d features is computationally infeasible).
        nunique = np.array([len(np.unique(X[:, j])) for j in range(X.shape[1])])
        active = np.where(nunique >= 2)[0]
        if active.size == 0:
            ranking = np.arange(X.shape[1], dtype=int)
            return BaselineOutcome("JMIM", "OK", ranking, ranking[:max_budget],
                                   min(max_budget, len(ranking)),
                                   time.perf_counter() - t0,
                                   notes="all columns constant", registry_key="JMIM")
        if active.size > pool:
            pool_idx_local = _mi_prefilter_pool(X[:, active], y, pool)
            candidate = active[pool_idx_local]
        else:
            candidate = active
        counts = np.bincount(y)
        min_class = int(counts[counts > 0].min()) if counts.size else 0
        k = int(min(5, min_class - 1))
        if k < 2:
            return BaselineOutcome(
                "JMIM", "FAILED", None, None, 0, time.perf_counter() - t0,
                error=(f"kNN MI estimator requires smallest train class >= 3 "
                       f"(smallest={min_class}); cannot run authentic mifs JMIM"),
                registry_key="JMIM")
        sel = mifs.MutualInformationFeatureSelector(
            method="JMIM", k=k, n_features=int(min(max_budget, candidate.size)),
            categorical=True, n_jobs=1, verbose=0)
        sel.fit(X[:, candidate], y)
        sub_rank = np.asarray(sel.ranking_, dtype=int)
        lead = candidate[sub_rank]
        tail = np.setdiff1d(np.arange(X.shape[1], dtype=int), lead, assume_unique=False)
        ranking = np.concatenate([lead, tail]).astype(int)
        note = (f"k={k} (default 5 capped by smallest class {min_class}); "
                f"candidate pool={len(candidate)} of d={X.shape[1]} "
                f"(top univariate MI; kNN-MI pool protocol)")
        if active.size < X.shape[1]:
            note += f"; excluded {int((nunique < 2).sum())} constant columns"
        return BaselineOutcome("JMIM", "OK", ranking, ranking[:max_budget],
                               min(max_budget, len(ranking)), time.perf_counter() - t0,
                               notes=note, registry_key="JMIM")
    except Exception as exc:
        med_uniq = float(np.median([len(np.unique(X[:, j]))
                                    for j in range(min(200, X.shape[1]))]))
        note = (f"median unique values/feature (first 200) = {med_uniq:.0f}; "
                "kNN MI (Ross/Battista) requires effectively continuous features")
        return BaselineOutcome(
            "JMIM", "FAILED", None, None, 0, time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc} | {note}",
            registry_key="JMIM")


def _select_fcbf(X: np.ndarray, y: np.ndarray, max_budget: int, seed: int) -> BaselineOutcome:
    t0 = time.perf_counter()
    try:
        path = BUNDLE_ROOT / "FCFB/FCBF_module/FCBF_module.py"
        mod = _load_module_from_file("_neucom_fcbf_module", path)
        Xd = Discretizer(method="quantile", bins=10).fit_transform(X).astype(float)
        f = mod.FCBF(th=0.01)
        f.fit(Xd, y)
        subset = np.asarray(f.idx_sel, dtype=int)
        return BaselineOutcome("FCBF", "OK", None, subset, len(subset),
                               time.perf_counter() - t0,
                               notes="native subset; no ranking asserted",
                               registry_key="FCBF")
    except Exception as exc:
        return BaselineOutcome("FCBF", "FAILED", None, None, 0,
                               time.perf_counter() - t0,
                               error=f"{type(exc).__name__}: {exc}", registry_key="FCBF")


def _select_frfs(X: np.ndarray, y: np.ndarray, max_budget: int, seed: int) -> BaselineOutcome:
    t0 = time.perf_counter()
    try:
        n, d = X.shape
        # frlearn FRFS materializes an (n, n, d) similarity tensor (float64).
        est_bytes = n * n * d * 8
        limit_bytes = 4 * (1024 ** 3)
        if est_bytes > limit_bytes:
            return BaselineOutcome(
                "FRFS", "FAILED", None, None, 0, time.perf_counter() - t0,
                error=(f"infeasible on this host: FRFS allocates ~{est_bytes/1e9:.1f} GB "
                       f"(n={n}, d={d}); limit {limit_bytes/1e9:.1f} GB"),
                registry_key="FRFS")
        _ensure_path(BUNDLE_ROOT / "FRFS/fuzzy-rough-learn")
        from frlearn.neighbours.feature_preprocessors import FRFS
        model = FRFS()(X, y)
        mask = np.asarray(model.selection, dtype=bool)
        subset = np.where(mask)[0].astype(int)
        return BaselineOutcome("FRFS", "OK", None, subset, int(subset.size),
                               time.perf_counter() - t0,
                               notes="native subset (positive-region greedy)",
                               registry_key="FRFS")
    except MemoryError as exc:
        return BaselineOutcome("FRFS", "FAILED", None, None, 0, time.perf_counter() - t0,
                               error=f"MemoryError: {exc}", registry_key="FRFS")
    except Exception as exc:
        return BaselineOutcome("FRFS", "FAILED", None, None, 0,
                               time.perf_counter() - t0,
                               error=f"{type(exc).__name__}: {exc}", registry_key="FRFS")


def _select_ppfs(X: np.ndarray, y: np.ndarray, max_budget: int, seed: int) -> BaselineOutcome:
    t0 = time.perf_counter()
    try:
        _ensure_path(BUNDLE_ROOT / "PPFS")
        import pandas as pd
        from PyImpetus import PPIMBC
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        model = PPIMBC(cv=0, num_simul=5, simul_type=0, simul_size=0.2,
                       sig_test_type="non-parametric", random_state=int(seed) % (2**31),
                       verbose=0, p_val_thresh=0.05, n_jobs=1)
        model.fit(df, y)
        subset = None
        for attr in ("MB", "selected_feat_indices_", "selected_features_"):
            val = getattr(model, attr, None)
            if val is None:
                continue
            if hasattr(val, "columns"):  # DataFrame of selected columns
                cols = [str(c) for c in val.columns]
                subset = np.array([int(c[1:]) if c.startswith("f") else int(c)
                                   for c in cols], dtype=int)
                break
            arr = np.asarray(val)
            if arr.dtype == bool:
                subset = np.where(arr)[0].astype(int)
                break
            if arr.dtype.kind in "USO":  # string column names
                subset = np.array([int(str(c)[1:]) if str(c).startswith("f") else int(str(c))
                                   for c in arr.ravel()], dtype=int)
                break
            subset = arr.astype(int).ravel()
            break
        if subset is None:
            # fall back to transformed columns
            Xt = model.transform(df)
            subset = np.asarray([int(c.split("f")[1]) for c in Xt.columns], dtype=int)
        return BaselineOutcome("PPFS", "OK", None, subset, int(subset.size),
                               time.perf_counter() - t0,
                               notes="native Markov-blanket subset (may be empty)",
                               registry_key="PPFS")
    except Exception as exc:
        return BaselineOutcome("PPFS", "FAILED", None, None, 0,
                               time.perf_counter() - t0,
                               error=f"{type(exc).__name__}: {exc}", registry_key="PPFS")


def _select_quickselection(X: np.ndarray, y: np.ndarray, max_budget: int, seed: int,
                           epochs: int = 3, workdir: Optional[Path] = None) -> BaselineOutcome:
    t0 = time.perf_counter()
    try:
        import numpy as _np
        import scipy.sparse
        qs_dir = BUNDLE_ROOT / "QuickSelection/repo/QuickSelection"
        _ensure_path(qs_dir)
        from Sparse_DAE import Sparse_DAE
        from other_classes import Sigmoid, tanh, Linear, MSE
        from fs_utils import feature_selection

        _np.random.seed(int(seed) % (2**31 - 1))
        # numpy>=2 compatibility: bundled Sparse_DAE uses np.in1d (removed in
        # numpy 2) on structured-array row views.  The original np.in1d
        # flattens its inputs; np.isin does not always, so provide a
        # behavior-identical raveling alias (no algorithmic change).
        if not hasattr(_np, "in1d"):
            def _in1d_compat(a, b, **kw):
                return _np.isin(_np.asarray(a).ravel(), _np.asarray(b).ravel(), **kw)
            _np.in1d = _in1d_compat
        n, d = X.shape
        hidden = 1000
        workdir = Path(workdir or Path("."))
        workdir.mkdir(parents=True, exist_ok=True)
        prefix = str(workdir / "qs_run_")  # Sparse_DAE concatenates strings
        (workdir / "qs_run_").mkdir(parents=True, exist_ok=True)  # for metrics.txt

        act = tanh if d < 500 else Linear   # driver uses tanh for madelon (d<=500 family), else Linear
        dae = Sparse_DAE((d, hidden, d), (Sigmoid, act), epsilon=13)
        Xtr = X.astype(float)
        # driver uses batch_size=100; for n < 100 that yields zero batches and
        # the SET evolution then fails (empty pdw). Documented adaptation:
        batch = int(min(100, max(1, Xtr.shape[0])))
        dae.fit(Xtr, y.astype(float).ravel(), Xtr, y.astype(float).ravel(),
                loss=MSE, epochs=int(epochs), batch_size=batch,
                learning_rate=0.01, momentum=0.9, weight_decay=0.00001,
                zeta=0.2, dropoutrate=0.2, testing=False, save_filename=prefix,
                noise_factor=0.2)
        w1 = dae.w[1]   # input -> hidden
        w2 = dae.w[2]   # hidden -> output
        k = int(min(max_budget, d))
        idx = feature_selection(w1, w2, k, "Node Strength(in)")
        ranking = np.asarray(idx, dtype=int).ravel()
        # fs_strength_w returns top-k descending strength: prefix-stable ranking
        return BaselineOutcome("QuickSelection", "OK", ranking, ranking[:k],
                               len(ranking), time.perf_counter() - t0,
                               notes=f"epochs={epochs}, hidden={hidden}, epsilon=13, zeta=0.2",
                               registry_key="QuickSelection")
    except Exception as exc:
        return BaselineOutcome("QuickSelection", "FAILED", None, None, 0,
                               time.perf_counter() - t0,
                               error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                               registry_key="QuickSelection")


def run_baseline(method: str, X_train: np.ndarray, y_train: np.ndarray,
                 max_budget: int, seed: int = 42,
                 quickselection_epochs: int = 3,
                 workdir: Optional[Path] = None) -> BaselineOutcome:
    """Dispatch a baseline by published name.  Proxies are impossible here."""
    if method in PROXY_BANNED:
        return BaselineOutcome(method, "UNAVAILABLE", None, None, 0, 0.0,
                               notes="proxy implementations are banned under published names",
                               error=f"'{method}' has no authentic local implementation; "
                                     "legacy numbers remain frozen, unverified artifacts.")
    # Explicit missing-dependency error: never fall back to a proxy.
    if method in {"mRMR", "ReliefF", "CMIM", "JMIM", "FCBF", "FRFS", "PPFS",
                  "QuickSelection"} and not BUNDLE_ROOT.exists():
        return BaselineOutcome(
            method, "UNAVAILABLE", None, None, 0, 0.0,
            notes="authentic baseline bundle missing in this environment",
            error=(f"Authentic baseline bundle not found at: {BUNDLE_ROOT}. "
                   f"Provide the local 'FHFAM bundle' directory (or adjust "
                   f"sifhfam/baselines.py::BUNDLE_ROOT) to run authentic "
                   f"'{method}'. Missing dependencies must NOT be substituted with "
                   f"a proxy implementation."),
        )
    dispatch: Dict[str, Callable[..., BaselineOutcome]] = {
        "mRMR": _select_mrmr,
        "ReliefF": _select_relieff,
        "CMIM": _select_cmim,
        "JMIM": _select_jmim,
        "FCBF": _select_fcbf,
        "FRFS": _select_frfs,
        "PPFS": _select_ppfs,
    }
    if method in dispatch:
        fn = dispatch[method]
        try:
            return fn(X_train, y_train, max_budget, seed)
        except TypeError:
            return fn(X_train, y_train, max_budget, seed)
    if method == "QuickSelection":
        return _select_quickselection(X_train, y_train, max_budget, seed,
                                      epochs=quickselection_epochs, workdir=workdir)
    if method == "ATR":
        return BaselineOutcome(method, "UNAVAILABLE", None, None, 0, 0.0,
                               notes="multi-label method; task-incompatible",
                               error="ATR is a multi-label feature selector; not run for single-label setting.")
    return BaselineOutcome(method, "UNAVAILABLE", None, None, 0, 0.0,
                           error=f"Unknown method '{method}'.")


def write_manifest_csv(path: Path) -> Path:
    import csv
    rows = baseline_registry()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method_name", "status", "exact_source_path", "package_repository_name",
              "version_or_commit", "supervised_unsupervised", "output_type",
              "parameters_used", "random_seed_behavior", "preprocessing_required",
              "task_compatibility", "primary_paper_metadata_found_locally", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return path
