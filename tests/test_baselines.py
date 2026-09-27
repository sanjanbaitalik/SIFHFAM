import numpy as np
import pandas as pd
import pytest

from sifhfam.baselines import (
    PROXY_BANNED,
    baseline_registry,
    run_baseline,
    write_manifest_csv,
)


def test_registry_has_required_columns(tmp_path):
    p = write_manifest_csv(tmp_path / "manifest.csv")
    df = pd.read_csv(p)
    required = {"method_name", "status", "exact_source_path", "package_repository_name",
                "version_or_commit", "supervised_unsupervised", "output_type",
                "parameters_used", "random_seed_behavior", "preprocessing_required",
                "task_compatibility", "primary_paper_metadata_found_locally", "notes"}
    assert required <= set(df.columns)


def test_proxy_names_cannot_be_exported():
    for name in ["HIFS", "UDFS", "NDFS", "HSIC-Lasso", "ReliefF_proxy", "mRMR_proxy"]:
        X = np.random.RandomState(0).normal(size=(30, 5))
        y = (np.random.RandomState(1).rand(30) > 0.5).astype(int)
        ob = run_baseline(name, X, y, max_budget=3, seed=0)
        if name in PROXY_BANNED:
            assert ob.status == "UNAVAILABLE"
            assert ob.ranking is None and ob.native_subset is None
            assert ob.error


def test_unavailable_baselines_marked_in_registry():
    rows = baseline_registry()
    by_name = {r["method_name"]: r for r in rows}
    for name in ["HIFS", "UDFS", "NDFS", "HSIC-Lasso"]:
        assert by_name[name]["status"] == "UNAVAILABLE/NOT VERIFIED"
        assert by_name[name]["exact_source_path"] == ""
    assert by_name["ATR"]["status"] == "excluded_task_incompatible"


def test_authentic_baselines_run_on_tiny_problem():
    rng = np.random.RandomState(0)
    n, d = 80, 12
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.1 * rng.normal(size=n) > 0).astype(int)
    results = {}
    for m in ["mRMR", "ReliefF", "CMIM", "JMIM", "FCBF", "FRFS"]:
        ob = run_baseline(m, X, y, max_budget=6, seed=0)
        results[m] = ob
        assert ob.status == "OK", f"{m}: {ob.error}"
        if ob.ranking is not None:
            assert len(np.unique(ob.ranking)) == len(ob.ranking)
        if ob.native_subset is not None:
            assert len(ob.native_subset) == ob.n_selected
    # ranking methods must yield at least budget features on this toy set
    for m in ["mRMR", "ReliefF", "CMIM", "JMIM"]:
        assert results[m].n_selected >= 4


def test_every_runnable_method_has_registry_row():
    rows = {r["method_name"] for r in baseline_registry()}
    for m in ["mRMR", "ReliefF", "CMIM", "JMIM", "FCBF", "FRFS", "PPFS",
              "QuickSelection", "ATR", "HIFS", "UDFS", "NDFS", "HSIC-Lasso"]:
        assert m in rows
