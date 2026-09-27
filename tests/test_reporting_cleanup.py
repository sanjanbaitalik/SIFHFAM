"""Regression tests for the post-processing reporting cleanup (Issues A-E).

All tests in this file are fully self-contained: no datasets and no external
baseline bundle are required.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sifhfam.reporting_only import (
    active_failure_count,
    build_synthetic_summary,
    build_xor_parity_summary,
    cleanup_gate_decision,
    compare_hash_frames,
    gate_failure_summary,
    normalize_failure_log,
    validate_xor_summary,
)


# ---------------------------------------------------------------- Issue A - #

def _make_synthetic_raw() -> pd.DataFrame:
    """Two sample sizes x 2 scenarios x 2 noise regimes, 5 seeds each."""
    rows = []
    for n in (100, 300):
        for scenario in ("xor2", "parity3"):
            for noise in ("small", "large"):
                for seed in range(5):
                    rows.append({
                        "scenario": scenario,
                        "n": n,
                        "noise_regime": noise,
                        "seed": seed,
                        "method": "canonical_default",
                        "all_selected": 0.0 if seed < 4 else 1.0,  # mean 0.2
                        "recall": 0.1 * seed,
                        "precision": 0.05 * seed,
                        "f1": 0.02 * seed,
                        "accuracy": 0.5 + 0.01 * seed,
                        "screening_recall_interaction": 0.5,
                    })
                    rows.append({
                        "scenario": scenario,
                        "n": n,
                        "noise_regime": noise,
                        "seed": seed,
                        "method": "ReliefF",
                        "all_selected": 1.0,
                        "recall": 1.0,
                        "precision": 0.5,
                        "f1": 0.6,
                        "accuracy": 0.6,
                        "screening_recall_interaction": np.nan,
                    })
    return pd.DataFrame(rows)


def test_xor_summary_preserves_sample_size_n():
    """Issue A regression: group-key `n` must stay the experimental sample size,
    never be overwritten by a seed count."""
    raw = _make_synthetic_raw()
    summary = build_xor_parity_summary(raw)

    # 1) n is the experimental sample size
    assert set(summary["n"].unique()) == {100, 300}
    # 2) seed count lives in a distinct column
    assert "n_seeds" in summary.columns
    assert set(summary["n_seeds"].unique()) == {5}
    # 3) grouping keys unique
    keys = ["scenario", "n", "noise_regime", "method"]
    assert not summary.duplicated(subset=keys).any()
    # 4) no raw rows lost or duplicated
    assert int(summary["n_seeds"].sum()) == len(raw)
    # 5) sample sizes and seed counts are not conflated
    assert not (summary["n"] == summary["n_seeds"]).any() or True  # 100/300 vs 5
    assert (summary["n"] != summary["n_seeds"]).all()


def test_xor_summary_reproduces_raw_means_exactly():
    raw = _make_synthetic_raw()
    summary = build_xor_parity_summary(raw)
    rep = validate_xor_summary(raw, summary)
    assert rep["ok"], rep["errors"]
    # canonical finding derived from raw, not hard-coded
    canon = summary[summary["method"] == "canonical_default"]
    assert float(canon["p_all_selected"].mean()) == pytest.approx(0.2)
    assert rep["canonical_zero_complete_recovery"] is False  # raw has 1.0s here
    relief = summary[summary["method"] == "ReliefF"]
    assert float(relief["p_all_selected"].mean()) == pytest.approx(1.0)


def test_synthetic_summary_keeps_both_sample_sizes():
    raw = _make_synthetic_raw()
    summary = build_synthetic_summary(raw)
    assert set(summary["n"].unique()) == {100, 300}
    assert summary.duplicated(subset=["scenario", "n", "noise_regime", "method"]).sum() == 0


def test_validation_detects_seed_count_overwrite():
    """If someone reintroduces the bug (n = seed count), validation must fail."""
    raw = _make_synthetic_raw()
    buggy = build_xor_parity_summary(raw)
    buggy["n"] = buggy["n_seeds"]  # simulate the old overwrite
    rep = validate_xor_summary(raw, buggy)
    assert not rep["ok"]


# ---------------------------------------------------------------- Issue B - #

def test_paired_means_use_identical_indices_as_test():
    from sifhfam.statistics import paired_comparison

    rng = np.random.default_rng(0)
    datasets = [f"ds{i}" for i in range(6)]
    ref = rng.random(6)
    other = rng.random(6)
    other[2] = np.nan          # baseline missing on ds2
    other[5] = np.nan          # and on ds5

    row = paired_comparison(ref, other, method="B", ref_method="REF",
                            dataset_labels=datasets)
    paired_idx = [i for i in range(6)
                  if np.isfinite(ref[i]) and np.isfinite(other[i])]
    assert row["n_pairs"] == len(paired_idx) == 4
    # means computed from exactly the paired intersection
    assert row["reference_mean_paired"] == pytest.approx(
        float(np.mean(ref[paired_idx])))
    assert row["comparator_mean_paired"] == pytest.approx(
        float(np.mean(other[paired_idx])))
    # the reference global mean must NOT leak into the paired mean
    assert row["reference_mean_paired"] != pytest.approx(float(np.mean(ref)))
    # legacy aliases carry the paired quantities too
    assert row["ref_mean"] == row["reference_mean_paired"]
    assert row["baseline_mean"] == row["comparator_mean_paired"]
    # paired dataset labels match the intersection
    assert row["paired_datasets"] == ";".join(datasets[i] for i in paired_idx)


def test_missing_baseline_dataset_cannot_leak_into_reference_mean():
    from sifhfam.statistics import paired_comparison, run_statistics

    # raw run-level frame: ref on 4 datasets, baseline only on 2
    rows = []
    for ds, ref_v in [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]:
        rows.append({"dataset": ds, "run": 0, "method": "SIFHFAM_canonical",
                     "accuracy": ref_v})
    for ds, b_v in [("a", 0.5), ("b", 0.4)]:
        rows.append({"dataset": ds, "run": 0, "method": "BASE", "accuracy": b_v})
    df = pd.DataFrame(rows)
    w, friedman, ranks, pivot = run_statistics(df, ref_method="SIFHFAM_canonical")
    assert len(w) == 1
    row = w.iloc[0]
    assert int(row["n_pairs"]) == 2
    assert row["reference_mean_paired"] == pytest.approx((0.9 + 0.8) / 2)
    assert row["comparator_mean_paired"] == pytest.approx((0.5 + 0.4) / 2)
    # global ref mean over 4 datasets must differ from the paired mean
    assert row["reference_mean_paired"] != pytest.approx((0.9 + 0.8 + 0.7 + 0.6) / 4)


def test_holm_alignment_with_baseline_labels():
    from sifhfam.statistics import holm_correction, run_statistics

    rows = []
    for ds in ("a", "b", "c", "d", "e"):
        rows.append({"dataset": ds, "run": 0, "method": "SIFHFAM_canonical",
                     "accuracy": 0.9})
    # BASE_SMALL has tiny p (identical structure shifted), BASE_NOISY larger p
    for ds, v in zip(("a", "b", "c", "d", "e"), (0.5, 0.5, 0.5, 0.5, 0.51)):
        rows.append({"dataset": ds, "run": 0, "method": "BASE_SMALL", "accuracy": v})
    for ds, v in zip(("a", "b", "c", "d", "e"), (0.7, 0.6, 0.8, 0.55, 0.65)):
        rows.append({"dataset": ds, "run": 0, "method": "BASE_NOISY", "accuracy": v})
    df = pd.DataFrame(rows)
    w, _, _, _ = run_statistics(df, ref_method="SIFHFAM_canonical")
    # recompute holm independently and check label alignment
    order = np.argsort(w["p_raw"].to_numpy(dtype=float))
    # recompute via the public helper on the SAME (possibly unsorted) vector
    holm = holm_correction(w["p_raw"].to_numpy(dtype=float))
    for i, row in w.iterrows():
        assert row["p_holm"] == pytest.approx(holm[i])
        assert row["significant_holm_0.05"] == bool(np.isfinite(holm[i]) and holm[i] < 0.05)
    # labels unchanged by correction
    assert set(w["method"]) == {"BASE_SMALL", "BASE_NOISY"}


def test_friedman_is_complete_case_and_labelled_omnibus():
    from sifhfam.statistics import run_statistics

    rows = []
    for ds in ("a", "b", "c", "d"):
        for m, v in [("SIFHFAM_canonical", 0.9), ("M2", 0.8), ("M3", 0.7)]:
            rows.append({"dataset": ds, "run": 0, "method": m, "accuracy": v})
    # incomplete method -> excluded from complete-case analysis
    rows.append({"dataset": "a", "run": 0, "method": "PARTIAL", "accuracy": 0.5})
    df = pd.DataFrame(rows)
    w, friedman, ranks, pivot = run_statistics(df, ref_method="SIFHFAM_canonical")
    fr = friedman.iloc[0]
    assert int(fr["n_methods"]) == 3
    assert "PARTIAL" in str(fr["excluded_methods"])
    assert "omnibus" in str(fr["interpretation"])
    # paired table still includes the partial baseline (with n_pairs=1)
    partial_row = w[w["method"] == "PARTIAL"].iloc[0]
    assert int(partial_row["n_pairs"]) == 1


# ---------------------------------------------------------------- Issue C - #

def test_statistics_source_has_no_false_cd_label():
    """No generated code/output may call the base rank-SE term a Nemenyi CD."""
    root = Path(__file__).resolve().parents[1]
    src = root / "sifhfam" / "statistics.py"
    text = src.read_text(encoding="utf-8").lower()
    assert "nemenyi" not in text
    assert "cd=" not in text
    assert "critical difference" not in text
    # also scan the runner + final gate + stats sources for the claim itself
    # (reporting_only.py intentionally CONTAINS the word as the audit detector
    # regex that scans generated evidence files for it)
    for rel in ("run_sifhfam.py",
                "sifhfam/final_gate.py",
                "sifhfam/experiments.py",
                "sifhfam/statistics.py"):
        t = (root / rel).read_text(encoding="utf-8")
        assert "nemenyi" not in t.lower(), rel
    # the old filename must never be used as a SAVE TARGET anymore (references
    # in deletion/audit code are legitimate)
    for rel in ("run_sifhfam.py",
                "sifhfam/final_gate.py",
                "sifhfam/experiments.py",
                "sifhfam/statistics.py"):
        t = (root / rel).read_text(encoding="utf-8")
        assert 'rank_boxplot_cd.png")' not in t, (
            f"{rel} still saves the old mislabelled figure name")


def test_rank_plot_outputs_have_descriptive_names(tmp_path):
    from sifhfam.statistics import plot_rank_distribution

    ranks = pd.DataFrame({"method": ["A", "B"], "average_rank": [1.2, 1.8],
                          "analysis": ["complete_case", "complete_case"]})
    pivot = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]},
                         index=["d1", "d2"])
    out = tmp_path / "rank_distribution.png"
    p = plot_rank_distribution(ranks, pivot, out)
    if p is not None:  # matplotlib available in the evidence environment
        assert p.name == "rank_distribution.png"
        assert "cd" not in p.name.lower()


# ---------------------------------------------------------------- Issue D - #

def test_resolved_failures_do_not_count_as_active(tmp_path):
    log = tmp_path / "failure_log.csv"
    pd.DataFrame([{
        "timestamp": "2026-09-22T19:32:31", "section": "reporting",
        "where": "validation", "message": "failed tables: 1",
    }]).to_csv(log, index=False)

    def resolve(row):
        return "validation now passes" if row["section"] == "reporting" else None

    df = normalize_failure_log(log, is_resolved=resolve, write=True)
    assert active_failure_count(df) == 0
    assert bool(df.iloc[0]["resolved"]) is True
    assert df.iloc[0]["status"] == "resolved"
    assert "validation now passes" in str(df.iloc[0]["resolution"])
    # provenance preserved (original message retained)
    assert str(df.iloc[0]["message"]) == "failed tables: 1"
    # round-trip: re-normalizing keeps it resolved
    df2 = normalize_failure_log(log, is_resolved=lambda r: None, write=True)
    assert active_failure_count(df2) == 0


def test_active_failures_still_count(tmp_path):
    log = tmp_path / "failure_log.csv"
    pd.DataFrame([
        {"timestamp": "t1", "section": "reporting", "where": "v",
         "message": "failed tables: 1"},
        {"timestamp": "t2", "section": "experiment", "where": "x",
         "message": "boom"},
    ]).to_csv(log, index=False)
    df = normalize_failure_log(log, is_resolved=lambda r: None, write=True)
    assert active_failure_count(df) == 2


def test_cleanup_gate_blocked_by_active_failure_or_bad_inputs():
    ok = dict(freeze_ok=True, tests_ok=True, reporting_ok=True,
              active_failures=0, raw_intact=True)
    assert cleanup_gate_decision(**ok) == "READY_FOR_MANUSCRIPT_TEXT_EDITS"
    assert cleanup_gate_decision(**{**ok, "active_failures": 1}) == "BLOCKED"
    assert cleanup_gate_decision(**{**ok, "reporting_ok": False}) == "BLOCKED"
    assert cleanup_gate_decision(**{**ok, "freeze_ok": False}) == "BLOCKED"
    assert cleanup_gate_decision(**{**ok, "tests_ok": False}) == "BLOCKED"
    assert cleanup_gate_decision(**{**ok, "raw_intact": False}) == "BLOCKED"


def test_gate_failure_summary_on_real_log_if_present():
    """If the evidence package is present, active must be 0 after cleanup."""
    evidence = Path(__file__).resolve().parents[1] / "SIFHFAM_EVIDENCE"
    log = evidence / "12_final_gate" / "failure_log.csv"
    if not log.exists():
        pytest.skip("evidence package not present")
    df = pd.read_csv(log)
    if "resolved" not in df.columns:
        pytest.skip("failure log not normalized yet (run: python -m sifhfam.reporting_only)")
    s = gate_failure_summary(evidence)
    assert s["n_active"] == 0, f"active failures remain: {s}"


# --------------------------------------------------- raw integrity helper - #

def test_hash_comparison_detects_changes(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("x")
    b.write_text("y")
    before = pd.DataFrame([{"file": "a", "sha256": "h1", "size": 1},
                           {"file": "b", "sha256": "h2", "size": 1}])
    after = pd.DataFrame([{"file": "a", "sha256": "h1", "size": 1},
                          {"file": "b", "sha256": "H2_CHANGED", "size": 1}])
    cmp_df = compare_hash_frames(before, after)
    statuses = dict(zip(cmp_df["file"], cmp_df["status"]))
    assert statuses["a"] == "UNCHANGED"
    assert statuses["b"] == "CHANGED"
