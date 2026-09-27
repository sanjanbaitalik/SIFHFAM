import csv
from pathlib import Path

import pytest

from sifhfam import freeze_guard


def test_freeze_manifest_exists_and_nonempty():
    root = freeze_guard.project_root()
    manifest = freeze_guard.evidence_root(root) / "00_freeze" / "legacy_sha256_manifest.csv"
    if not manifest.exists():
        pytest.skip("freeze not created yet (run: python -m sifhfam.freeze_guard create)")
    rows = freeze_guard.read_manifest(manifest)
    assert len(rows) > 100
    kinds = {r["kind"] for r in rows}
    assert "legacy_result" in kinds and "legacy_code" in kinds


def test_legacy_hash_unchanged():
    root = freeze_guard.project_root()
    out = freeze_guard.evidence_root(root) / "00_freeze"
    if not (out / "legacy_sha256_manifest.csv").exists():
        pytest.skip("freeze not created yet")
    report = freeze_guard.verify_freeze(root, out)
    assert report["ok"] is True
    assert report["n_changed"] == 0
    assert report["n_missing"] == 0
    assert report["n_added"] == 0


def test_manifest_covers_required_files():
    root = freeze_guard.project_root()
    manifest = freeze_guard.evidence_root(root) / "00_freeze" / "legacy_sha256_manifest.csv"
    if not manifest.exists():
        pytest.skip("freeze not created yet")
    rows = {r["relative_path"] for r in freeze_guard.read_manifest(manifest)}
    for required in ["sif_hfam.py", "run_sif_hfam_experiment.py",
                     "reviewer_revision/sifhfam_selector.py",
                     "reviewer_revision/baseline_selectors.py",
                     "run_reviewer_revision.py",
                     "fs_experiments/feature_selection_methods.py",
                     "Outputs/run_manifest.json"]:
        assert required in rows, f"missing {required} from freeze manifest"
    assert any(p.startswith("Outputs/") for p in rows)
    assert any(p.startswith("fs_experiments/") for p in rows)
