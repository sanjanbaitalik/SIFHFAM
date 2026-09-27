"""Freeze and re-verify legacy result/code artifacts (prompt section 1)."""
from __future__ import annotations

import csv
import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# Files/directories that constitute the frozen legacy surface.
LEGACY_RESULT_GLOBS: Sequence[str] = (
    "Outputs",
    "fs_experiments",
)

LEGACY_CODE_FILES: Sequence[str] = (
    "sif_hfam.py",
    "run_sif_hfam_experiment.py",
    "run_sif_hfam_example.py",
    "evaluation.py",
    "sensitivity_figure.py",
    "convert_mat_to_csv.py",
    "run_reviewer_revision.py",
    "README_REVIEWER_REVISION.md",
    "reviewer_revision/__init__.py",
    "reviewer_revision/config.py",
    "reviewer_revision/data_utils.py",
    "reviewer_revision/metrics.py",
    "reviewer_revision/baseline_selectors.py",
    "reviewer_revision/sifhfam_selector.py",
    "reviewer_revision/run_all_revision_experiments.py",
    "reviewer_revision/requirements_reviewer_revision.txt",
    "fs_experiments/feature_selection_methods.py",
    "fs_experiments/run_fs_experiments.py",
    "fs_experiments/run_fs_selection.py",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def evidence_root(root: Path | None = None) -> Path:
    return (root or project_root()) / "SIFHFAM_EVIDENCE"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(q for q in p.rglob("*") if q.is_file() and "__pycache__" not in q.parts))
    return out


def legacy_paths(root: Path) -> List[Path]:
    paths: List[Path] = []
    for g in LEGACY_RESULT_GLOBS:
        paths.append(root / g)
    for rel in LEGACY_CODE_FILES:
        paths.append(root / rel)
    return [p for p in paths if p.exists()]


def hash_manifest(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for p in _iter_files(legacy_paths(root)):
        rel = p.relative_to(root).as_posix()
        kind = "legacy_result" if any(rel == g or rel.startswith(g + "/") for g in LEGACY_RESULT_GLOBS) else "legacy_code"
        rows.append({
            "relative_path": rel,
            "kind": kind,
            "size_bytes": p.stat().st_size,
            "sha256": sha256_of_file(p),
        })
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def environment_report() -> str:
    lines: List[str] = []
    lines.append(f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"python: {sys.version}")
    lines.append(f"executable: {sys.executable}")
    lines.append(f"platform: {platform.platform()}")
    lines.append(f"machine: {platform.machine()}")
    lines.append(f"processor: {platform.processor()}")
    try:
        import os as _os
        lines.append(f"cpu_count: {os.cpu_count()}")
    except Exception:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        lines.append(f"ram_total_bytes: {vm.total}")
        lines.append(f"ram_available_bytes: {vm.available}")
    except Exception as exc:
        lines.append(f"ram: unavailable ({exc})")
    for mod in ("numpy", "scipy", "sklearn", "pandas", "matplotlib", "psutil"):
        try:
            m = __import__(mod)
            lines.append(f"{mod}: {getattr(m, '__version__', 'unknown')}")
        except Exception as exc:
            lines.append(f"{mod}: not available ({exc})")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        lines.append(f"env {var}: {os.environ.get(var, '<unset>')}")
    try:
        import numpy as np
        config = np.show_config(mode="dicts") if hasattr(np, "show_config") else {}
        lines.append(f"numpy_config: {config}")
    except Exception as exc:
        lines.append(f"numpy_config: unavailable ({exc})")
    return "\n".join(lines) + "\n"


def git_status_report(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--branch"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            return out.stdout + out.stderr
        return f"git status failed (exit {out.returncode}):\n{out.stdout}\n{out.stderr}"
    except FileNotFoundError:
        return "NO GIT REPOSITORY: `git` executable not found or directory is not a git repository.\n"
    except Exception as exc:
        return f"git status error: {exc}\n"


def create_freeze(root: Path | None = None, out_dir: Path | None = None) -> Path:
    root = root or project_root()
    out = out_dir or (evidence_root(root) / "00_freeze")
    out.mkdir(parents=True, exist_ok=True)

    rows = hash_manifest(root)
    write_csv(
        out / "legacy_sha256_manifest.csv",
        rows,
        ("relative_path", "kind", "size_bytes", "sha256"),
    )

    result_rows = [r for r in rows if r["kind"] == "legacy_result"]
    write_csv(
        out / "legacy_result_inventory.csv",
        [{k: r[k] for k in ("relative_path", "size_bytes", "sha256")} for r in result_rows],
        ("relative_path", "size_bytes", "sha256"),
    )

    code_rows = [r for r in rows if r["kind"] == "legacy_code"]
    write_csv(
        out / "legacy_code_inventory.csv",
        [{k: r[k] for k in ("relative_path", "size_bytes", "sha256")} for r in code_rows],
        ("relative_path", "size_bytes", "sha256"),
    )

    (out / "environment_before.txt").write_text(environment_report(), encoding="utf-8")
    (out / "git_status_before.txt").write_text(git_status_report(root), encoding="utf-8")

    lineage = """# Legacy result lineage (frozen)

This file records, to the best of code-level inspection, **which implementation
produced which existing result surface**.  It is written *before* any new code
was added, and the hashed files are immutable inputs for re-verification.

## Frozen surfaces

| Surface | Suspected producing code | Notes |
|---|---|---|
| `Outputs/00_config` .. `Outputs/08_ordered_association` (incl. `run_manifest.json`) | `reviewer_revision/run_all_revision_experiments.py` invoked by `run_reviewer_revision.py` | `Outputs/run_manifest.json` records datasets=14 paper datasets, methods=`SIFHFAM,SIFFAM,ReliefF,mRMR,HSIC-Lasso,HIFS,UDFS,NDFS`, n_runs=10, sections incl. main/sensitivity/ablation/stats/stability/dose/ifs/default. |
| `Outputs/01_main_repeated/*` method columns HSIC-Lasso, HIFS, UDFS, NDFS | `reviewer_revision/baseline_selectors.py` proxy functions `_fast_hsic_proxy`, `_fast_hifs_proxy`, `_fast_udfs_proxy`, `_fast_ndfs_proxy` | These are explicitly named `*_proxy` in source; they are **not** authentic implementations of the published methods. |
| `Outputs/01_main_repeated/*` method column `ReliefF` | `reviewer_revision/baseline_selectors.py::_fast_relieff_proxy` | ANOVA-F ranking, **not** ReliefF (Kira et al. / skrebate). |
| `Outputs/01_main_repeated/*` method column `mRMR` | `reviewer_revision/baseline_selectors.py::_fast_mrmr` | ANOVA-F relevance + Pearson correlation redundancy pool greedy; **not** the MI-based mRMR of Peng et al. |
| `Outputs/01_main_repeated/*` method columns `SIFHFAM`, `SIFFAM` | `reviewer_revision/sifhfam_selector.py` | Uses ANOVA-F screening + correlation redundancy + sampled hyperedges + binary coverage greedy. Not identical to `sif_hfam.py`. |
| `fs_experiments/fs_results.csv`, `fs_experiments/selected_features/*` | `fs_experiments/run_fs_experiments.py`, `fs_experiments/run_fs_selection.py`, `fs_experiments/feature_selection_methods.py` | Contains HIFS/UDFS/NDFS/HSIC-Lasso outputs from functions documented in-source as proxies/placeholders; SPEC/LS are additional legacy baselines. |
| `Outputs/03_hyperparameter_sensitivity/*` | `reviewer_revision/run_all_revision_experiments.py` sensitivity section | lambda/alpha/beta sweep on the reviewer-revision selector. |
| `Outputs/05_ablation/*` | same, ablation section | vertex-only/no-redundancy/no-hesitation/K variants of the reviewer-revision selector. |
| `Outputs/07_stability_consistency/*` | same, stability section | Jaccard/Kuncheva across repeated splits. |
| Legacy one-off scripts | `run_sif_hfam_experiment.py` + `sif_hfam.py` + `evaluation.py` | `evaluation.py` selects features on the **full** dataset before repeated hold-out splitting (selection/evaluation leakage) — see method audit. |

## Scientific status flags (pre-audit, refined in `01_method_audit/`)

- Proxy-named baseline outputs are preserved for provenance but are **not
  verified evidence** for the published method names HIFS/UDFS/NDFS/HSIC-Lasso/
  ReliefF/mRMR as produced by `reviewer_revision/baseline_selectors.py`.
- `fs_experiments` HIFS/UDFS/NDFS are in-source documented proxies/placeholders.
- The two SIFHFAM implementations (`sif_hfam.py` vs `reviewer_revision/
  sifhfam_selector.py`) implement materially different pipelines; the lineage of
  every table must be checked before retention.

No file listed in `legacy_sha256_manifest.csv` may change during this revision
task; `freeze_guard.verify_freeze()` fails loudly on any mismatch.
"""
    (out / "LEGACY_RESULT_LINEAGE.md").write_text(lineage, encoding="utf-8")
    return out


def verify_freeze(root: Path | None = None, out_dir: Path | None = None) -> Dict[str, object]:
    """Re-hash frozen artifacts and write `legacy_sha256_verification.csv`.

    Returns a report dict; ``ok`` is False if any frozen artifact changed,
    disappeared, or a new file appeared under a frozen result directory.
    """
    root = root or project_root()
    out = out_dir or (evidence_root(root) / "00_freeze")
    manifest_path = out / "legacy_sha256_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing freeze manifest: {manifest_path}")
    original = read_manifest(manifest_path)
    current_rows = hash_manifest(root)
    current = {r["relative_path"]: r for r in current_rows}

    rows: List[Dict[str, object]] = []
    n_ok = n_changed = n_missing = n_added = 0
    for rec in original:
        rel = rec["relative_path"]
        cur = current.get(rel)
        if cur is None:
            status = "MISSING"
            n_missing += 1
            new_hash = ""
            old_size = rec["size_bytes"]
            new_size = ""
        else:
            new_hash = str(cur["sha256"])
            old_size = rec["size_bytes"]
            new_size = str(cur["size_bytes"])
            if new_hash == rec["sha256"]:
                status = "OK"
                n_ok += 1
            else:
                status = "CHANGED"
                n_changed += 1
        rows.append({
            "relative_path": rel,
            "status": status,
            "expected_sha256": rec["sha256"],
            "actual_sha256": new_hash,
            "expected_size": old_size,
            "actual_size": new_size,
        })

    orig_set = {r["relative_path"] for r in original}
    for rel in sorted(set(current) - orig_set):
        n_added += 1
        rows.append({
            "relative_path": rel,
            "status": "ADDED",
            "expected_sha256": "",
            "actual_sha256": current[rel]["sha256"],
            "expected_size": "",
            "actual_size": current[rel]["size_bytes"],
        })

    write_csv(
        out / "legacy_sha256_verification.csv",
        rows,
        ("relative_path", "status", "expected_sha256", "actual_sha256",
         "expected_size", "actual_size"),
    )

    ok = (n_changed == 0 and n_missing == 0 and n_added == 0)
    report = {
        "ok": ok,
        "n_ok": n_ok,
        "n_changed": n_changed,
        "n_missing": n_missing,
        "n_added": n_added,
        "manifest": str(manifest_path),
        "verification": str(out / "legacy_sha256_verification.csv"),
    }
    if not ok:
        bad = [r["relative_path"] for r in rows if r["status"] != "OK"]
        raise AssertionError(
            "FROZEN ARTIFACT VERIFICATION FAILED for: " + ", ".join(bad[:20])
            + (f" (+{len(bad)-20} more)" if len(bad) > 20 else "")
        )
    return report


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Freeze / verify legacy artifacts")
    ap.add_argument("action", choices=["create", "verify"])
    args = ap.parse_args()
    if args.action == "create":
        p = create_freeze()
        print(f"Freeze manifest written to {p}")
    else:
        rep = verify_freeze()
        print(rep)
