"""Code-release readiness artifacts (prompt section 18). Nothing is published."""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_release_docs(root: Path, evidence: Path) -> Dict[str, Path]:
    out = evidence / "11_reproducibility"
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    requirements = """# requirements_evidence.txt
#
# Python: 3.14 was used for the evidence run; Python >= 3.10 should work.
# Everything in the [pip-installable] block comes from PyPI:
#   pip install -r requirements_evidence.txt
#
# ---- PyPI-installable block ----
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
psutil>=5.9
pytest>=7.0
# Used by authentic baselines reached via path insertion (all PyPI-installable):
joblib>=1.2
tqdm>=4.60
networkx>=2.8
# category_encoders is only needed if the pypi `mrmr` package is imported;
# that package is NOT used under the published mRMR name (see baseline manifest).
#
# [NOT pip-installable from this repository - external local requirements]
#   * Dataset directory:  <project_root>/Updated Dataset/<name>_X.csv + _Y.csv
#       (external data; NOT bundled. Provide your own copies.)
#   * Authentic baseline bundle: <workspace>/FHFAM bundle/
#       (local source checkouts of skfeature, scikit-rebate, mifs, frlearn,
#        PyImpetus, FCBF_module, QuickSelection; NOT redistributed here.)
#   A missing dataset or bundle must surface as an explicit error; it must
#   NEVER trigger a silent fallback to an inauthentic proxy baseline.
"""
    (out / "requirements_evidence.txt").write_text(requirements, encoding="utf-8")
    paths["requirements"] = out / "requirements_evidence.txt"

    repro = f"""# REPRODUCIBILITY

Generated: {datetime.now(timezone.utc).isoformat()}

## Environment

- Python: {sys.version.split()[0]}
- NumPy {np.__version__}

Exact package versions used for the evidence run are recorded in
`08_runtime_scalability/environment.json` and `00_freeze/environment_before.txt`.

## Directory layout

```
SIFHFAM_v2/                          # project root
  Outputs/                           # FROZEN legacy results (never modified)
  fs_experiments/                    # FROZEN legacy results (never modified)
  Updated Dataset/                   # <name>_X.csv / <name>_Y.csv pairs
  sif_hfam.py                        # FROZEN legacy implementation A
  reviewer_revision/                 # FROZEN legacy implementation B
  sifhfam/                      # canonical implementation package
  run_sifhfam.py                # top-level CLI runner
  tests/                        # test suite
  SIFHFAM_EVIDENCE/          # NEW evidence root (all new outputs)
```

## Dataset-path assumptions

- Datasets are CSV pairs `Updated Dataset/<name>_X.csv` (headerless features) and
  `Updated Dataset/<name>_Y.csv` (headerless labels).
- Paper-14 list is defined in `sifhfam/config.py::PAPER14_DATASETS`.
- Singleton label classes (<2 samples) are removed at load time and the row
  counts are recorded in the run metadata.

## Seed policy

- Base seed default: 42 (`--seed` to change).
- Per-run seeds: `run_seed(base, dataset, run)` (stable string hash; recorded).
- Exact train/test indices for every run are written to
  `SIFHFAM_EVIDENCE/03_equal_budget/split_indices.json`.
- Hyperedge sampling uses `numpy.random.default_rng(cfg.random_state)`.
- Threading: `n_jobs=1` everywhere (sklearn classifiers, ReliefF, mifs, PPIMBC);
  documented exceptions: none.

## Exact commands

```bash
# 0) reporting-only regeneration (NO experiments; safe in any environment
#    that has the raw evidence CSVs):
python -m sifhfam.reporting_only

# 1) create/verify the legacy freeze (final-gate verification step)
python -m sifhfam.freeze_guard create   # only if the manifest is absent
python -m sifhfam.freeze_guard verify   # expects n_changed=n_missing=n_added=0

# 2) FULL test suite (requires Updated Dataset/ AND FHFAM bundle/ - see the
#    test dependency table below):
python -m pytest tests/ -q

# 2b) LIGHTWEIGHT self-contained subset (no datasets, no baseline bundle):
python -m pytest tests/test_ifs_strongness.py \
    tests/test_redundancy_information.py \
    tests/test_objective.py \
    tests/test_theory_scope.py \
    tests/test_reporting.py \
    tests/test_reporting_cleanup.py -q

# 3) smoke profile
python run_sifhfam.py --profile smoke --datasets colon,madelon --runs 2

# 4) core reviewer run
python run_sifhfam.py --profile core --datasets paper14 --runs 10

# 5) full run (adds expensive baselines, full synthetic suite, scaling,
#    consensus stability, strict-strong ablation)
python run_sifhfam.py --profile full --datasets paper14 --runs 10

# Sections can be selected explicitly:
python run_sifhfam.py --profile core --sections freeze,audit,equal_budget
```

## External environment requirements (what a clean checkout does NOT contain)

A fresh clone of this repository is **not** self-contained. It requires:

1. **Python**: >= 3.10 (3.14 used for the evidence run; exact versions in
   `08_runtime_scalability/environment.json`).
2. **PyPI packages**: everything listed above `requirements_evidence.txt`
   under the pip-installable block (`pip install -r requirements_evidence.txt`).
   These include numpy, pandas, scipy, scikit-learn, matplotlib, psutil,
   pytest, joblib, tqdm, networkx.
3. **Authentic baseline packages**: shipped as *local source checkouts* in the
   sibling `FHFAM bundle/` directory (skfeature, scikit-rebate, mifs,
   frlearn/fuzzy-rough-learn, PyImpetus, FCBF_module, QuickSelection). They are
   reached via explicit path insertion in `sifhfam/baselines.py`
   (`BUNDLE_ROOT`), **not** via pip. If the bundle is absent, those baselines
   return an explicit `UNAVAILABLE` result naming the missing path.
4. **Datasets**: external CSV pairs under `Updated Dataset/` (see
   "Dataset-path assumptions"). They are not bundled or redistributed here.

## Test dependency classification

| Test file | Needs `Updated Dataset/` | Needs `FHFAM bundle/` | Notes |
|---|---|---|---|
| `test_ifs_strongness.py` | no | no | fully self-contained |
| `test_redundancy_information.py` | no | no | fully self-contained |
| `test_objective.py` | no | no | fully self-contained |
| `test_theory_scope.py` | no | no | fully self-contained |
| `test_reporting.py` | no | no | fully self-contained |
| `test_reporting_cleanup.py` | no | no | fully self-contained regression tests |
| `test_freeze_guard.py` | no | no | needs the committed freeze manifest only |
| `test_screening_splits_synthetic.py` | **yes** (colon) | no | split/screening/fit tests load `colon` |
| `test_baselines.py` | no | **partly** | registry/proxy tests are self-contained; `test_authentic_baselines_run_on_tiny_problem` needs the bundle |

Commands for the self-contained subset are given above (2b). If a dataset or
the baseline bundle is unavailable, run the self-contained subset and report
the environmental limitation honestly - do not fabricate results for the
dependent tests.

## Missing-dependency policy (important)

A missing external dependency must **never** silently fall back to an
inauthentic proxy implementation of a published method:

- `sifhfam/baselines.py::PROXY_BANNED` forbids exporting proxies under
  published names (HIFS/UDFS/NDFS/HSIC-Lasso/...);
- a missing `FHFAM bundle/` yields an explicit `UNAVAILABLE` outcome with the
  exact missing path;
- a missing dataset directory raises `FileNotFoundError` explaining the
  expected `<name>_X.csv` / `<name>_Y.csv` layout and how to point the code
  at your data (`load_dataset(..., data_dir=...)`).

## Baseline dependency notes

- Authentic baselines are loaded from the local `FHFAM bundle/` via explicit
  path insertion (see `sifhfam/baselines.py` and
  `04_baselines/baseline_manifest.csv`).
- `mRMR`/`CMIM`: scikit-feature (skfeature), discrete input (train-only
  quantile discretization); high-d MI prefilter pool=1000 is documented per run.
- `ReliefF`: scikit-rebate (skrebate).
- `JMIM`: mifs package (`method='JMIM'`); kNN MI estimator requires effectively
  continuous features and smallest train class >= 3; failures are logged.
- `FCBF`: local FCFB/FCBF_module (Yu & Liu algorithm).
- `FRFS`: fuzzy-rough-learn (frlearn).
- `PPFS`: PyImpetus PPIMBC.
- `QuickSelection`: local Sparse_DAE + fs_utils; two documented adaptations
  (batch_size=min(100,n); numpy>=2 np.in1d alias) are noted in the manifest.
- `ATR`: multi-label, excluded as task-incompatible.
- `HIFS`/`UDFS`/`NDFS`/`HSIC-Lasso`: UNAVAILABLE/NOT VERIFIED - no authentic
  local implementation; legacy numbers remain frozen, unverified artifacts.

## License notes for bundled third-party code

- scikit-rebate: MIT (license file in bundle).
- scikit-feature: BSD-style (see bundle README/LICENSE).
- fuzzy-rough-learn: see bundle LICENSE.
- FCBF_module: see bundle LICENSE.
- PyImpetus: MIT (per setup.py classifiers/README).
- mifs: BSD-3-Clause (module docstring author note).
- mrmr package: see bundle LICENSE (not used under the published mRMR name).
- QuickSelection: see bundle repo README/citation header.
- ATR: see bundle README (not used).

Do NOT redistribute third-party code inside a release package unless its
license permits it; reference the upstream repositories instead.

## `.gitignore` recommendations

```
__pycache__/
*.pyc
.pytest_cache/
SIFHFAM_EVIDENCE/**/_quickselection_tmp/
SIFHFAM_EVIDENCE/**/*.npz
*.egg-info/
.ipynb_checkpoints/
# keep legacy results but consider not force-adding huge .npy blobs if LFS used
```

## Clean release manifest

See `11_reproducibility/release_manifest.csv` (sha256 of every file in
`sifhfam/`, the runner, and the test suite).
"""
    (out / "REPRODUCIBILITY.md").write_text(repro, encoding="utf-8")
    paths["reproducibility"] = out / "REPRODUCIBILITY.md"

    # release manifest of NEW code only
    rows = []
    for rel_root in ("sifhfam", "tests"):
        base = root / rel_root
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".txt", ".md"}:
                rows.append({"path": p.relative_to(root).as_posix(),
                             "size_bytes": p.stat().st_size,
                             "sha256": _sha256(p)})
    runner = root / "run_sifhfam.py"
    if runner.exists():
        rows.append({"path": "run_sifhfam.py",
                     "size_bytes": runner.stat().st_size,
                     "sha256": _sha256(runner)})
    import csv as _csv
    mp = out / "release_manifest.csv"
    with mp.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    paths["release_manifest"] = mp

    gi = out / "gitignore_recommendations.txt"
    gi.write_text(
        "__pycache__/\n*.pyc\n.pytest_cache/\n"
        "SIFHFAM_EVIDENCE/**/_quickselection_tmp/\n"
        "SIFHFAM_EVIDENCE/**/*.npz\n"
        "*.egg-info/\n.ipynb_checkpoints/\n"
        "# do not git-commit secrets: none are used (all seeds are public constants)\n",
        encoding="utf-8")
    paths["gitignore"] = gi

    checklist = """# CODE_RELEASE_CHECKLIST

- [x] No manuscript .tex/.bib/response file was edited by this task.
- [x] Legacy results frozen with SHA-256 manifest and re-verified
      (`00_freeze/legacy_sha256_manifest.csv`, `legacy_sha256_verification.csv`).
- [x] All new outputs live under `SIFHFAM_EVIDENCE/`.
- [x] Proxy implementations are banned from new tables
      (`sifhfam/baselines.py::PROXY_BANNED`).
- [x] Baseline registry with exact source paths and authenticity labels
      (`04_baselines/baseline_manifest.csv`).
- [x] Unavailable baselines (HIFS/UDFS/NDFS/HSIC-Lasso) explicitly marked
      UNAVAILABLE/NOT VERIFIED; no silent substitution.
- [x] Split indices and seeds persisted (`03_equal_budget/split_indices.json`).
- [x] Test-only data never used for screening, selection, or tuning
      (all selectors fit on train folds only; tests assert this).
- [x] Thread control: n_jobs=1 recorded in configs; environment.json records
      BLAS/thread env vars.
- [x] Reporting validators gate the final report
      (`12_final_gate/reporting_validation.json`).
- [x] Statistical tests are two-sided with Holm correction; raw inputs saved.
- [x] No theorem/citation/runtime/baseline parameter invented; complexity
      derived from the implemented pipeline.
- [x] Failures (XOR limitations, JMIM dataset failures, low stability, runtime
      corrections) are reported in `12_final_gate/FINAL_CODE_REVISION_REPORT.md`.
- [ ] Independent human review of the evidence package before manuscript edits.
"""
    (out / "CODE_RELEASE_CHECKLIST.md").write_text(checklist, encoding="utf-8")
    paths["checklist"] = out / "CODE_RELEASE_CHECKLIST.md"
    return paths
