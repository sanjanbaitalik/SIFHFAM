# SIFHFAM

**Structured Intuitionistic Fuzzy Hypergraph Feature Association Map**

## Scope

This repository contains the source code for the SIFHFAM feature-selection
framework: an **intuitionistic-fuzzy-weighted feature-hypergraph** method with
a **representative-coverage** objective solved by greedy selection.

The pipeline covers: training-only relevance screening, discretization,
information-theoretic vertex/group evidences, intuitionistic (mu, nu, pi)
degrees, hyperedge construction with non-negative weights, monotone submodular
coverage objectives, greedy selection, downstream evaluation, stability and
statistical analyses, synthetic interaction studies, and reporting/validation
utilities.

This is a research codebase. It makes no claims of state-of-the-art accuracy,
superior scalability, synergy recovery, or guaranteed feature recovery; those
questions are evaluated empirically by the included experiment sections and
tests.

## Repository layout

```
SIFHFAM_GitHub/
├── README.md
├── .gitignore
├── requirements.txt
├── sifhfam/            # canonical implementation package
├── run_sifhfam.py      # top-level CLI runner
└── tests/              # test suite (self-contained + external-resource tests)
```

## Installation

```bash
git clone <repository-url>
cd SIFHFAM_GitHub
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python >= 3.10 (the reference environment used Python 3.14 with
NumPy 2.5, SciPy 1.18, scikit-learn 1.8, pandas 3.0 - see
`requirements.txt` for the recorded versions).

## External datasets

**Benchmark datasets are not distributed with this repository.**

Provide your own dataset directory containing headerless CSV pairs:

```text
<data_dir>/<name>_X.csv   # features: n_samples rows x d columns
<data_dir>/<name>_Y.csv   # labels:   n_samples rows
```

and point the runner at it with `--data-dir`:

```bash
python run_sifhfam.py --datasets all --data-dir /path/to/datasets --verify-only
```

Without datasets, dataset-dependent sections and tests fail with an explicit
`FileNotFoundError` describing the expected layout. No toy or proxy datasets
are substituted automatically.

## Usage

Minimal executable command (reporting-only, safe without experiments):

```bash
python -m sifhfam.reporting_only
```

Smoke experiment run (requires datasets):

```bash
python run_sifhfam.py --profile smoke --datasets colon --runs 2 \
    --data-dir /path/to/datasets
```

Other runner options: `--profile {smoke,core,full}`, `--datasets paper14|all|<csv>`,
`--runs N`, `--seed N`, `--sections <list>`, `--verify-only`,
`--data-dir /path/to/datasets`. See `python run_sifhfam.py --help`.

## Tests

### Self-contained tests (no external resources)

```bash
python -m pytest tests/test_ifs_strongness.py \
    tests/test_redundancy_information.py \
    tests/test_objective.py \
    tests/test_theory_scope.py \
    tests/test_reporting.py \
    tests/test_reporting_cleanup.py -q
```

### External-resource-dependent tests

- `tests/test_screening_splits_synthetic.py` - most cases
  require a benchmark dataset directory (e.g. `colon`); synthetic-generator
  tests inside the file are self-contained.
- `tests/test_baselines.py::test_authentic_baselines_run_on_tiny_problem`
  - requires the optional local baseline bundle (see below).
- `tests/test_freeze_guard.py` - skips unless a freeze manifest
  from the research environment is present.

These are expected to skip or fail where the corresponding external resource is
intentionally absent; they are not regressions of this code-only distribution.

## Optional authenticated baselines

Authentic baseline wrappers (mRMR, ReliefF, CMIM, JMIM, FCBF, FRFS, PPFS,
QuickSelection) load a **local baseline bundle** from a sibling directory
named `FHFAM bundle/` next to this repository (see
`sifhfam/baselines.py::BUNDLE_ROOT`). The bundle is an external,
optional dependency and is **not** redistributed here. When it is missing,
baseline runs return an explicit `UNAVAILABLE` result naming the missing path -
they never fall back to proxy implementations under published method names.
Unavailable methods (HIFS, UDFS, NDFS, HSIC-Lasso) remain marked
UNAVAILABLE/NOT VERIFIED by design.

## Outputs

Experiment results are generated locally under `SIFHFAM_EVIDENCE/` (created on
first run) and are intentionally **not tracked by Git** (see `.gitignore`).
This repository ships no results, figures, or tables.

## Reproducibility

- Deterministic seeds: base seed via `--seed` (default 42); per-run seeds are
  derived deterministically and exact train/test split indices are persisted
  with each run.
- Training-only preprocessing: screening, discretization, and feature
  selection are fit on the training fold only.
- External datasets: results depend on the user-provided dataset directory.
- Optional baseline dependencies: authentic baselines require the external
  bundle; their absence is reported explicitly.

## Citation

Citation information will be added after publication.
