#!/usr/bin/env python3
"""Top-level runner for the Neurocomputing revision evidence package.

Examples
--------
Smoke:
    python run_sifhfam.py --profile smoke --datasets colon,madelon --runs 2

Core reviewer run:
    python run_sifhfam.py --profile core --datasets paper14 --runs 10

Full run:
    python run_sifhfam.py --profile full --datasets paper14 --runs 10

Sections (comma separated):
    freeze,audit,manifest,equal_budget,native,classifiers,sensitivity,ablation,
    synthetic,runtime,stability,stats,reporting,figures,release,finalgate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sifhfam.config import PAPER14_DATASETS, profile_grids  # noqa: E402
from sifhfam import freeze_guard  # noqa: E402


def parse_datasets(arg: str, data_dir: Optional[Path] = None) -> List[str]:
    if arg.lower() in {"paper14", "paper", "default"}:
        return list(PAPER14_DATASETS)
    if arg.lower() == "all":
        data_dir = Path(data_dir) if data_dir else ROOT / "Updated Dataset"
        stems = []
        for p in sorted(data_dir.glob("*_X.csv")):
            stem = p.name[: -len("_X.csv")]
            if (data_dir / f"{stem}_Y.csv").exists():
                stems.append(stem)
        if not stems:
            raise FileNotFoundError(
                f"--datasets all found no dataset pairs in: {data_dir}. "
                "Provide the external dataset directory with --data-dir "
                "(expected <data_dir>/<name>_X.csv and <data_dir>/<name>_Y.csv). "
                "Datasets are not distributed with this repository."
            )
        return stems
    return [s.strip() for s in arg.split(",") if s.strip()]


ALL_SECTIONS = [
    "freeze", "audit", "manifest", "equal_budget", "native", "classifiers",
    "sensitivity", "ablation", "synthetic", "runtime", "stability", "stats",
    "reporting", "figures", "release", "finalgate",
]

CORE_SECTIONS = [
    "freeze", "audit", "manifest", "equal_budget", "native", "classifiers",
    "sensitivity", "ablation", "synthetic", "runtime", "stability", "stats",
    "reporting", "figures", "release", "finalgate",
]

FULL_SECTIONS = list(CORE_SECTIONS)

SMOKE_SECTIONS = [
    "freeze", "audit", "manifest", "equal_budget", "native", "classifiers",
    "sensitivity", "ablation", "synthetic", "runtime", "stability", "stats",
    "reporting", "figures", "release", "finalgate",
]


def _status_from_path(p: Path) -> bool:
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "all_passed" in data:
            return bool(data["all_passed"])
    except Exception:
        return False
    return False


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Neurocomputing revision evidence runner")
    ap.add_argument("--profile", choices=["smoke", "core", "full"], default="smoke")
    ap.add_argument("--datasets", default="paper14",
                    help="'paper14', 'all', or comma-separated names")
    ap.add_argument("--runs", type=int, default=None, help="override n_runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sections", default=None,
                    help="comma-separated subset of: " + ",".join(ALL_SECTIONS))
    ap.add_argument("--verify-only", action="store_true",
                    help="only verify the freeze, then exit")
    ap.add_argument("--data-dir", default=None,
                    help="external dataset directory containing <name>_X.csv and "
                         "<name>_Y.csv pairs (default: <repo>/Updated Dataset). "
                         "Datasets are NOT distributed with this repository.")
    args = ap.parse_args(argv)
    data_dir = (Path(args.data_dir).expanduser().resolve() if args.data_dir
                else None)

    from sifhfam.experiments import (
        RunContext,
        run_ablation,
        run_classifier_generality,
        run_equal_budget,
        run_native_cardinality,
        run_reporting_validation,
        run_runtime,
        run_sensitivity,
        run_stability,
        run_statistics,
        run_synthetic,
    )
    from sifhfam.audit import run_method_audit
    from sifhfam.baselines import write_manifest_csv
    from sifhfam.figures import run_figures
    from sifhfam.release_docs import run_release_docs
    from sifhfam.final_gate import run_final_gate

    datasets = parse_datasets(args.datasets, data_dir)
    if args.sections:
        sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    else:
        sections = {"smoke": SMOKE_SECTIONS, "core": CORE_SECTIONS,
                    "full": FULL_SECTIONS}[args.profile]

    evidence = ROOT / "SIFHFAM_EVIDENCE"
    evidence.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    results: Dict[str, object] = {}
    freeze_ok = False
    tests_ok: Optional[bool] = None
    smoke_ok: Optional[bool] = None
    core_status = "not_run"
    full_status = "not_run"
    created_or_modified: List[str] = []
    errors: List[str] = []

    print(f"[run_sifhfam] profile={args.profile} datasets={datasets} "
          f"runs={args.runs} sections={sections}")

    # ---------------- freeze ----------------
    if "freeze" in sections or args.verify_only:
        try:
            if not (evidence / "00_freeze" / "legacy_sha256_manifest.csv").exists():
                freeze_guard.create_freeze(ROOT)
                created_or_modified.append("SIFHFAM_EVIDENCE/00_freeze/* (created)")
            rep = freeze_guard.verify_freeze(ROOT)
            freeze_ok = bool(rep["ok"])
            results["freeze"] = rep
            print(f"[freeze] verified: {rep['n_ok']} files OK")
        except AssertionError as exc:
            freeze_ok = False
            errors.append(f"FREEZE VERIFICATION FAILED: {exc}")
            print(f"[freeze] FAILED: {exc}", file=sys.stderr)
    if args.verify_only:
        print(json.dumps(results.get("freeze", {}), indent=2))
        return 0 if freeze_ok else 2

    ctx = RunContext.create(ROOT, args.profile, datasets, args.runs, args.seed,
                            data_dir=data_dir)
    print(f"[context] evidence={ctx.evidence}")
    print(f"[context] data-dir={ctx.load_dir()}"
          + ("" if ctx.load_dir().exists() else "  (missing: provide --data-dir)"))

    def step(name, fn, *a, **kw):
        if name not in sections:
            return
        t0 = time.time()
        print(f"[section] {name} ...", flush=True)
        try:
            r = fn(*a, **kw)
            results[name] = r
            dt = time.time() - t0
            print(f"[section] {name} done in {dt:.1f}s")
            if isinstance(r, dict):
                for k, v in r.items():
                    created_or_modified.append(str(Path(v).relative_to(ROOT))
                                               if isinstance(v, Path) and str(v).startswith(str(ROOT))
                                               else f"{name}:{k}")
            else:
                created_or_modified.append(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"[section] {name} FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()

    # ---------------- audit & manifest ----------------
    step("audit", run_method_audit,
         evidence / "01_method_audit", ROOT, datasets,
         (args.profile != "smoke") or True)
    step("manifest", write_manifest_csv, evidence / "04_baselines" / "baseline_manifest.csv")

    # ---------------- experiments ----------------
    step("equal_budget", run_equal_budget, ctx)
    step("native", run_native_cardinality, ctx)
    step("classifiers", run_classifier_generality, ctx)
    step("sensitivity", run_sensitivity, ctx)
    step("ablation", run_ablation, ctx)
    step("synthetic", run_synthetic, ctx)
    step("runtime", run_runtime, ctx)
    step("stability", run_stability, ctx)
    step("stats", run_statistics, ctx)
    step("reporting", run_reporting_validation, ctx)
    step("figures", run_figures, evidence)
    step("release", run_release_docs, ROOT, evidence)

    # ---------------- final gate ----------------
    if "finalgate" in sections:
        # infer statuses from prior state / artifacts
        tests_ok = _infer_tests(evidence)
        smoke_ok = _infer_profile_run(evidence, "smoke") if args.profile != "smoke" else True
        if args.profile == "smoke":
            smoke_ok = not errors
        # prefer explicit profile markers written by dedicated runs
        core_marker = _infer_profile_run(evidence, "core")
        full_marker = _infer_profile_run(evidence, "full")
        if args.profile in ("core", "full") and not errors:
            core_status = "OK"
        if core_marker:
            core_status = "OK"
        if args.profile == "full" and not errors:
            full_status = "OK"
        if full_marker:
            full_status = "OK"
        if args.profile in ("core", "full") and errors:
            core_status = f"ERRORS: {len(errors)}" if not core_marker else core_status
        try:
            gate = run_final_gate(evidence, freeze_ok, tests_ok, smoke_ok,
                                  core_status=core_status, full_status=full_status)
            results["finalgate"] = gate
            created_or_modified.append(str(gate["report"].relative_to(ROOT)))
            created_or_modified.append(str(gate["matrix"].relative_to(ROOT)))
            rep = _status_from_path(evidence / "12_final_gate" / "reporting_validation.json")
            print(f"[finalgate] reporting validation passed={rep}")
        except Exception as exc:
            errors.append(f"finalgate: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    # ---------------- completion message ----------------
    total = time.time() - t_start
    print("\n================ COMPLETION REPORT ================")
    print(f"profile: {args.profile}   elapsed: {total:.1f}s")
    print("files/dirs created or modified (new evidence only):")
    for item in sorted(set(created_or_modified))[:40]:
        print(f"  - {item}")
    if len(set(created_or_modified)) > 40:
        print(f"  ... (+{len(set(created_or_modified)) - 40} more)")
    print(f"frozen-baseline hash verification: "
          f"{'OK' if freeze_ok else 'FAILED/NOT RUN'}")
    print(f"test command: python -m pytest tests/ -q  "
          f"(status: {tests_ok})")
    print(f"smoke command: python run_sifhfam.py --profile smoke "
          f"--datasets colon,madelon --runs 2  (status: {smoke_ok})")
    print(f"core command: python run_sifhfam.py --profile core "
          f"--datasets paper14 --runs 10  (status: {core_status})")
    print(f"full command: python run_sifhfam.py --profile full "
          f"--datasets paper14 --runs 10  (status: {full_status})")
    bm = evidence / "04_baselines" / "baseline_manifest.csv"
    if bm.exists():
        import pandas as pd
        df = pd.read_csv(bm)
        auth = df[df["status"].astype(str).str.startswith("authentic")]["method_name"].tolist()
        unav = df[df["status"] == "UNAVAILABLE/NOT VERIFIED"]["method_name"].tolist()
        print(f"authentic baselines integrated: {', '.join(auth)}")
        print(f"baselines excluded/unverified: {', '.join(unav)} "
              f"(+ ATR multi-label incompatible, mrmr-pypi not published-mRMR)")
    if errors:
        print("errors during run:")
        for e in errors:
            print(f"  ! {e}")
    else:
        print("section errors: none")
    gate_md = evidence / "12_final_gate" / "FINAL_CODE_REVISION_REPORT.md"
    print(f"FINAL_CODE_REVISION_REPORT.md: {gate_md}")
    decision = "NOT DETERMINED"
    if gate_md.exists():
        for line in gate_md.read_text(encoding="utf-8").splitlines():
            if line.startswith("**") and ("GO" in line):
                decision = line.replace("*", "").strip()
                break
    print(f"final code-evidence gate: {decision}")
    print("====================================================\n")
    return 0 if (freeze_ok and not errors) else 1


def _infer_tests(evidence: Path) -> Optional[bool]:
    marker = evidence / "12_final_gate" / "pytest_status.json"
    if marker.exists():
        try:
            with marker.open("r", encoding="utf-8") as f:
                return bool(json.load(f).get("passed"))
        except Exception:
            return None
    return None


def _infer_profile_run(evidence: Path, profile: str) -> Optional[bool]:
    marker = evidence / f"profile_run_status_{profile}.json"
    if marker.exists():
        try:
            with marker.open("r", encoding="utf-8") as f:
                return bool(json.load(f).get("ok"))
        except Exception:
            return None
    return None


if __name__ == "__main__":
    sys.exit(main())
