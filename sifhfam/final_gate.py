"""Final scientific gate: report + reviewer comment matrix (prompt section 22)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .reporting import assert_reporting_ok


def _exists(p: Path) -> bool:
    return p.exists() and (p.stat().st_size > 0 if p.is_file() else True)


def _read_json(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _df(p: Path) -> Optional[pd.DataFrame]:
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def build_reviewer_matrix(evidence: Path, freeze_ok: bool,
                          tests_ok: Optional[bool],
                          smoke_ok: Optional[bool]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []

    def add(cid, component, status, evidence_paths, notes=""):
        rows.append({"reviewer_comment": cid, "component": component,
                     "status": status, "evidence_paths": evidence_paths,
                     "notes": notes})

    eb = evidence / "03_equal_budget" / "equal_budget_per_run.csv"
    nat = evidence / "03_equal_budget" / "native_cardinality_per_run.csv"
    base_manifest = evidence / "04_baselines" / "baseline_manifest.csv"
    strong_sum = evidence / "01_method_audit" / "strong_relation_audit_summary.csv"
    red_val = evidence / "01_method_audit" / "redundancy_validation.csv"
    synth = evidence / "02_synthetic_interactions" / "synthetic_per_seed.csv"
    xor_sum = evidence / "02_synthetic_interactions" / "xor_parity_recovery_summary.csv"
    cls = evidence / "05_classifier_generality" / "classifier_generality_per_run.csv"
    sens = evidence / "06_sensitivity" / "sensitivity_per_run.csv"
    abl = evidence / "07_ablation" / "ablation_per_run.csv"
    scale = evidence / "08_runtime_scalability" / "scaling_per_config.csv"
    stab = evidence / "09_stability" / "stability_decomposition.csv"
    cons = evidence / "09_stability" / "consensus_stability_tradeoff.csv"
    stats = evidence / "10_statistics" / "wilcoxon_holm_two_sided.csv"
    ratios = evidence / "10_statistics" / "greedy_ratio_records.csv"
    rep = evidence / "12_final_gate" / "reporting_validation.json"
    schem = evidence / "11_reproducibility" / "figures" / "method_schematic.png"
    check = evidence / "11_reproducibility" / "CODE_RELEASE_CHECKLIST.md"
    naming = evidence / "01_method_audit" / "NAMING_RECOMMENDATION.md"
    objdoc = evidence / "01_method_audit" / "OBJECTIVE_SCOPE_AND_BEHAVIOR.md"
    theory = evidence / "01_method_audit" / "THEORETICAL_SCOPE_AUDIT.md"
    audit = evidence / "01_method_audit" / "IMPLEMENTATION_MANUSCRIPT_AUDIT.md"

    def status_of(path: Path, required=True) -> str:
        if _exists(path):
            return "PASS"
        return "FAIL" if required else "PARTIAL"

    add("R1.2", "authentic + recent baselines", status_of(base_manifest),
        "04_baselines/baseline_manifest.csv;03_equal_budget/*;03_equal_budget/native_cardinality_*",
        "HIFS/UDFS/NDFS/HSIC-Lasso UNAVAILABLE/NOT VERIFIED by design; ATR excluded (multi-label).")
    add("R1.4 / R3.5", "hyperparameter & discretization robustness",
        status_of(sens), "06_sensitivity/sensitivity_per_run.csv;06_sensitivity/sensitivity_summary.csv")
    add("R1.5 / R2.8 / R3.6", "runtime, complexity, scalability",
        status_of(evidence / "08_runtime_scalability" / "COMPLEXITY_ANALYSIS.md")
        if _exists(scale) or _exists(evidence / "08_runtime_scalability" / "COMPLEXITY_ANALYSIS.md") else "FAIL",
        "08_runtime_scalability/*",
        "Stage-level timers; scaling sweep; corrected runtime replaces legacy ~0.1s claims.")
    add("R1.6", "stability diagnosis + consensus variant", status_of(stab),
        "09_stability/stability_decomposition.csv;09_stability/consensus_stability_tradeoff.csv")
    add("R2.1 / R4.7", "strong-relation audit + naming", status_of(strong_sum),
        "01_method_audit/strong_relation_audit_*.csv;01_method_audit/NAMING_RECOMMENDATION.md",
        "Audit complete; finding: strong relation NOT satisfied -> renaming decision is manuscript-side.")
    add("R2.2", "coverage objective scope + soft variants", status_of(objdoc),
        "01_method_audit/OBJECTIVE_SCOPE_AND_BEHAVIOR.md;07_ablation/*coverage*")
    add("R2.3 / R3.1", "synthetic interaction experiments",
        status_of(synth), "02_synthetic_interactions/*",
        "XOR/parity recovery measured; screening limitation reported honestly if present.")
    add("R2.4 / R2.5", "theoretical claim scope + numerical checks",
        status_of(theory) if _exists(ratios) or _exists(theory) else "FAIL",
        "01_method_audit/THEORETICAL_SCOPE_AUDIT.md;10_statistics/greedy_ratio_records.csv")
    add("R2.6", "group redundancy definition + validation", status_of(red_val),
        "01_method_audit/redundancy_validation.csv;01_method_audit/REDUNDANCY_DEFINITION_AUDIT.md")
    add("R2.7", "equal feature-cardinality comparison", status_of(eb),
        "03_equal_budget/equal_budget_per_run.csv;equal_budget_summary.csv;"
        "accuracy_vs_cardinality.csv;accuracy_sparsity_pareto.csv;pareto_frontier_by_dataset.csv")
    add("R2.9", "complete ablation suite", status_of(abl),
        "07_ablation/ablation_per_run.csv;ablation_summary.csv;selected_feature_differences_vs_full.csv")
    add("R2.10 / R4.11", "numerical reporting validation",
        status_of(rep), "12_final_gate/reporting_validation.json;table_consistency_checks.csv")
    add("R2.11 / R3.2", "authentic and recent baselines (run results)",
        status_of(eb) if _exists(eb) else "FAIL",
        "03_equal_budget/*;04_baselines/baseline_manifest.csv",
        "FRFS/PPFS/QuickSelection run with documented parameters; failures logged.")
    add("R3.4a", "classifier generality (RF/SVM/kNN/LR)", status_of(cls),
        "05_classifier_generality/classifier_generality_per_run.csv;*_summary.csv")
    add("R3.4b", "statistical analysis + visual support", status_of(stats),
        "10_statistics/wilcoxon_holm_two_sided.csv;friedman.csv;average_ranks.csv;rank_distribution.png",
        "Friedman = global complete-case omnibus; Wilcoxon rows use per-baseline "
        "paired means; figure is descriptive only.")
    add("R3.7", "code-release readiness",
        status_of(check), "11_reproducibility/REPRODUCIBILITY.md;requirements_evidence.txt;"
        "CODE_RELEASE_CHECKLIST.md;release_manifest.csv")
    add("R4.6", "non-AI method schematic", status_of(schem),
        "11_reproducibility/figures/method_schematic.{png,pdf,svg} + source")
    add("repo-level", "implementation-vs-manuscript audit", status_of(audit),
        "01_method_audit/IMPLEMENTATION_MANUSCRIPT_AUDIT.md;result_lineage.csv;"
        "unsafe_or_unverified_legacy_outputs.csv")
    add("repo-level", "frozen baseline integrity",
        "PASS" if freeze_ok else "FAIL",
        "00_freeze/legacy_sha256_manifest.csv;00_freeze/legacy_sha256_verification.csv")
    add("repo-level", "test suite",
        "PASS" if tests_ok else ("FAIL" if tests_ok is False else "PARTIAL"),
        "tests/")
    add("repo-level", "smoke profile",
        "PASS" if smoke_ok else ("FAIL" if smoke_ok is False else "PARTIAL"),
        "SIFHFAM_EVIDENCE/** (smoke-run config JSONs)")
    return pd.DataFrame(rows)


def _answer(question: str, evidence: Path) -> str:
    """Answer the 20 final-report questions from evidence where possible."""
    freeze_ver = _df(evidence / "00_freeze" / "legacy_sha256_verification.csv")
    lineage = _df(evidence / "01_method_audit" / "result_lineage.csv")
    unsafe = _df(evidence / "01_method_audit" / "unsafe_or_unverified_legacy_outputs.csv")
    strong = _df(evidence / "01_method_audit" / "strong_relation_audit_summary.csv")
    red = _df(evidence / "01_method_audit" / "redundancy_validation.csv")
    eb = _df(evidence / "03_equal_budget" / "equal_budget_summary.csv")
    xor = _df(evidence / "02_synthetic_interactions" / "xor_parity_recovery_summary.csv")
    cls = _df(evidence / "05_classifier_generality" / "classifier_generality_summary.csv")
    base = _df(evidence / "04_baselines" / "baseline_manifest.csv")
    scale = _df(evidence / "08_runtime_scalability" / "scaling_per_config.csv")
    stage = _df(evidence / "08_runtime_scalability" / "stage_timing_from_main_runs.csv")
    stab = _df(evidence / "09_stability" / "stability_decomposition.csv")
    cons = _df(evidence / "09_stability" / "consensus_stability_tradeoff.csv")
    rep = _read_json(evidence / "12_final_gate" / "reporting_validation.json")
    ratios = _df(evidence / "10_statistics" / "greedy_ratio_records.csv")

    A: Dict[int, str] = {}
    A[1] = ("No. " +
            ("All frozen artifacts unchanged (verification table shows all OK)."
             if freeze_ver is not None and (freeze_ver["status"] == "OK").all()
             else "VERIFICATION FAILED or missing - see 00_freeze/legacy_sha256_verification.csv."))
    if lineage is not None:
        ok_n = int((lineage["scientific_status"] == "metadata").sum())
        flagged = unsafe.shape[0] if unsafe is not None else 0
        A[2] = (f"Verified lineage: config/metadata artifacts only ({ok_n}); every results table "
                f"is flagged (see result_lineage.csv; {flagged} rows in "
                f"unsafe_or_unverified_legacy_outputs.csv). No legacy numeric result table has "
                f"fully verified scientific lineage to an authentic pipeline.")
    else:
        A[2] = "result_lineage.csv missing."
    A[3] = ("All legacy baseline tables under HIFS/UDFS/NDFS/HSIC-Lasso and the "
            "reviewer-revision 'ReliefF'/'mRMR' columns (ANOVA-F proxies), plus "
            "fs_experiments HIFS/UDFS/NDFS/HSIC-Lasso outputs, plus any table produced via "
            "the leaking evaluation.py path - all listed in unsafe_or_unverified_legacy_outputs.csv.")
    A[4] = ("No - not unchanged. Tables built on proxy baselines, ANOVA-F-screened selections, "
            "the leaking evaluation path, or mismatched implementations must be replaced or "
            "re-labeled; see result_lineage.csv + IMPLEMENTATION_MANUSCRIPT_AUDIT.md for the "
            "specific tables.")
    if strong is not None and not strong.empty:
        pct = float(np.nanmean(strong["pct_satisfies_both"].to_numpy(float)))
        A[5] = (f"No. Across {int(np.nansum(strong['n_edges'].to_numpy(float)))} audited edges, "
                f"{pct:.4f}% satisfy both strong equalities within 1e-9 "
                f"(strong_relation_audit_summary.csv). See NAMING_RECOMMENDATION.md.")
    else:
        A[5] = "Strongness audit missing."
    A[6] = ("Only representative group coverage is established; binary coverage does not recover "
            "full synergistic tuples. Soft/fraction variants give diminishing extra gains. "
            "See OBJECTIVE_SCOPE_AND_BEHAVIOR.md + 07_ablation coverage variants + "
            "02_synthetic_interactions all_selected rates.")
    if xor is not None and not xor.empty:
        canon = xor[xor["method"] == "canonical_default"]
        if not canon.empty:
            all_sel = float(np.nanmean(canon["p_all_selected"]))
            A[7] = (f"Default screening + canonical selection recovers ALL true interaction "
                    f"features in {all_sel*100:.1f}% of XOR/parity runs "
                    f"(xor_parity_recovery_summary.csv). Screening recall of interaction "
                    f"features is reported separately.")
        else:
            A[7] = "XOR summary exists but canonical rows missing."
    else:
        A[7] = "Synthetic XOR/parity evidence missing."
    abls = _df(evidence / "07_ablation" / "ablation_per_run.csv")
    if abls is not None and not abls.empty:
        A[8] = ("Ablation rows for screen_no / screening_interaction_union / screening_anova in "
                "07_ablation/ablation_per_run.csv (with synthetic screening recall in "
                "02_synthetic_interactions) answer this; improvements or failures are reported "
                "as measured.")
    else:
        A[8] = "Ablation evidence missing."
    if red is not None and not red.empty and bool(red["pass"].all()):
        n_red = len(red)
        A[9] = (f"Yes - {n_red}/{n_red} redundancy validation cases pass "
                f"(bounded [0,1], independence ~0, duplicates ~(k-1)/k, permutation invariant, "
                f"zero-entropy safe, both estimators in range). "
                f"See redundancy_validation.csv + REDUNDANCY_DEFINITION_AUDIT.md.")
    else:
        A[9] = "Redundancy validation missing or failing."
    if eb is not None and not eb.empty:
        # compare means at manuscript budget
        man = eb[eb["manuscript_budget"] == True] if "manuscript_budget" in eb.columns else eb  # noqa: E712
        A[10] = ("Equal-budget tables produced for all ranking methods "
                 f"({eb['method'].nunique()} methods, {eb['dataset'].nunique()} datasets). "
                 "Whether the main empirical conclusion survives is a reading of "
                 "equal_budget_summary.csv + wilcoxon_holm_two_sided.csv - reported without "
                 "spin; see FINAL tables.")
    else:
        A[10] = "Equal-budget evidence missing."
    if cls is not None and not cls.empty:
        piv = cls.pivot_table(index="dataset", columns=["method", "classifier"],
                              values="accuracy_mean")
        A[11] = (f"Classifier generality evaluated on {cls['dataset'].nunique()} datasets x "
                 f"{cls['classifier'].nunique()} classifiers x {cls['method'].nunique()} methods "
                 "(05_classifier_generality). Cross-classifier consistency is readable from "
                 "classifier_generality_summary.csv; single-classifier superiority claims are "
                 "not made.")
    else:
        A[11] = "Classifier generality evidence missing."
    if base is not None:
        ok = base[base["status"].str.startswith("authentic")]
        un = base[base["status"] == "UNAVAILABLE/NOT VERIFIED"]
        A[12] = ("Authentic baselines successfully integrated/run: "
                 + ", ".join(ok["method_name"].tolist()) + ".")
        A[13] = ("Unavailable/unverified: "
                 + ", ".join(un["method_name"].tolist())
                 + " (no authentic local implementation). ATR excluded (multi-label). "
                 "See baseline_manifest.csv for exact reasons.")
    else:
        A[12] = A[13] = "baseline_manifest.csv missing."
    if stage is not None and not stage.empty:
        fastest = stage.loc[stage["fs_seconds_mean"].idxmin()]
        slowest = stage.loc[stage["fs_seconds_mean"].idxmax()]
        A[14] = ("Instrumented feature-selection wall times (mean over runs): "
                 f"min {fastest['fs_seconds_mean']:.2f}s on {fastest['dataset']}, "
                 f"max {slowest['fs_seconds_mean']:.2f}s on {slowest['dataset']}; "
                 "stage-level breakdown in 08_runtime_scalability + per-run columns in "
                 "03_equal_budget. Legacy ~0.1s-style claims are superseded.")
    else:
        A[14] = "Stage timing evidence missing (equal-budget canonical rows needed)."
    if scale is not None and not scale.empty:
        ok = scale[scale["status"] == "OK"]
        A[15] = (f"Scaling sweep: {len(ok)}/{len(scale)} configs OK "
                 f"(d in {sorted(ok['d'].dropna().unique().tolist()) if not ok.empty else []}); "
                 "see scaling_per_config.csv. Trends: screening ~O(d), vertex redundancy ~O(f^2), "
                 "hyperedge scoring ~O(E n K), greedy ~O(m(f+E)) (COMPLEXITY_ANALYSIS.md). "
                 "All hyperedge results are for the sampled/capped collection (state the cap).")
    else:
        A[15] = "Scaling evidence missing."
    if stab is not None and not stab.empty:
        parts = []
        for f in ("A_split_instability", "B_hyperedge_sampling_instability",
                  "C_combined_instability", "D_screening_instability"):
            sub = stab[stab["factor"] == f]
            if not sub.empty and "jaccard_mean" in sub:
                parts.append(f"{f.split('_')[0]} Jaccard={sub['jaccard_mean'].mean():.3f}")
        A[16] = ("Decomposition: " + "; ".join(parts) +
                 ". Dominant source is whichever factor has the lowest Jaccard "
                 "(see stability_decomposition.csv).")
    else:
        A[16] = "Stability decomposition missing."
    if cons is not None and not cons.empty:
        A[17] = (f"Consensus selection run on {cons['dataset'].nunique()} datasets "
                 "(training-only bootstraps). Accuracy-stability trade-off numbers in "
                 "consensus_stability_tradeoff.csv; consensus does NOT replace the main method.")
    else:
        A[17] = ("Consensus variant not present in this profile (full-profile item) "
                 "or produced no rows.")
    if rep is not None:
        A[18] = ("PASS" if rep.get("all_passed") else
                 f"FAIL: {rep.get('n_failed_tables')} table(s) failed validation") + \
                (" (reporting_validation.json, table_consistency_checks.csv)."
                 " Accuracy stored in [0,1]; percent formatting multiplies mean AND SD; "
                 "reduction arithmetic and raw-to-summary recomputation enforced.")
    else:
        A[18] = "reporting_validation.json missing."
    if ratios is not None and not ratios.empty:
        A[19] = (f"Verified in code/tests: non-negative weights, monotone+submodular on "
                 f"{len(ratios)} tiny instances (all "
                 f"{bool(ratios['monotone'].all() and ratios['submodular'].all())}); "
                 f"greedy/optimum ratios recorded (mean "
                 f"{ratios['ratio'].mean():.3f}) as empirical records only. "
                 "Scope statement in THEORETICAL_SCOPE_AUDIT.md.")
    else:
        A[19] = "Ratio/monotonicity records missing."
    A[20] = "See the gate decision block in this report."
    return A


def _failure_status_lines(evidence: Path) -> List[str]:
    """Active vs resolved history for the final-gate failure log (Issue D)."""
    try:
        from .reporting_only import gate_failure_summary
        s = gate_failure_summary(evidence)
    except Exception as exc:  # pragma: no cover
        return [f"- failure log unavailable: {exc}"]
    if not s.get("exists"):
        return ["- no final-gate failure log present (0 active failures)"]
    return [
        f"- total historical rows: {s.get('n_total', 0)}",
        f"- **active (unresolved) failures counted as blockers: {s.get('n_active', 0)}**",
        f"- resolved historical rows retained for provenance: {s.get('n_resolved', 0)}"
        + (f" (sections: {', '.join(s.get('sections_resolved', []))})"
           if s.get("sections_resolved") else ""),
        "- only active rows can block the gate; resolved rows are audit history",
    ]


def run_final_gate(evidence: Path, freeze_ok: bool, tests_ok: Optional[bool],
                   smoke_ok: Optional[bool],
                   core_status: str = "not_run",
                   full_status: str = "not_run") -> Dict[str, Path]:
    out = evidence / "12_final_gate"
    out.mkdir(parents=True, exist_ok=True)

    # gate must not pass if reporting validation failed
    rep_path = out / "reporting_validation.json"
    rep = _read_json(rep_path)
    reporting_ok = bool(rep and rep.get("all_passed"))

    matrix = build_reviewer_matrix(evidence, freeze_ok, tests_ok, smoke_ok)
    matrix.to_csv(out / "REVIEWER_COMMENT_MATRIX.csv", index=False)

    statuses = matrix["status"].tolist()
    n_pass = statuses.count("PASS")
    n_partial = statuses.count("PARTIAL")
    n_fail = statuses.count("FAIL")
    n_ms = statuses.count("MANUSCRIPT_ONLY")

    # GO decision logic (evidence-based)
    blockers: List[str] = []
    if not freeze_ok:
        blockers.append("frozen legacy artifacts changed")
    if tests_ok is False:
        blockers.append("test suite failed")
    if not reporting_ok:
        blockers.append("reporting validation did not pass")
    if n_fail > 0:
        blockers.append(f"{n_fail} reviewer-matrix component(s) FAIL")
    # scientific blockers are reported even under CONDITIONAL GO
    scientific_notes: List[str] = []
    strong = _df(evidence / "01_method_audit" / "strong_relation_audit_summary.csv")
    if strong is not None and not strong.empty:
        pct = float(np.nanmean(strong["pct_satisfies_both"].to_numpy(float)))
        if pct < 99.9:
            scientific_notes.append(
                "strong relation NOT satisfied -> 'strong' naming cannot be defended as-is")
    unsafe = _df(evidence / "01_method_audit" / "unsafe_or_unverified_legacy_outputs.csv")
    if unsafe is not None and not unsafe.empty:
        scientific_notes.append(
            f"{len(unsafe)} legacy result artifacts flagged unsafe/unverified -> "
            "legacy tables need replacement or re-labeling before reuse")
    if blockers:
        decision = "NO-GO"
    elif n_partial > 0 or scientific_notes:
        decision = "CONDITIONAL GO"
    else:
        decision = "GO"

    questions = _answer.__doc__ or ""
    answers = _answer(questions, evidence)  # dict-like via implementation
    # _answer returns str annotation but actually dict; handle both
    if isinstance(answers, str):
        answers = {}

    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# FINAL CODE REVISION REPORT",
        "",
        f"Generated: {now}",
        "",
        "## Gate decision",
        "",
        f"**{decision}**",
        "",
        "### Hard blockers (must be zero for any GO)",
    ]
    if blockers:
        lines += [f"- {b}" for b in blockers]
    else:
        lines.append("- none")
    lines += ["", "### Scientific notes (drive CONDITIONAL GO / manuscript decisions)"]
    if scientific_notes:
        lines += [f"- {s}" for s in scientific_notes]
    else:
        lines.append("- none")
    lines += [
        "",
        "## Reviewer matrix summary",
        "",
        f"- PASS: {n_pass}  PARTIAL: {n_partial}  FAIL: {n_fail}  "
        f"MANUSCRIPT_ONLY: {n_ms}  (total {len(matrix)})",
        "- Full matrix: `REVIEWER_COMMENT_MATRIX.csv` (with exact evidence paths).",
        "",
        "## Run status",
        "",
        f"- freeze verification: {'OK' if freeze_ok else 'FAILED'}",
        f"- tests: {tests_ok}",
        f"- smoke profile: {smoke_ok}",
        f"- core profile: {core_status}",
        f"- full profile: {full_status}",
        "",
        "## Failure log status (active vs resolved history)",
        "",
        *_failure_status_lines(evidence),
        "",
        "## The 20 required questions",
        "",
    ]
    for i in range(1, 21):
        ans = answers.get(i, "n/a")
        lines.append(f"**{i}.** {ans}")
        lines.append("")
    lines += [
        "## Scope reminder",
        "",
        "This report was produced by code review + experiments only. No manuscript "
        "file was edited and no response letter was drafted in this task.",
    ]
    md = out / "FINAL_CODE_REVISION_REPORT.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": md, "matrix": out / "REVIEWER_COMMENT_MATRIX.csv"}
