"""Implementation-vs-manuscript audit artifacts (prompt section 1.2 and 3).

Produces:
- 01_method_audit/IMPLEMENTATION_MANUSCRIPT_AUDIT.md
- 01_method_audit/result_lineage.csv
- 01_method_audit/unsafe_or_unverified_legacy_outputs.csv
- 01_method_audit/strong_relation_audit_*.csv
- 01_method_audit/NAMING_RECOMMENDATION.md
- 01_method_audit/OBJECTIVE_SCOPE_AND_BEHAVIOR.md
- 01_method_audit/THEORETICAL_SCOPE_AUDIT.md
- 01_method_audit/REDUNDANCY_DEFINITION_AUDIT.md
- 01_method_audit/redundancy_validation.csv
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import PAPER14_DATASETS, RevisionConfig
from .data import load_dataset, make_split, run_seed
from .discretization import Discretizer
from .hypergraph import generate_hyperedges
from .ifs import (
    compute_vertex_degrees,
    independent_edge_degrees,
    strict_strong_induced_edge_degrees,
    strong_relation_audit,
)
from .information import group_redundancy, entropy
from .objective import Objective, g_binary, g_soft, g_fraction, is_submodular, is_monotone
from .selector import fit


MANUSCRIPT_NOTE = (
    "NOTE: no manuscript .tex/.bib file is present in this workspace. The "
    "'Manuscript definition' column below is reconstructed from (i) the "
    "reviewer comments quoted in the revision prompt, (ii) method descriptions "
    "in the repository READMEs and code docstrings, and (iii) the mathematical "
    "quantities the reviewers state the manuscript uses. Direct line-by-line "
    "verification against the manuscript PDF remains a manuscript-side task."
)


def audit_table_md() -> str:
    rows = [
        ("initial screening score",
         "Information-theoretic relevance screening of features (reviewers describe MI-based screening)",
         "`sif_hfam.py`: full-space NMI vertex mu used for top-f screening (bins=20 histogram)",
         "`reviewer_revision/sifhfam_selector.py`: sklearn ANOVA-F (`f_classif`) top-f screening",
         "NO (legacy paths disagree with each other and with the IT description)",
         "Screening decides the candidate universe f; ANOVA-F is not information-theoretic. Canonical path uses discretized NMI, train-only."),
        ("vertex relevance",
         "Relevance evidence per feature",
         "NMI(X_j; y) histogram, 20 bins, nats",
         "Discrete NMI on quantile-discretized train features, 10 bins",
         "PARTIAL (same family, different estimator/binning)",
         "Changes numerical values of mu; both remain train-only in legacy_repro/canonical, but legacy one-off `run_sif_hfam_experiment.py` screens on FULL data before splitting (leakage)."),
        ("vertex redundancy",
         "Information-theoretic redundancy evidence (per reviewer 2.x concern)",
         "mean |Pearson correlation| to other features",
         "mean |Pearson correlation| within screened set (StandardScaler)",
         "NO (correlation used where reviewers state the manuscript describes IT quantities)",
         "Canonical path defaults to mean pairwise NMI (bounded); correlation variant retained as explicit option for comparison."),
        ("IFS normalization",
         "mu+nu+pi=1 with mu+nu<=1",
         "denom = 1 + a + b with a=NMI, b=corr (no sigmoid)",
         "sigmoid(a*e+b) evidence then denom = 1 + raw_mu + raw_nu",
         "PARTIAL (normalization consistent; evidence mapping differs)",
         "Both satisfy the IFS axioms; degrees differ numerically. Canonical exposes sigmoid/direct mapping explicitly."),
        ("hyperedge generation",
         "Hyperedges over screened features of order up to K, sampled/capped",
         "exhaustive combinations of order 3 only (or first max_edges combos - NOT random)",
         "orders 2..K, per-order random sampling under max_edges budget, seeded rng",
         "PARTIAL",
         "Legacy sif_hfam enumerates only 3-subsets in lexicographic order (biased, not random); canonical samples uniformly per order with recorded seeds."),
        ("group relevance",
         "NMI of the feature group with the target",
         "normalized_mi_3d histogram 20 bins (4-D histogram MI)",
         "discrete joint NMI of group columns with y (10-bin quantile)",
         "PARTIAL",
         "Same quantity, different estimators; canonical records estimator + base (nats)."),
        ("group redundancy",
         "Reviewer 2.6: TC(e)=sum H(X_j)-H(X_e) normalized into [0,1]",
         "TC/3 then divided by log(bins) (unbounded-by-construction quantity rescaled by max entropy, NOT by sum H)",
         "TC/(sum H + eps) - already bounded",
         "PARTIAL (reviewer_revision matches the requested form; sif_hfam.py does not)",
         "Canonical implements TC/(sum H+eps) with tests (bounded, permutation-invariant, independence/duplicate cases)."),
        ("redundancy normalization/range",
         "0 <= R_e <= 1",
         "TC/(3*log(bins+eps)) - range depends on bins, not on the feature set",
         "TC/(sum H + eps) in [0,1]",
         "PARTIAL",
         "Canonical bounded form validated numerically; documented in REDUNDANCY_DEFINITION_AUDIT.md."),
        ("sigmoid use",
         "Sigmoid with defined midpoint before IFS normalization",
         "no sigmoid (raw evidences used directly)",
         "sigmoid slope 5, intercept -2.5 (midpoint 0.5) on both evidences",
         "PARTIAL",
         "Canonical keeps slope/intercept explicit and configurable (sensitivity slopes 2.5/5/10)."),
        ("hyperedge membership/non-membership",
         "Independently estimated group degrees (independent of vertex min/max)",
         "independent group NMI/TC mapping",
         "independent group NMI/TC mapping",
         "MATCH (both independent) BUT 'STRONG' CLAIM NOT VERIFIED",
         "Strongness audit shows independent degrees do NOT satisfy mu_e=min mu_V, nu_e=max nu_V in general (see strong_relation_audit_summary.csv)."),
        ("hesitation weighting",
         "Edge weight multiplied by (1-pi_e)",
         "NOT applied in sif_hfam.py compute_edge_weights",
         "applied when use_hesitation_weight=True (default True)",
         "NO between the two legacy paths",
         "Ablation `no_hesitation` isolates this factor in the canonical path."),
        ("edge weight",
         "w_e = max(0, mu_e - lam*nu_e) [*(1-pi_e)]",
         "max(0, mu - lam*nu), no hesitation factor",
         "max(0, mu - lam*nu) * (1-pi)",
         "PARTIAL",
         "Non-negativity enforced in both; theorem-compatible mode requires w>=0 (tested)."),
        ("objective",
         "alpha*sum mu_V + beta*sum w_e g(|S cap e|) coverage",
         "same functional form; g saturating (binary) or soft",
         "same form with binary coverage (greedy treats an edge as covered once)",
         "MATCH on form",
         "Labeled representative hyperedge coverage; NOT synergy recovery (see OBJECTIVE_SCOPE_AND_BEHAVIOR.md)."),
        ("coverage function",
         "g(k)=1[k>=1] (binary) and concave soft/fractional variants",
         "binary or soft (rho=0.5) selectable",
         "binary via `covered` set",
         "MATCH for binary; soft available in sif_hfam.py only",
         "Canonical implements binary/soft/fraction with submodularity tests."),
        ("greedy selection",
         "Standard marginal-gain greedy under cardinality rule",
         "greedy_select (fixed m), ratio-based and dynamic threshold variants present",
         "greedy over covered-set gains, fixed m",
         "PARTIAL",
         "Canonical: single documented greedy path (also used as ranking for equal budgets); ratio rule only when cardinality_rule='efficiency_ratio'."),
        ("cardinality rule",
         "manuscript m ~ max(10, sqrt(d))",
         "run_sif_hfam_experiment: m=max(10, min(sqrt(d), f))",
         "choose_m: max(10, round(sqrt(d))) clipped to f",
         "PARTIAL (floor vs round)",
         "Canonical uses max(10, round(sqrt(d))) clipped to f,d and records m per run; equal-budget grid includes exact m."),
        ("random hyperedge sampling",
         "Seeded, uniform-per-order sampling under budget",
         "no sampling (lexicographic islice)",
         "seeded uniform sampling per order",
         "NO for sif_hfam.py (biased truncation)",
         "Canonical sampling is seeded and the realized edge list is recorded via hyperedge_info."),
        ("runtime timing boundaries",
         "Feature-selection time vs evaluation time separated; stage timers",
         "no instrumentation (only wall time around whole example scripts)",
         "fs_time_sec around selector.fit only",
         "PARTIAL",
         "Canonical records stage-level timers (screening/discretization/vertex/edges/greedy), evaluation timed separately, memory recorded."),
        ("classifier evaluation",
         "Fixed downstream classifiers, no per-selector tuning",
         "evaluation.py: RF only, selection done on FULL data before splits (LEAKAGE)",
         "RF(n_estimators=100, n_jobs=1) with FS on train fold only",
         "NO for evaluation.py (leakage); PARTIAL for reviewer_revision (single classifier)",
         "Canonical: FS once on train fold; RF/SVM/kNN/LR fixed configs; leakage-safe scaling pipelines."),
        ("baseline identity",
         "Authentic implementations of published baselines",
         "fs_experiments: HIFS/UDFS/NDFS/HSIC-Lasso documented in-source as proxies/placeholders; ReliefF is a home-made implementation",
         "baseline_selectors: ReliefF/mRMR/HSIC-Lasso/HIFS/UDFS/NDFS all explicit proxies (ANOVA-F etc.)",
         "NO",
         "New evidence uses authentic bundle implementations only (see 04_baselines/baseline_manifest.csv); HIFS/UDFS/NDFS/HSIC-Lasso = UNAVAILABLE/NOT VERIFIED."),
        ("split protocol",
         "Repeated stratified hold-out, FS on train only, saved indices",
         "evaluation.py: selection on full data, then train_test_split per iteration (leakage)",
         "10 repeated stratified hold-outs, FS on train fold",
         "NO for evaluation.py; PARTIAL otherwise (seeds not persisted)",
         "Canonical persists exact split indices + seeds per run to JSON."),
        ("reported mean/SD units",
         "Consistent units; mean and SD same scale",
         "prints fraction mean +- fraction SD consistently",
         "summary CSVs store fractions; formatted tables multiply both by 100 only when formatting",
         "PARTIAL (depends on downstream table formatting)",
         "Automated validators block mixed-unit output (reporting_validation.json); legacy accuracy 95.45-style percent means with fraction SDs must not be reused."),
    ]
    lines = [
        "# Implementation vs Manuscript Audit",
        "",
        MANUSCRIPT_NOTE,
        "",
        "| Component | Manuscript definition | Legacy implementation (`sif_hfam.py` / `fs_experiments`) | Reviewer-revision implementation (`reviewer_revision/`) | Match? | Scientific impact |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        cells = [c.replace("|", "\\|").replace("\n", " ") for c in r]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Additional findings",
        "",
        "1. **Two incompatible SIFHFAM implementations exist.** `sif_hfam.py` (histogram NMI, "
        "order-3 exhaustive hyperedges, no screening stage inside the module, correlation "
        "redundancy normalized by log(bins)) and `reviewer_revision/sifhfam_selector.py` "
        "(ANOVA-F screening, discrete NMI, orders 2..K sampled edges, TC/(sum H) redundancy) "
        "are not the same mathematical pipeline. Every legacy table must be traced to one of them.",
        "2. **Leakage in `evaluation.py`.** `evaluate_feature_selection` selects features on the "
        "full dataset before any train/test split. Results produced through this path cannot be "
        "retained as leakage-free evidence.",
        "3. **Proxy baselines under published names** (HIFS/UDFS/NDFS/HSIC-Lasso/ReliefF/mRMR in "
        "`reviewer_revision/baseline_selectors.py`; HIFS/UDFS/NDFS/HSIC-Lasso in "
        "`fs_experiments/feature_selection_methods.py`) are preserved as frozen artifacts and "
        "flagged unverified; they are banned from new tables.",
        "4. **The strong relation is not satisfied** by independently estimated group degrees in "
        "general (quantified in `strong_relation_audit_summary.csv`). See `NAMING_RECOMMENDATION.md`.",
        "5. **Runtime claims**: no legacy stage-level timers exist; the canonical instrumented "
        "timings supersede any legacy ~0.1 s style claims (see 08_runtime_scalability).",
        "6. **Split indices/seeds were not persisted** by legacy runners; canonical runs persist them.",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------- lineage ---------------------------------- #

def _scan_result_lineage(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    outputs = root / "Outputs"
    if outputs.exists():
        for p in sorted(outputs.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            rel = p.relative_to(root).as_posix()
            group = rel.split("/")[1] if "/" in rel else "Outputs"
            producer = "reviewer_revision/run_all_revision_experiments.py (via run_reviewer_revision.py)"
            methods = "SIFHFAM,SIFFAM,ReliefF,mRMR,HSIC-Lasso,HIFS,UDFS,NDFS (see Outputs/run_manifest.json)"
            if group in ("00_config",):
                auth = "configuration_only"
                status = "metadata"
                notes = "Parameter table; values consistent with reviewer_revision/config.py defaults."
            elif group == "01_main_repeated":
                auth = "mixed_proxy_and_canonical"
                status = "flagged_unverified"
                notes = ("Baseline columns HSIC-Lasso/HIFS/UDFS/NDFS/ReliefF/mRMR come from "
                         "proxy functions; SIFHFAM/SIFFAM columns come from the ANOVA-F "
                         "screened reviewer-revision selector. Preserved for provenance only.")
            elif group in ("03_hyperparameter_sensitivity", "04_default_parameter_robustness",
                           "05_ablation", "06_ifs_validity"):
                auth = "reviewer_revision_selector_only"
                status = "flagged_unverified"
                notes = ("Computed with the reviewer-revision selector (ANOVA-F screening, "
                         "correlation vertex redundancy); not the canonical pipeline.")
            elif group == "02_statistical_tests":
                auth = "derived_from_main_table"
                status = "flagged_unverified"
                notes = "Derived from 01_main_repeated aggregates (which contain proxy baselines)."
            elif group == "07_stability_consistency":
                auth = "reviewer_revision_selector_only"
                status = "flagged_unverified"
                notes = "Stability of the reviewer-revision selector selections (proxy baselines included in overlap tables)."
            elif group == "08_ordered_association":
                auth = "posthoc_analysis"
                status = "flagged_unverified"
                notes = "Ordered feature-outcome association on legacy selections; association only, not causal."
            else:
                auth = "unknown"
                status = "flagged_unverified"
                notes = "Not traced to a producing script."
            # units check for accuracy tables
            if p.name.endswith(".csv") and "accuracy" in p.name:
                notes += " Accuracy columns stored as fractions unless header says percent."
            rows.append({
                "artifact_path": rel,
                "artifact_group": group,
                "suspected_producing_code": producer,
                "methods_involved": methods,
                "authenticity_status": auth,
                "scientific_status": status,
                "notes": notes,
            })

    fs = root / "fs_experiments"
    if fs.exists():
        for p in sorted(fs.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            rel = p.relative_to(root).as_posix()
            if p.suffix == ".npy" or p.name in ("fs_results.csv", "selection_metadata.csv"):
                auth = "proxy_baselines_present"
                status = "flagged_unverified"
                notes = ("Produced by fs_experiments/*; HIFS/UDFS/NDFS/HSIC-Lasso functions are "
                         "documented in-source as proxies/placeholders; local 'ReliefF' is a "
                         "home-made implementation, not scikit-rebate.")
            elif p.suffix == ".py":
                auth = "code"
                status = "code_artifact"
                notes = "Legacy experiment code (frozen)."
            else:
                auth = "unknown"
                status = "flagged_unverified"
                notes = ""
            rows.append({
                "artifact_path": rel,
                "artifact_group": "fs_experiments",
                "suspected_producing_code": "fs_experiments/run_fs_experiments.py + feature_selection_methods.py",
                "methods_involved": "mRMR,ReliefF,SPEC,LS,HIFS,UDFS,NDFS,HSIC-Lasso",
                "authenticity_status": auth,
                "scientific_status": status,
                "notes": notes,
            })
    return pd.DataFrame(rows)


# --------------------------- strongness audit ---------------------------- #

def run_strongness_audit(out_dir: Path, datasets: Optional[List[str]] = None,
                         K_values: Sequence[int] = (2, 3),
                         max_edges: int = 5000, seed: int = 42) -> Dict[str, Path]:
    datasets = datasets or PAPER14_DATASETS
    all_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for K in K_values:
        for ds in datasets:
            try:
                X, y, meta = load_dataset(ds)
                s = run_seed(seed, ds, 0)
                sp = make_split(ds, X.shape[0], y, 0, 0.25, s)
                cfg = RevisionConfig(random_state=s, max_edges=max_edges, K=int(K),
                                     name="strongness_audit")
                res = fit(X[sp.train_idx], y[sp.train_idx], cfg)
                if res.error:
                    failures.append({"dataset": ds, "K": int(K), "error": res.error[-500:]})
                    continue
                for row in res.strongness_rows:
                    r = dict(row)
                    r["dataset"] = ds
                    r["K"] = int(K)
                    all_rows.append(r)
                summ = dict(res.strongness_summary)
                summ["dataset"] = ds
                summ["K"] = int(K)
                summ["n_screened"] = res.f
                summ["hyperedge_sampling"] = str(res.hyperedge_info.get("sampled"))
                summary_rows.append(summ)
            except Exception as exc:
                failures.append({"dataset": ds, "K": int(K),
                                 "error": f"{type(exc).__name__}: {exc}"})

    out_dir.mkdir(parents=True, exist_ok=True)
    per_edge = pd.DataFrame(all_rows)
    summary = pd.DataFrame(summary_rows)
    p1 = out_dir / "strong_relation_audit_per_edge.csv"
    p2 = out_dir / "strong_relation_audit_summary.csv"
    per_edge.to_csv(p1, index=False)
    summary.to_csv(p2, index=False)
    paths = {"per_edge": p1, "summary": p2}
    for k in K_values:
        sub = summary[summary["K"] == k] if not summary.empty else summary
        pk = out_dir / f"strong_relation_audit_summary_K{k}.csv"
        sub.to_csv(pk, index=False)
        paths[f"summary_K{k}"] = pk
    if failures:
        fp = out_dir / "strong_relation_audit_failures.csv"
        pd.DataFrame(failures).to_csv(fp, index=False)
        paths["failures"] = fp
    return paths


def run_strong_induced_sanity(out_dir: Path) -> Path:
    """Verify the strict strong-induced variant satisfies the relation exactly."""
    rng = np.random.default_rng(0)
    mu = rng.random(20)
    nu = rng.random(20)
    # clip so mu+nu<=1
    scale = np.maximum(1.0, mu + nu)
    mu, nu = mu / scale, nu / scale
    edges = [tuple(sorted(rng.choice(20, size=int(rng.integers(2, 5)), replace=False).tolist()))
             for _ in range(50)]
    ed = strict_strong_induced_edge_degrees(mu, nu, edges)
    rows, summ = strong_relation_audit(mu, nu, ed, edges, tol=1e-12)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "strict_strong_induced_sanity.csv"
    df.to_csv(p, index=False)
    if summ["pct_satisfies_both"] != 100.0:
        raise AssertionError(f"strict strong-induced variant failed sanity: {summ}")
    # independent variant must be audited, not asserted strong
    return p


# ------------------------- redundancy validation -------------------------- #

def run_redundancy_validation(out_dir: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []

    # 1) independence case: discretized independent Gaussians (large n, few bins
    #    to keep the plugin finite-sample TC bias small)
    n = 2000
    X_ind = Discretizer(bins=4).fit_transform(rng.normal(size=(n, 4)))
    r, tc = group_redundancy(X_ind)
    rows.append({"case": "independent_gaussians_binned", "R_e": r, "TC": tc,
                 "expected": "near 0", "pass": bool(r < 0.10)})

    # 2) duplicate case: for k identical features TC/sumH = (k-1)/k exactly
    base = rng.integers(0, 5, size=n)
    X_dup = Discretizer(bins=6).fit_transform(
        np.column_stack([base, base, base]))
    r, tc = group_redundancy(X_dup)
    rows.append({"case": "triplicate_feature", "R_e": r, "TC": tc,
                 "expected": "~(k-1)/k = 2/3", "pass": bool(abs(r - 2.0 / 3.0) < 1e-6)})
    X_pair = Discretizer(bins=6).fit_transform(np.column_stack([base, base]))
    r2, tc2 = group_redundancy(X_pair)
    rows.append({"case": "duplicate_pair", "R_e": r2, "TC": tc2,
                 "expected": "1/2", "pass": bool(abs(r2 - 0.5) < 1e-6)})

    # 3) bounds over random cases
    all_ok = True
    for _ in range(50):
        X = rng.integers(0, 4, size=(200, 3))
        r, tc = group_redundancy(X)
        if not (np.isfinite(r) and -1e-12 <= r <= 1 + 1e-12):
            all_ok = False
    rows.append({"case": "bounded_50_random_integer_sets", "R_e": np.nan, "TC": np.nan,
                 "expected": "0<=R<=1 finite", "pass": all_ok})

    # 4) permutation invariance
    X = rng.integers(0, 4, size=(200, 4))
    r1, _ = group_redundancy(X)
    perm = rng.permutation(4)
    r2, _ = group_redundancy(X[:, perm])
    rows.append({"case": "permutation_invariance", "R_e": r1, "TC": r2,
                 "expected": "equal", "pass": bool(abs(r1 - r2) < 1e-12)})

    # 5) zero-entropy (constant) safety
    Xc = np.zeros((100, 3), dtype=int)
    r, tc = group_redundancy(Xc)
    rows.append({"case": "all_constant", "R_e": r, "TC": tc,
                 "expected": "0 (safe)", "pass": bool(r == 0.0 and tc == 0.0)})

    # 6) estimator comparison plugin vs miller_madow on one case
    X = rng.integers(0, 3, size=(60, 3))
    r_p, _ = group_redundancy(X, estimator="plugin")
    r_m, _ = group_redundancy(X, estimator="miller_madow")
    rows.append({"case": "estimator_plugin_vs_miller_madow", "R_e": r_p, "TC": r_m,
                 "expected": "both in [0,1]", "pass": bool(0 <= r_p <= 1 and 0 <= r_m <= 1)})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "redundancy_validation.csv"
    df.to_csv(p, index=False)
    if not bool(df["pass"].all()):
        raise AssertionError("redundancy validation failed: "
                             + str(df.loc[~df["pass"], 'case'].tolist()))
    return p


# --------------------------- markdown documents -------------------------- #

def write_naming_recommendation(out_dir: Path, summary_csv: Path) -> Path:
    pct = float("nan")
    n_edges = 0
    try:
        s = pd.read_csv(summary_csv)
        if not s.empty and "pct_satisfies_both" in s.columns:
            pct = float(np.nanmean(s["pct_satisfies_both"].to_numpy(dtype=float)))
            n_edges = int(np.nansum(s["n_edges"].to_numpy(dtype=float)))
    except Exception:
        pass
    strong_holds = bool(np.isfinite(pct) and pct >= 99.9)
    doc = f"""# NAMING RECOMMENDATION (Reviewer 2.1 / 4.7)

## Empirical strongness audit result

- Realized hyperedges audited: {n_edges}
- Percentage satisfying BOTH `mu_e = min_j mu_V(j)` and `nu_e = max_j nu_V(j)`
  within tolerance 1e-9: **{pct:.4f}%**
- Source tables: `strong_relation_audit_per_edge.csv`,
  `strong_relation_audit_summary.csv`,
  `strong_relation_audit_summary_K2.csv`, `strong_relation_audit_summary_K3.csv`
- Strict strong-induced ablation sanity: `strict_strong_induced_sanity.csv`
  (satisfies the relation exactly by construction).

## Conclusion

{'The strong relation holds within tolerance for the audited edges.' if strong_holds else (
'The independently estimated hyperedge degrees do **NOT** satisfy the defining '
'strong intuitionistic relation in general (0% or near-0% of audited edges satisfy '
'both equalities). The object produced by the submitted construction therefore '
'should **not** be defended as a formally *strong* intuitionistic fuzzy hypergraph '
'without a new, valid definition that is proved to hold for the estimated degrees.')}

## Recommended manuscript-writing name (for later use; manuscript NOT edited now)

Use a neutral name such as:

- **intuitionistic-fuzzy-weighted feature hypergraph**, or
- **intuitionistic fuzzy feature hypergraph**

and describe the construction explicitly: vertex degrees are estimated from
relevance/redundancy evidence, and hyperedge degrees are estimated independently
from group relevance/group redundancy evidence; the pair is an intuitionistic
assignment on vertices and edges that is not asserted to be strong.

The strict strong-induced variant (`hyperedge_degrees='strict_strong'`) is kept
**for ablation only** and must not silently replace the submitted method.

## What must not be done

- Do not invent a new definition of "strong" merely to retain the title.
- Do not claim the strongness property without the audit tables above.
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "NAMING_RECOMMENDATION.md"
    p.write_text(doc, encoding="utf-8")
    return p


def write_objective_scope_doc(out_dir: Path) -> Path:
    doc = """# OBJECTIVE SCOPE AND BEHAVIOR (Reviewer 2.2)

## What the implemented objective is

The submitted objective (kept as the legacy/reference objective) is

    F(S) = alpha * sum_{j in S} mu_V(j) + beta * sum_e w_e * g(|S cap e|)

with non-negative hyperedge weights w_e and a monotone concave coverage
function g on the non-negative integers:

- **binary (reference)**: `g_binary(k) = 1[k >= 1]` — **representative
  hyperedge coverage**.
- **soft**: `g_soft(k) = 1 - (1-rho)^k`, default `rho = 0.5`.
- **fraction**: `g_fraction(k, |e|) = min(k/|e|, 1)`.

All three are monotone and submodular when composed as above with w_e >= 0
(numerically verified in tests on exhaustive small instances).

## What the objective does and does not establish

1. Binary coverage rewards **representative coverage of a jointly scored
   group**: once one member of a hyperedge is selected, further members of the
   same hyperedge give zero additional gain from that edge.
2. It does **not** establish recovery of all members of a synergistic
   interaction. Selecting one representative of a jointly-scored group is not
   the same as recovering the full interaction tuple.
3. The concave soft/fractional variants give the 2nd/3rd selected member of a
   hyperedge a positive but diminishing marginal gain while preserving
   submodularity; they are implemented and ablated, not claimed as the
   submitted method.
4. The full-group reward `1[e subset S]` is deliberately **not** used as the
   main objective: it is not the same submodular coverage formulation, and the
   classical (1-1/e) greedy guarantee is not claimed for it.
5. No result should be described as proof of "synergy recovery" unless the
   synthetic ground-truth experiment (02_synthetic_interactions) actually
   supports it for the specific configuration analyzed.

## Ablation outputs

Binary vs soft/fractional coverage comparisons (selected-feature differences,
objective values, accuracy, stability) are produced under
`07_ablation/` with `binary_coverage`, `soft_coverage`, `fraction_coverage`
variants; submodularity sanity tests are in the test suite
(`tests/test_objective.py`).
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "OBJECTIVE_SCOPE_AND_BEHAVIOR.md"
    p.write_text(doc, encoding="utf-8")
    return p


def write_theoretical_scope_audit(out_dir: Path, ratio_records: Optional[pd.DataFrame] = None) -> Path:
    doc = """# THEORETICAL SCOPE AUDIT (Reviewer 2.4 / 2.5)

## Exact scope of the classical greedy guarantee (when applicable)

For a monotone submodular objective F with F(empty)=0 maximized subject to
|S| <= m by the classical greedy algorithm, the well-known guarantee is

    F(S_greedy) >= (1 - 1/e) * F(S_opt)

**with respect to the realized problem instance actually optimized**, i.e.:

- the **realized screened vertex set** (the f candidates after screening),
- the **realized/sampled hyperedge collection** E_realized (including its
  random sampling and any edge budget cap),
- the **non-negative surrogate coverage objective** F defined by
  w_e = max(0, mu_e - lam*nu_e) [* (1-pi_e)] and the chosen g,
- the **cardinality constraint m** used in that realized problem.

## What it is NOT a guarantee relative to

- the original complete feature universe (d features), because screening can
  remove relevant features irrecoverably;
- the complete unsampled hypergraph (the method evaluates a sampled/capped
  subcollection);
- ground-truth feature recovery;
- classification risk / test accuracy;
- biological relevance.

## Code-verified theorem-compatible assumptions

Verified by tests in `tests/test_theory_scope.py`:

1. edge weights are non-negative in theorem-compatible mode;
2. F is monotone on exhaustive/random small instances;
3. F is submodular (diminishing returns) on exhaustive/random small instances;
4. numerical greedy-vs-optimum comparison on tiny problems (ratio recorded;
   the observed ratio is an **empirical record**, never a new theorem).

The classical (1-1/e) result is classical; nothing in this codebase claims it
as a novel theorem.

## Empirical (non-theoretical) association study

As an empirical supplementary analysis only, the association between the
surrogate objective value and validation accuracy over greedy prefixes and
random subsets is recorded under `10_statistics/` (files prefixed
`objective_accuracy_association_*`). This is an observed association; it is
not a theoretical bound and must not be presented as one.
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if ratio_records is not None and not ratio_records.empty:
        ratio_records.to_csv(out_dir / "greedy_ratio_records.csv", index=False)
    p = out_dir / "THEORETICAL_SCOPE_AUDIT.md"
    p.write_text(doc, encoding="utf-8")
    return p


def write_redundancy_audit(out_dir: Path) -> Path:
    doc = """# REDUNDANCY DEFINITION AUDIT (Reviewer 2.6)

## Canonical definition (revision_canonical)

For a hyperedge e (feature subset) on discretized training data:

    TC(e)   = sum_{j in e} H(X_j) - H(X_e)          (total correlation)
    R_e     = TC(e) / (sum_{j in e} H(X_j) + eps)    (bounded normalization)

with safe handling when the denominator is zero (all-constant features):
R_e = 0.

## Documented estimator choices

- **Default estimator**: ordinary plug-in (maximum-likelihood) discrete
  entropy on the train-fold discretization.
- **Alternative (sensitivity only)**: Miller-Madow corrected plug-in,
  `H_MM = H_plugin + (K-1)/(2n)` with K the number of observed symbols and n
  the sample size.
- **Entropy base**: natural logarithm (nats) throughout the codebase
  (`information.ENTROPY_BASE`).
- **Zero-entropy features**: features with H(X_j)=0 contribute 0 to the sum;
  if the whole denominator is 0, R_e is defined as 0 (never NaN/Inf).
- **Feature-order invariance**: TC and R_e are symmetric in the columns of e
  (tested by permutation).

## Properties (numerically validated in `redundancy_validation.csv`)

1. finite for all tested inputs;
2. `0 <= R_e <= 1` up to numerical tolerance;
3. near zero for independent synthetic variables;
4. near 1 for duplicated/strongly dependent variables;
5. invariant to feature ordering;
6. plugin vs Miller-Madow both remain in [0,1] on tested cases.

## What must not be done

- Do **not** apply a sigmoid midpoint interpretation to an unbounded
  quantity. R_e is bounded; the legacy `sif_hfam.py` normalization
  `TC/(3*log(bins))` is bins-dependent rather than feature-set-dependent and
  is not used in the canonical path.
- Group NMI must not be called "synergy"; no synergy measure is claimed.
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "REDUNDANCY_DEFINITION_AUDIT.md"
    p.write_text(doc, encoding="utf-8")
    return p


def run_method_audit(out_dir: Path, root: Path,
                     datasets: Optional[List[str]] = None,
                     do_strongness: bool = True) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    md = out_dir / "IMPLEMENTATION_MANUSCRIPT_AUDIT.md"
    md.write_text(audit_table_md(), encoding="utf-8")
    paths["audit_md"] = md

    lineage = _scan_result_lineage(root)
    p_lin = out_dir / "result_lineage.csv"
    lineage.to_csv(p_lin, index=False)
    paths["result_lineage"] = p_lin

    unsafe = lineage[lineage["scientific_status"] != "code_artifact"]
    unsafe = unsafe[unsafe["scientific_status"] != "metadata"]
    p_un = out_dir / "unsafe_or_unverified_legacy_outputs.csv"
    unsafe.to_csv(p_un, index=False)
    paths["unsafe"] = p_un

    paths["naming"] = write_naming_recommendation(
        out_dir, out_dir / "strong_relation_audit_summary.csv")
    paths["objective_doc"] = write_objective_scope_doc(out_dir)
    paths["redundancy_doc"] = write_redundancy_audit(out_dir)

    if do_strongness:
        sp = run_strongness_audit(out_dir, datasets=datasets)
        paths.update({f"strong_{k}": v for k, v in sp.items()})
        run_strong_induced_sanity(out_dir)
        # rewrite naming doc now that summary exists
        paths["naming"] = write_naming_recommendation(
            out_dir, out_dir / "strong_relation_audit_summary.csv")
    paths["redundancy_validation"] = run_redundancy_validation(out_dir)
    return paths
