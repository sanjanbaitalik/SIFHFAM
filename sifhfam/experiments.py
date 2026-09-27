"""Experiment orchestration for the Neurocomputing revision evidence package.

Every section writes into its own folder under SIFHFAM_EVIDENCE/,
checkpoints incrementally (append + resume), and never touches frozen legacy
outputs.
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .baselines import run_baseline, write_manifest_csv
from .config import (
    PAPER14_DATASETS,
    REPRESENTATIVE_DATASETS_PRESELECTED,
    RevisionConfig,
    budget_grid,
    choose_f,
    choose_m,
    manuscript_m,
    profile_grids,
)
from .data import Split, load_dataset, make_split, run_seed, save_splits
from .evaluation import CLASSIFIER_NAMES, evaluate_all_classifiers, evaluate_selected
from .objective import Objective, exhaustive_greedy_ratio, greedy_path, is_monotone, is_submodular
from .reporting import (
    assert_reporting_ok,
    run_validations,
    write_reporting_validation,
)
from .selector import SelectionResult, fit
from .stability import consensus_select, jaccard, kuncheva, nogueira_stability, pairwise_summary
from .synthetic import (
    export_interaction_example_table,
    generate,
    marginal_association_check,
    recovery_metrics,
)


# ----------------------------- utilities -------------------------------- #

def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: object, path: Path) -> None:
    _ensure(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def read_csv_or_none(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def append_rows(path: Path, rows: List[Dict[str, object]],
                key_cols: Sequence[str]) -> int:
    """Append rows to CSV, skipping keys already present (resume support)."""
    if not rows:
        return 0
    df_new = pd.DataFrame(rows)
    existing = read_csv_or_none(path)
    n_added = 0
    if existing is not None and not existing.empty and set(key_cols) <= set(existing.columns):
        keys_exist = set(map(tuple, existing[list(key_cols)].astype(str).to_numpy()))
        keep = []
        for _, r in df_new.iterrows():
            k = tuple(str(r[c]) for c in key_cols)
            if k not in keys_exist:
                keys_exist.add(k)
                keep.append(r)
        if not keep:
            return 0
        df_new = pd.DataFrame(keep)
    header = not path.exists()
    _ensure(path.parent)
    df_new.to_csv(path, mode="a", header=header, index=False)
    return len(df_new)


def load_existing_keys(path: Path, key_cols: Sequence[str]) -> set:
    """Set of existing composite keys for cheap resume checks."""
    df = read_csv_or_none(path)
    if df is None or df.empty or not set(key_cols) <= set(df.columns):
        return set()
    return set(map(tuple, df[list(key_cols)].astype(str).to_numpy()))


def summarize(raw: pd.DataFrame, group_cols: Sequence[str],
              value_cols: Sequence[str]) -> pd.DataFrame:
    aggs = {c: ["mean", "std", "min", "max"] for c in value_cols}
    out = raw.groupby(list(group_cols)).agg(aggs)
    out.columns = ["_".join(col).strip("_") for col in out.columns.values]
    return out.reset_index()


def selection_from_result(res: SelectionResult, budget: int) -> np.ndarray:
    order = res.greedy_order_screened
    mapped = res.screened_idx[order] if len(order) else np.array([], dtype=int)
    return mapped[: int(budget)].astype(int)


def failure_log(path: Path, section: str, where: str, message: str) -> None:
    _ensure(path.parent)
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "section": section,
           "where": where, "message": message[:4000]}
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


# ------------------------------ context --------------------------------- #

@dataclass
class RunContext:
    root: Path
    evidence: Path
    profile: str
    datasets: List[str]
    n_runs: int
    seed: int
    grids: Dict[str, object]
    test_size: float = 0.25
    data_dir: Optional[Path] = None   # external dataset directory (never bundled)

    @classmethod
    def create(cls, root: Path, profile: str, datasets: Optional[Sequence[str]] = None,
               n_runs: Optional[int] = None, seed: int = 42,
               data_dir: Optional[Path] = None) -> "RunContext":
        grids = profile_grids(profile)
        ds = list(datasets) if datasets else list(PAPER14_DATASETS)
        runs = int(n_runs if n_runs is not None else grids["n_runs"])
        resolved = (Path(data_dir).expanduser().resolve() if data_dir
                    else root / "Updated Dataset")
        return cls(root=root, evidence=root / "SIFHFAM_EVIDENCE",
                   profile=profile, datasets=ds, n_runs=runs, seed=seed, grids=grids,
                   data_dir=resolved)

    def load_dir(self) -> Path:
        """Directory containing <name>_X.csv / <name>_Y.csv pairs."""
        return self.data_dir if self.data_dir is not None else self.root / "Updated Dataset"

    def out(self, name: str) -> Path:
        return _ensure(self.evidence / name)

    def cfg(self, **kw) -> RevisionConfig:
        return RevisionConfig(random_state=self.seed, **kw)


# --------------------------- main + equal budget ------------------------- #

RANKING_METHODS_CORE = ["SIFHFAM_canonical", "SIFFAM_vertex_only",
                        "mRMR", "ReliefF", "CMIM", "JMIM"]
NATIVE_METHODS = ["FCBF", "FRFS", "PPFS"]


def _fit_canonical_with_path(X, y, cfg: RevisionConfig, max_budget: int) -> SelectionResult:
    return fit(X, y, cfg, max_path_budget=int(max_budget))


def run_equal_budget(ctx: RunContext) -> Dict[str, Path]:
    out = ctx.out("03_equal_budget")
    raw_path = out / "equal_budget_per_run.csv"
    fails = out / "failure_log.csv"
    write_json({
        "profile": ctx.profile, "n_runs": ctx.n_runs, "seed": ctx.seed,
        "datasets": ctx.datasets,
        "budget_grid_rule": "unique(sorted(clipped([10,25,50,100,manuscript_m]))), "
                            "values > d or > screened ranking length f removed",
        "ranking_methods": RANKING_METHODS_CORE,
        "evaluation": "RF(n_estimators=100, n_jobs=1), FS on train fold only",
    }, out / "config.json")
    splits_meta: List[dict] = []

    ranking_methods = list(RANKING_METHODS_CORE)
    expensive = set(ctx.grids.get("expensive_baselines", []))
    if "QuickSelection" in expensive:
        ranking_methods.append("QuickSelection")

    rows: List[Dict[str, object]] = []
    eb_keys = load_existing_keys(raw_path, ["dataset", "run", "method", "budget"])
    for ds in ctx.datasets:
        try:
            X, y, meta = load_dataset(ds, ctx.load_dir())
        except Exception as exc:
            failure_log(fails, "equal_budget", ds, f"load failed: {exc}")
            continue
        d = X.shape[1]
        base_cfg = RevisionConfig(random_state=ctx.seed, name="SIFHFAM_canonical")
        f_ref = choose_f(d, base_cfg)
        budgets = [b for b in budget_grid(d) if b <= f_ref]
        if not budgets:
            budgets = [min(manuscript_m(d), f_ref)]
        max_budget = max(budgets)
        m_manu = min(manuscript_m(d), f_ref)

        for run in range(ctx.n_runs):
            def _done(method: str) -> bool:
                return all((ds, str(run), method, str(b)) in eb_keys for b in budgets)

            seed_r = run_seed(ctx.seed, ds, run)
            sp = make_split(ds, X.shape[0], y, run, ctx.test_size, seed_r)
            splits_meta.append(sp.to_dict())
            Xtr, ytr = X[sp.train_idx], y[sp.train_idx]
            Xte, yte = X[sp.test_idx], y[sp.test_idx]

            # --- canonical + vertex-only (one fit each) ---
            for method, over in (
                ("SIFHFAM_canonical", {}),
                ("SIFFAM_vertex_only", {"use_hyperedges": False, "beta": 0.0,
                                        "K": 1, "name": "SIFFAM_vertex_only"}),
            ):
                if _done(method):
                    continue
                cfg = RevisionConfig(**{**{"random_state": ctx.seed, "name": method}, **over})
                try:
                    res = _fit_canonical_with_path(Xtr, ytr, cfg, max_budget)
                    if res.error:
                        failure_log(fails, "equal_budget", f"{ds}/{run}/{method}", res.error[-800:])
                        continue
                    for b in budgets:
                        sel = selection_from_result(res, b)
                        ev = evaluate_selected(Xtr, ytr, Xte, yte, sel, "RF", seed_r)
                        rows.append({
                            "dataset": ds, "run": run, "method": method, "budget": b,
                            "manuscript_budget": bool(b == m_manu),
                            "n_selected": len(sel), "d": d, "f_screened": res.f,
                            "feature_reduction": 1.0 - len(sel) / d,
                            "accuracy": ev.accuracy,
                            "balanced_accuracy": ev.balanced_accuracy,
                            "macro_f1": ev.macro_f1,
                            "fs_seconds": res.fs_time_sec,
                            "eval_seconds": ev.eval_seconds,
                            "objective_value": res.objective_value,
                            "selected_indices": ";".join(map(str, sel)),
                            "screening_strategy": cfg.screening_strategy,
                            "K": cfg.K, "g_type": cfg.g_type,
                            "error": ev.error or "",
                        })
                except Exception as exc:
                    failure_log(fails, "equal_budget", f"{ds}/{run}/{method}",
                                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

            # --- authentic ranking baselines ---
            for method in ranking_methods:
                if method in ("SIFHFAM_canonical", "SIFFAM_vertex_only"):
                    continue
                if _done(method):
                    continue
                try:
                    out_b = run_baseline(
                        method, Xtr, ytr, max_budget, seed=seed_r,
                        quickselection_epochs=int(ctx.grids.get("quickselection_epochs", 3)),
                        workdir=out / "_quickselection_tmp",
                    )
                    if out_b.status != "OK" or out_b.ranking is None:
                        failure_log(fails, "equal_budget", f"{ds}/{run}/{method}",
                                    f"status={out_b.status}; {out_b.error or out_b.notes}")
                        # record NaN rows so summaries show the gap honestly
                        for b in budgets:
                            rows.append({
                                "dataset": ds, "run": run, "method": method, "budget": b,
                                "manuscript_budget": bool(b == m_manu),
                                "n_selected": 0, "d": d, "f_screened": np.nan,
                                "feature_reduction": np.nan,
                                "accuracy": np.nan, "balanced_accuracy": np.nan,
                                "macro_f1": np.nan, "fs_seconds": out_b.seconds,
                                "eval_seconds": np.nan, "objective_value": np.nan,
                                "selected_indices": "",
                                "screening_strategy": "", "K": np.nan, "g_type": "",
                                "error": (out_b.error or out_b.notes or out_b.status)[:500],
                            })
                        continue
                    ranking = np.asarray(out_b.ranking, dtype=int)
                    for b in budgets:
                        sel = ranking[:min(b, len(ranking))]
                        ev = evaluate_selected(Xtr, ytr, Xte, yte, sel, "RF", seed_r)
                        rows.append({
                            "dataset": ds, "run": run, "method": method, "budget": b,
                            "manuscript_budget": bool(b == m_manu),
                            "n_selected": len(sel), "d": d, "f_screened": np.nan,
                            "feature_reduction": 1.0 - len(sel) / d,
                            "accuracy": ev.accuracy,
                            "balanced_accuracy": ev.balanced_accuracy,
                            "macro_f1": ev.macro_f1,
                            "fs_seconds": out_b.seconds,
                            "eval_seconds": ev.eval_seconds,
                            "objective_value": np.nan,
                            "selected_indices": ";".join(map(str, sel)),
                            "screening_strategy": "", "K": np.nan, "g_type": "",
                            "notes": out_b.notes or "", "error": ev.error or "",
                        })
                except Exception as exc:
                    failure_log(fails, "equal_budget", f"{ds}/{run}/{method}",
                                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

            # checkpoint after each run
            if rows:
                append_rows(raw_path, rows,
                            ["dataset", "run", "method", "budget"])
                eb_keys.update((r["dataset"], str(r["run"]), r["method"], str(r["budget"]))
                               for r in rows)
                rows = []

    if rows:
        append_rows(raw_path, rows, ["dataset", "run", "method", "budget"])
    write_json(splits_meta, out / "split_indices.json")

    raw = read_csv_or_none(raw_path)
    paths = {"raw": raw_path}
    if raw is not None and not raw.empty:
        summary = summarize(raw, ["dataset", "method", "budget", "manuscript_budget"],
                            ["accuracy", "balanced_accuracy", "macro_f1",
                             "n_selected", "feature_reduction", "fs_seconds"])
        summary.to_csv(out / "equal_budget_summary.csv", index=False)

        # accuracy vs cardinality (dataset-level means)
        avc = (raw.groupby(["dataset", "method", "budget"], as_index=False)
               .agg(accuracy_mean=("accuracy", "mean"),
                    accuracy_std=("accuracy", "std"),
                    n_selected_mean=("n_selected", "mean"),
                    feature_reduction_mean=("feature_reduction", "mean")))
        avc.to_csv(out / "accuracy_vs_cardinality.csv", index=False)

        # pareto (n_selected vs accuracy) per dataset over method x budget means
        pareto_rows = []
        for ds, g in avc.groupby("dataset"):
            pts = g.dropna(subset=["accuracy_mean"]).copy()
            frontier = []
            for _, r in pts.iterrows():
                dominated = ((pts["n_selected_mean"] <= r["n_selected_mean"]) &
                             (pts["accuracy_mean"] >= r["accuracy_mean"]) &
                             ((pts["n_selected_mean"] < r["n_selected_mean"]) |
                              (pts["accuracy_mean"] > r["accuracy_mean"]))).any()
                if not dominated:
                    frontier.append({**r.to_dict(), "dataset": ds, "is_frontier": True})
            for _, r in pts.iterrows():
                pareto_rows.append({**r.to_dict(), "dataset": ds,
                                    "is_frontier": bool(not any(
                                        f["method"] == r["method"] and f["budget"] == r["budget"]
                                        for f in frontier))})
        pareto = pd.DataFrame(pareto_rows)
        pareto.to_csv(out / "accuracy_sparsity_pareto.csv", index=False)
        pareto[pareto["is_frontier"]].to_csv(out / "pareto_frontier_by_dataset.csv", index=False)
        paths.update({"summary": out / "equal_budget_summary.csv",
                      "avc": out / "accuracy_vs_cardinality.csv",
                      "pareto": out / "accuracy_sparsity_pareto.csv"})
    return paths


def run_native_cardinality(ctx: RunContext) -> Dict[str, Path]:
    """Native-cardinality table for subset-only methods (no fabricated ranking)."""
    out = ctx.out("03_equal_budget")
    raw_path = out / "native_cardinality_per_run.csv"
    fails = out / "failure_log.csv"
    native_runs = int({"smoke": 1, "core": 2, "full": 5}.get(ctx.profile, 2))
    methods = list(NATIVE_METHODS)
    rows: List[Dict[str, object]] = []
    nat_keys = load_existing_keys(raw_path, ["dataset", "run", "method"])
    write_json({"profile": ctx.profile, "native_runs": native_runs, "methods": methods,
                "note": "subset methods with no defensible ranking; run count recorded"},
               out / "native_config.json")

    for ds in ctx.datasets:
        try:
            X, y, meta = load_dataset(ds, ctx.load_dir())
        except Exception as exc:
            failure_log(fails, "native", ds, str(exc))
            continue
        d = X.shape[1]
        for run in range(native_runs):
            seed_r = run_seed(ctx.seed, ds, run)
            sp = make_split(ds, X.shape[0], y, run, ctx.test_size, seed_r)
            Xtr, ytr = X[sp.train_idx], y[sp.train_idx]
            Xte, yte = X[sp.test_idx], y[sp.test_idx]
            for method in methods:
                if (ds, str(run), method) in nat_keys:
                    continue
                try:
                    ob = run_baseline(method, Xtr, ytr,
                                      max_budget=manuscript_m(d), seed=seed_r)
                    subset = ob.native_subset
                    if ob.status != "OK" or subset is None:
                        failure_log(fails, "native", f"{ds}/{run}/{method}",
                                    f"{ob.status}: {ob.error}")
                        rows.append({"dataset": ds, "run": run, "method": method,
                                     "status": ob.status, "n_selected": 0, "d": d,
                                     "feature_reduction": np.nan, "accuracy": np.nan,
                                     "balanced_accuracy": np.nan, "macro_f1": np.nan,
                                     "fs_seconds": ob.seconds, "selected_indices": "",
                                     "error": (ob.error or "")[:400]})
                        continue
                    subset = np.asarray(subset, dtype=int)
                    ev = evaluate_selected(Xtr, ytr, Xte, yte, subset, "RF", seed_r)
                    rows.append({
                        "dataset": ds, "run": run, "method": method, "status": "OK",
                        "n_selected": len(subset), "d": d,
                        "feature_reduction": 1.0 - len(subset) / d,
                        "accuracy": ev.accuracy, "balanced_accuracy": ev.balanced_accuracy,
                        "macro_f1": ev.macro_f1, "fs_seconds": ob.seconds,
                        "selected_indices": ";".join(map(str, subset)),
                        "notes": ob.notes or "", "error": ev.error or "",
                    })
                except Exception as exc:
                    failure_log(fails, "native", f"{ds}/{run}/{method}",
                                f"{type(exc).__name__}: {exc}")
                if rows:
                    append_rows(raw_path, rows, ["dataset", "run", "method"])
                    nat_keys.update((r["dataset"], str(r["run"]), r["method"]) for r in rows)
                    rows = []
    if rows:
        append_rows(raw_path, rows, ["dataset", "run", "method"])
    raw = read_csv_or_none(raw_path)
    paths = {"raw": raw_path}
    if raw is not None and not raw.empty:
        s = summarize(raw, ["dataset", "method"],
                      ["accuracy", "n_selected", "feature_reduction", "fs_seconds"])
        s.to_csv(out / "native_cardinality_summary.csv", index=False)
        paths["summary"] = out / "native_cardinality_summary.csv"
    return paths


# ------------------------- classifier generality ------------------------- #

def run_classifier_generality(ctx: RunContext) -> Dict[str, Path]:
    out = ctx.out("05_classifier_generality")
    raw_path = out / "classifier_generality_per_run.csv"
    fails = out / "failure_log.csv"
    classifiers = list(ctx.grids.get("classifiers", CLASSIFIER_NAMES))
    write_json({"profile": ctx.profile, "n_runs": ctx.n_runs, "classifiers": classifiers,
                "note": "FS once on train fold; same selected features for all classifiers"},
               out / "config.json")
    rows: List[Dict[str, object]] = []
    cg_keys = load_existing_keys(raw_path, ["dataset", "run", "method", "classifier"])
    for ds in ctx.datasets:
        try:
            X, y, meta = load_dataset(ds, ctx.load_dir())
        except Exception as exc:
            failure_log(fails, "classifiers", ds, str(exc))
            continue
        d = X.shape[1]
        for run in range(ctx.n_runs):
            seed_r = run_seed(ctx.seed, ds, run)
            sp = make_split(ds, X.shape[0], y, run, ctx.test_size, seed_r)
            Xtr, ytr = X[sp.train_idx], y[sp.train_idx]
            Xte, yte = X[sp.test_idx], y[sp.test_idx]
            for method, over in (
                ("SIFHFAM_canonical", {}),
                ("SIFFAM_vertex_only", {"use_hyperedges": False, "beta": 0.0,
                                        "K": 1, "name": "SIFFAM_vertex_only"}),
            ):
                if all((ds, str(run), method, cl) in cg_keys for cl in classifiers):
                    continue
                cfg = RevisionConfig(**{**{"random_state": ctx.seed, "name": method}, **over})
                try:
                    res = fit(Xtr, ytr, cfg, max_path_budget=manuscript_m(d))
                    if res.error:
                        failure_log(fails, "classifiers", f"{ds}/{run}/{method}", res.error[-600:])
                        continue
                    sel = res.selected
                    for ev in evaluate_all_classifiers(Xtr, ytr, Xte, yte, sel,
                                                       classifiers, seed_r):
                        rows.append({
                            "dataset": ds, "run": run, "method": method,
                            "classifier": ev.classifier, "n_selected": len(sel), "d": d,
                            "accuracy": ev.accuracy,
                            "balanced_accuracy": ev.balanced_accuracy,
                            "macro_f1": ev.macro_f1,
                            "fs_seconds": res.fs_time_sec,
                            "eval_seconds": ev.eval_seconds,
                            "selected_indices": ";".join(map(str, sel)),
                            "error": ev.error or "",
                        })
                except Exception as exc:
                    failure_log(fails, "classifiers", f"{ds}/{run}/{method}",
                                f"{type(exc).__name__}: {exc}")
            if rows:
                append_rows(raw_path, rows, ["dataset", "run", "method", "classifier"])
                rows = []
    if rows:
        append_rows(raw_path, rows, ["dataset", "run", "method", "classifier"])
    raw = read_csv_or_none(raw_path)
    paths = {"raw": raw_path}
    if raw is not None and not raw.empty:
        s = summarize(raw, ["dataset", "method", "classifier"],
                      ["accuracy", "balanced_accuracy", "macro_f1", "n_selected"])
        s.to_csv(out / "classifier_generality_summary.csv", index=False)
        paths["summary"] = out / "classifier_generality_summary.csv"
    return paths


# ------------------------------- sensitivity ----------------------------- #

def run_sensitivity(ctx: RunContext) -> Dict[str, Path]:
    out = ctx.out("06_sensitivity")
    raw_path = out / "sensitivity_per_run.csv"
    fails = out / "failure_log.csv"
    datasets = list(ctx.grids.get("sensitivity_datasets", REPRESENTATIVE_DATASETS_PRESELECTED))
    datasets = [d for d in datasets if d in ctx.datasets] or ctx.datasets[:1]
    n_runs = min(ctx.n_runs, int({"smoke": 2, "core": 5, "full": 10}[ctx.profile]))
    g = ctx.grids
    # Representative dataset list chosen BEFORE results (config.REPRESENTATIVE_...)
    write_json({"profile": ctx.profile, "datasets_preselected": datasets,
                "n_runs": n_runs, "seed": ctx.seed,
                "factors": {"lambda": g["lambda_grid"], "ab": g["ab_grid"],
                            "K": g["K_grid"], "max_edges": g["max_edges_grid"],
                            "bins": g["bins_grid"], "discretizer": g["discretizer_grid"],
                            "sigmoid_slopes": g["slope_grid"],
                            "screening_budget_multipliers": g["screening_budget_multipliers"]}},
               out / "config.json")
    rows: List[Dict[str, object]] = []
    sens_keys = load_existing_keys(raw_path, ["dataset", "run", "factor", "value"])

    def eval_cfg(ds, run, X, y, factor, value, cfg):
        if (ds, str(run), factor, str(value)) in sens_keys:
            return None
        seed_r = run_seed(ctx.seed, ds, run)
        sp = make_split(ds, X.shape[0], y, run, ctx.test_size, seed_r)
        Xtr, ytr = X[sp.train_idx], y[sp.train_idx]
        Xte, yte = X[sp.test_idx], y[sp.test_idx]
        res = fit(Xtr, ytr, cfg, max_path_budget=manuscript_m(X.shape[1]))
        if res.error:
            failure_log(fails, "sensitivity", f"{ds}/{run}/{factor}={value}", res.error[-500:])
            return None
        ev = evaluate_selected(Xtr, ytr, Xte, yte, res.selected, "RF", seed_r)
        return {"dataset": ds, "run": run, "factor": factor, "value": str(value),
                "n_selected": len(res.selected), "d": X.shape[1],
                "feature_reduction": 1.0 - len(res.selected) / X.shape[1],
                "accuracy": ev.accuracy, "fs_seconds": res.fs_time_sec,
                "f_screened": res.f, "K": cfg.K, "lam": cfg.lam,
                "alpha": cfg.alpha, "beta": cfg.beta, "bins": cfg.bins,
                "error": ev.error or ""}

    for ds in datasets:
        try:
            X, y, meta = load_dataset(ds, ctx.load_dir())
        except Exception as exc:
            failure_log(fails, "sensitivity", ds, str(exc))
            continue
        d = X.shape[1]
        f0 = choose_f(d, RevisionConfig())
        factor_specs: List[Tuple[str, object, RevisionConfig]] = []
        for v in g["lambda_grid"]:
            factor_specs.append(("lambda", v, RevisionConfig(lam=float(v))))
        for (a, b) in g["ab_grid"]:
            factor_specs.append((f"alpha_beta", f"{a},{b}",
                                 RevisionConfig(alpha=float(a), beta=float(b))))
        for v in g["K_grid"]:
            factor_specs.append(("K", int(v), RevisionConfig(K=int(v))))
        for v in g["max_edges_grid"]:
            factor_specs.append(("max_edges", int(v), RevisionConfig(max_edges=int(v))))
        for v in g["screening_budget_multipliers"]:
            f_val = int(max(10, min(d, round(float(v) * f0))))
            factor_specs.append((f"screening_budget_mult", float(v), RevisionConfig(f=f_val)))
        for v in g["discretizer_grid"]:
            factor_specs.append(("discretizer", v, RevisionConfig(discretizer=v)))
        for v in g["bins_grid"]:
            factor_specs.append(("bins", int(v), RevisionConfig(bins=int(v))))
        for v in g["slope_grid"]:
            s = float(v)
            factor_specs.append(("sigmoid_slope", s,
                                 RevisionConfig(a1=s, b1=-s / 2, a2=s, b2=-s / 2,
                                                a3=s, b3=-s / 2, a4=s, b4=-s / 2)))

        for run in range(n_runs):
            for factor, value, cfg in factor_specs:
                row = eval_cfg(ds, run, X, y, factor, value,
                               cfg.with_updates(random_state=ctx.seed))
                if row:
                    rows.append(row)
            if rows:
                append_rows(raw_path, rows, ["dataset", "run", "factor", "value"])
                rows = []
    if rows:
        append_rows(raw_path, rows, ["dataset", "run", "factor", "value"])
    raw = read_csv_or_none(raw_path)
    paths = {"raw": raw_path}
    if raw is not None and not raw.empty:
        s = summarize(raw, ["dataset", "factor", "value"],
                      ["accuracy", "n_selected", "fs_seconds", "f_screened"])
        s.to_csv(out / "sensitivity_summary.csv", index=False)
        paths["summary"] = out / "sensitivity_summary.csv"
    return paths


# -------------------------------- ablation ------------------------------- #

def ablation_variants(ctx: RunContext, d: int) -> List[Tuple[str, RevisionConfig, str]]:
    base = RevisionConfig()
    f0 = choose_f(d, base)
    variants: List[Tuple[str, RevisionConfig, str]] = []

    def add(name, cfg, note=""):
        variants.append((name, cfg.with_updates(name=name), note))

    wanted = list(ctx.grids.get("ablation_subset", ["full"]))
    table = {
        "full": (RevisionConfig(), "full canonical method (independent group degrees)"),
        "vertex_only": (RevisionConfig(use_hyperedges=False, beta=0.0, K=1),
                        "vertex-only (no hyperedges)"),
        "K2": (RevisionConfig(K=2), "hyperedge orders 2..2"),
        "K3": (RevisionConfig(K=3), "hyperedge orders 2..3 (default)"),
        "K4": (RevisionConfig(K=4), "hyperedge orders 2..4"),
        "no_nonmembership_degree": (RevisionConfig(use_nonmembership_degree=False),
                                    "remove non-membership evidence BEFORE IFS normalization "
                                    "(not the same as lambda=0)"),
        "no_redundancy_penalty": (RevisionConfig(use_redundancy=False),
                                  "remove redundancy penalty in edge weight (lam=0); "
                                  "nu still affects IFS degrees/hesitation"),
        "no_hesitation": (RevisionConfig(use_hesitation_weight=False),
                          "remove hesitation weighting (1-pi)"),
        "binary_coverage": (RevisionConfig(g_type="binary"), "binary representative coverage"),
        "soft_coverage": (RevisionConfig(g_type="soft", soft_rho=0.5), "soft concave coverage rho=0.5"),
        "fraction_coverage": (RevisionConfig(g_type="fraction"), "fractional concave coverage"),
        "alternative_hyperedges": (RevisionConfig(edge_orders="exact_K"),
                                   "alternative construction: only order-K edges"),
        "alternative_discretization": (RevisionConfig(discretizer="uniform"),
                                       "equal-width discretization"),
        "alternative_estimator": (RevisionConfig(entropy_estimator="miller_madow"),
                                  "Miller-Madow entropy estimator"),
        "screen_budget_low": (RevisionConfig(f=max(10, round(0.5 * f0))),
                              "screening budget 0.5x default heuristic"),
        "screen_budget_high": (RevisionConfig(f=min(d, round(2.0 * f0))),
                               "screening budget 2x default heuristic"),
        "screen_no": (RevisionConfig(screening_strategy="none"), "no screening (f=d)"),
        "screening_anova": (RevisionConfig(screening_strategy="anova_f"),
                            "ANOVA-F screening (legacy-compatible, non-default)"),
        "screening_interaction_union": (RevisionConfig(screening_strategy="interaction_union"),
                                        "interaction-aware screening union (ablation, non-default)"),
        "sampled_vs_exhaustive": (RevisionConfig(f=min(d, 40), max_edges=20000),
                                  "exhaustive edge construction feasible after reducing "
                                  "screening budget to f=40 (documented)"),
        "strict_strong": (RevisionConfig(hyperedge_degrees="strict_strong"),
                          "strict strong-induced hyperedge degrees (ablation only)"),
        "independent_group": (RevisionConfig(hyperedge_degrees="independent"),
                              "independent group-degree variant (= submitted construction)"),
    }
    for name in wanted:
        if name in table:
            cfg, note = table[name]
            add(name, cfg, note)
    # ensure at least full
    if not variants:
        add("full", RevisionConfig(), "full canonical method")
    return variants


def run_ablation(ctx: RunContext) -> Dict[str, Path]:
    out = ctx.out("07_ablation")
    raw_path = out / "ablation_per_run.csv"
    fails = out / "failure_log.csv"
    datasets = list(ctx.grids.get("sensitivity_datasets", REPRESENTATIVE_DATASETS_PRESELECTED))
    datasets = [d for d in datasets if d in ctx.datasets] or ctx.datasets[:1]
    n_runs = min(ctx.n_runs, int({"smoke": 2, "core": 5, "full": 10}[ctx.profile]))
    write_json({"profile": ctx.profile, "datasets": datasets, "n_runs": n_runs,
                "variants": list(ctx.grids.get("ablation_subset", []))},
               out / "config.json")
    rows: List[Dict[str, object]] = []
    for ds in datasets:
        try:
            X, y, meta = load_dataset(ds, ctx.load_dir())
        except Exception as exc:
            failure_log(fails, "ablation", ds, str(exc))
            continue
        d = X.shape[1]
        variants = ablation_variants(ctx, d)
        abl_keys = load_existing_keys(raw_path, ["dataset", "run", "variant"])
        for run in range(n_runs):
            seed_r = run_seed(ctx.seed, ds, run)
            sp = make_split(ds, X.shape[0], y, run, ctx.test_size, seed_r)
            Xtr, ytr = X[sp.train_idx], y[sp.train_idx]
            Xte, yte = X[sp.test_idx], y[sp.test_idx]
            for name, cfg, note in variants:
                if (ds, str(run), name) in abl_keys:
                    continue
                # Feasibility guard: no-screen means f = d; the default vertex
                # redundancy is O(d^2) pairwise NMI evaluations.  On high-d
                # datasets this is computationally infeasible -> recorded as a
                # failure row rather than running silently for hours.
                c0 = cfg.with_updates(random_state=ctx.seed)
                if c0.screening_strategy == "none" and d > 1200:
                    rows.append({
                        "dataset": ds, "run": run, "variant": name, "note": note,
                        "n_selected": 0, "d": d, "feature_reduction": np.nan,
                        "accuracy": np.nan, "balanced_accuracy": np.nan,
                        "macro_f1": np.nan, "fs_seconds": 0.0,
                        "objective_value": np.nan, "K": c0.K, "lam": c0.lam,
                        "alpha": c0.alpha, "beta": c0.beta, "g_type": c0.g_type,
                        "bins": c0.bins, "discretizer": c0.discretizer,
                        "entropy_estimator": c0.entropy_estimator,
                        "screening_strategy": c0.screening_strategy,
                        "f_screened": d, "hyperedge_degrees": c0.hyperedge_degrees,
                        "use_redundancy": c0.use_redundancy,
                        "use_nonmembership_degree": c0.use_nonmembership_degree,
                        "use_hesitation_weight": c0.use_hesitation_weight,
                        "selected_indices": "",
                        "error": (f"INFEASIBLE: no-screen ablation requires "
                                  f"O(d^2)={d*d} pairwise redundancy evaluations "
                                  f"(d={d} > 1200 threshold); recorded, not run"),
                    })
                    abl_keys.add((ds, str(run), name))
                    append_rows(raw_path, rows, ["dataset", "run", "variant"])
                    rows = []
                    failure_log(fails, "ablation", f"{ds}/{run}/{name}",
                                f"infeasible no-screen on d={d}")
                    continue
                try:
                    c = cfg.with_updates(random_state=ctx.seed)
                    res = fit(Xtr, ytr, c, max_path_budget=manuscript_m(d))
                    if res.error:
                        failure_log(fails, "ablation", f"{ds}/{run}/{name}", res.error[-500:])
                        continue
                    ev = evaluate_selected(Xtr, ytr, Xte, yte, res.selected, "RF", seed_r)
                    rows.append({
                        "dataset": ds, "run": run, "variant": name, "note": note,
                        "n_selected": len(res.selected), "d": d,
                        "feature_reduction": 1.0 - len(res.selected) / d,
                        "accuracy": ev.accuracy,
                        "balanced_accuracy": ev.balanced_accuracy,
                        "macro_f1": ev.macro_f1,
                        "fs_seconds": res.fs_time_sec,
                        "objective_value": res.objective_value,
                        "K": c.K, "lam": c.lam, "alpha": c.alpha, "beta": c.beta,
                        "g_type": c.g_type, "bins": c.bins,
                        "discretizer": c.discretizer,
                        "entropy_estimator": c.entropy_estimator,
                        "screening_strategy": c.screening_strategy,
                        "f_screened": res.f,
                        "hyperedge_degrees": c.hyperedge_degrees,
                        "use_redundancy": c.use_redundancy,
                        "use_nonmembership_degree": c.use_nonmembership_degree,
                        "use_hesitation_weight": c.use_hesitation_weight,
                        "selected_indices": ";".join(map(str, res.selected)),
                        "error": ev.error or "",
                    })
                except Exception as exc:
                    failure_log(fails, "ablation", f"{ds}/{run}/{name}",
                                f"{type(exc).__name__}: {exc}")
            if rows:
                append_rows(raw_path, rows, ["dataset", "run", "variant"])
                rows = []
    if rows:
        append_rows(raw_path, rows, ["dataset", "run", "variant"])
    raw = read_csv_or_none(raw_path)
    paths = {"raw": raw_path}
    if raw is not None and not raw.empty:
        s = summarize(raw, ["dataset", "variant"],
                      ["accuracy", "n_selected", "fs_seconds", "objective_value"])
        s.to_csv(out / "ablation_summary.csv", index=False)
        # selected-feature differences vs full
        if "full" in set(raw["variant"]):
            base = raw[raw["variant"] == "full"].set_index(["dataset", "run"])
            diffs = []

            def _selset(v) -> set:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return set()
                s = str(v)
                if not s or s == "nan":
                    return set()
                out = set()
                for tok in s.split(";"):
                    if tok and tok != "nan":
                        try:
                            out.add(int(tok))
                        except ValueError:
                            pass
                return out

            for v in sorted(set(raw["variant"]) - {"full"}):
                sub = raw[raw["variant"] == v].set_index(["dataset", "run"])
                common = base.index.intersection(sub.index)
                for key in common:
                    a = _selset(base.loc[key, "selected_indices"])
                    b = _selset(sub.loc[key, "selected_indices"])
                    diffs.append({"dataset": key[0], "run": key[1], "variant": v,
                                  "jaccard_vs_full": jaccard(sorted(a), sorted(b)),
                                  "n_only_in_full": len(a - b),
                                  "n_only_in_variant": len(b - a)})
            pd.DataFrame(diffs).to_csv(out / "selected_feature_differences_vs_full.csv",
                                       index=False)
        # cross-run selection stability per variant (incl. coverage variants)
        stab_rows = []
        for (ds, variant), g in raw.groupby(["dataset", "variant"]):
            sels = []
            for s in g["selected_indices"]:
                if isinstance(s, float) and np.isnan(s):
                    sels.append([])
                    continue
                toks = [] if (s is None or str(s) in ("", "nan")) else str(s).split(";")
                ints = []
                for tok in toks:
                    if tok and tok != "nan":
                        try:
                            ints.append(int(tok))
                        except ValueError:
                            pass
                sels.append(sorted(ints))
            sels = [s for s in sels if len(s) > 0]
            if len(sels) >= 2:
                stats = pairwise_summary(sels, int(g["d"].iloc[0]))
                stab_rows.append({"dataset": ds, "variant": variant, **stats})
        if stab_rows:
            pd.DataFrame(stab_rows).to_csv(out / "ablation_selection_stability.csv",
                                           index=False)
        paths["summary"] = out / "ablation_summary.csv"
        paths["stability"] = out / "ablation_selection_stability.csv"
    return paths


# ------------------------------- synthetic ------------------------------- #

def _synthetic_method_configs(ctx: RunContext, d: int) -> List[Tuple[str, RevisionConfig, str]]:
    """Configs compared in the synthetic interaction suite."""
    full_d_budget = d <= 60
    cfgs: List[Tuple[str, RevisionConfig, str]] = [
        ("canonical_default", RevisionConfig(K=3, g_type="binary"),
         "canonical with default univariate MI screening"),
        ("vertex_only", RevisionConfig(use_hyperedges=False, beta=0.0, K=1),
         "vertex-only"),
        ("K2", RevisionConfig(K=2), "orders 2..2"),
        ("K3", RevisionConfig(K=3), "orders 2..3"),
        ("binary_coverage", RevisionConfig(g_type="binary"), "binary coverage"),
        ("soft_coverage", RevisionConfig(g_type="soft"), "soft coverage rho=0.5"),
    ]
    if full_d_budget:
        cfgs.append(("no_screen", RevisionConfig(screening_strategy="none"),
                     "no screening (d small enough)"))
    cfgs.append(("interaction_union_screen",
                 RevisionConfig(screening_strategy="interaction_union"),
                 "interaction-aware screening union (non-default ablation)"))
    return cfgs


SYNTH_SCENARIOS_FULL = ["xor2", "parity3"]
SYNTH_SCENARIOS_REDUCED = ["main_effect", "redundant_copy", "mixed",
                           "interaction_noise", "correlated_noise"]


def run_synthetic(ctx: RunContext) -> Dict[str, Path]:
    out = ctx.out("02_synthetic_interactions")
    raw_path = out / "synthetic_per_seed.csv"
    fails = out / "failure_log.csv"
    seeds_n = int(ctx.grids.get("synthetic_seeds", 20))
    seeds = list(range(seeds_n))
    if ctx.profile == "smoke":
        n_regimes = [100]
        noise_regimes = {"large": 100}
    else:
        n_regimes = [100, 300]
        noise_regimes = {"small": 30, "large": 100}
    write_json({"profile": ctx.profile, "seeds": seeds,
                "n_regimes": n_regimes, "noise_regimes": noise_regimes,
                "scenarios_full_config": SYNTH_SCENARIOS_FULL,
                "scenarios_reduced_config": SYNTH_SCENARIOS_REDUCED,
                "note": ("seeds < 20 would be reported explicitly; profile settings above "
                         "record the actual count")},
               out / "config.json")
    rows: List[Dict[str, object]] = []
    gt_rows: List[Dict[str, object]] = []
    syn_keys = load_existing_keys(raw_path, ["scenario", "n", "noise_regime", "seed", "method"])

    scenarios = SYNTH_SCENARIOS_FULL + SYNTH_SCENARIOS_REDUCED
    for scenario in scenarios:
        full_cfg = scenario in SYNTH_SCENARIOS_FULL
        for n in n_regimes:
            for noise_name, n_noise in noise_regimes.items():
                for seed in seeds:
                    try:
                        if scenario == "main_effect":
                            ds_syn = generate("main_effect", n=n, seed=seed,
                                              n_signal=5, n_noise=n_noise)
                        elif scenario == "redundant_copy":
                            ds_syn = generate("redundant_copy", n=n, seed=seed,
                                              n_signal=4, n_copies=3, n_noise=n_noise)
                        elif scenario == "xor2":
                            ds_syn = generate("xor2", n=n, seed=seed, n_noise=n_noise)
                        elif scenario == "parity3":
                            ds_syn = generate("parity3", n=n, seed=seed, n_noise=n_noise)
                        elif scenario == "mixed":
                            ds_syn = generate("mixed", n=n, seed=seed,
                                              n_main=4, n_noise=n_noise)
                        elif scenario == "interaction_noise":
                            ds_syn = generate("interaction_noise", n=n, seed=seed,
                                              n_noise=n_noise, n_main=2)
                        elif scenario == "correlated_noise":
                            ds_syn = generate("correlated_noise", n=n, seed=seed,
                                              n_signal=5, n_noise=n_noise, rho=0.8)
                        else:
                            raise ValueError(scenario)
                    except Exception as exc:
                        failure_log(fails, "synthetic", f"{scenario}/{n}/{seed}",
                                    f"generate: {exc}")
                        continue
                    X, y = ds_syn.X, ds_syn.y
                    support = ds_syn.support
                    inter = ds_syn.interaction_support
                    d = X.shape[1]
                    gt_rows.append({
                        "scenario": scenario, "n": n, "noise_regime": noise_name,
                        "seed": seed, "d": d,
                        "support": ";".join(map(str, support.tolist())),
                        "interaction_support": ";".join(map(str, inter.tolist())),
                    })
                    split_seed = run_seed(ctx.seed, f"{scenario}_{n}_{noise_name}", seed)
                    sp = make_split(f"syn_{scenario}", X.shape[0], y, seed,
                                    ctx.test_size, split_seed)
                    Xtr, ytr = X[sp.train_idx], y[sp.train_idx]
                    Xte, yte = X[sp.test_idx], y[sp.test_idx]

                    method_specs: List[Tuple[str, Optional[RevisionConfig], str]] = []
                    if full_cfg:
                        for name, cfg, note in _synthetic_method_configs(ctx, d):
                            method_specs.append((name, cfg, note))
                    else:
                        method_specs = [
                            ("canonical_default", RevisionConfig(), "canonical default"),
                            ("soft_coverage", RevisionConfig(g_type="soft"), "soft coverage"),
                        ]
                    # authentic baselines
                    baseline_specs = ["ReliefF", "CMIM"]
                    if full_cfg:
                        baseline_specs.append("JMIM")

                    for name, cfg, note in method_specs:
                        if (scenario, str(n), noise_name, str(seed), name) in syn_keys:
                            continue
                        try:
                            c = cfg.with_updates(random_state=ctx.seed, name=name)
                            res = fit(Xtr, ytr, c, max_path_budget=manuscript_m(d))
                            if res.error:
                                failure_log(fails, "synthetic",
                                            f"{scenario}/{n}/{seed}/{name}", res.error[-400:])
                                continue
                            sel = res.selected
                            rec = recovery_metrics(sel, support)
                            rec_inter = recovery_metrics(sel, inter) if len(inter) else {
                                "precision": np.nan, "recall": np.nan, "f1": np.nan,
                                "jaccard": np.nan, "all_selected": np.nan}
                            screened = set(int(i) for i in res.screened_idx)
                            sup_set = set(int(i) for i in support)
                            inter_set = set(int(i) for i in inter)
                            screen_recall = (len(screened & sup_set) / len(sup_set)
                                             if sup_set else np.nan)
                            screen_recall_inter = (len(screened & inter_set) / len(inter_set)
                                                   if inter_set else np.nan)
                            ev = evaluate_selected(Xtr, ytr, Xte, yte, sel, "RF", split_seed)
                            rows.append({
                                "scenario": scenario, "n": n, "noise_regime": noise_name,
                                "seed": seed, "method": name, "kind": "sifhfam_config",
                                "d": d, "n_selected": rec["n_selected"],
                                "precision": rec["precision"], "recall": rec["recall"],
                                "f1": rec["f1"], "jaccard": rec["jaccard"],
                                "all_selected": rec["all_selected"],
                                "interaction_precision": rec_inter["precision"],
                                "interaction_recall": rec_inter["recall"],
                                "interaction_all_selected": rec_inter["all_selected"],
                                "screening_recall_support": screen_recall,
                                "screening_recall_interaction": screen_recall_inter,
                                "f_screened": res.f,
                                "accuracy": ev.accuracy,
                                "fs_seconds": res.fs_time_sec,
                                "note": note, "error": ev.error or "",
                            })
                        except Exception as exc:
                            failure_log(fails, "synthetic",
                                        f"{scenario}/{n}/{seed}/{name}",
                                        f"{type(exc).__name__}: {exc}")

                    for bm in baseline_specs:
                        if (scenario, str(n), noise_name, str(seed), bm) in syn_keys:
                            continue
                        try:
                            ob = run_baseline(bm, Xtr, ytr,
                                              max_budget=manuscript_m(d),
                                              seed=split_seed)
                            if ob.status != "OK" or ob.ranking is None:
                                failure_log(fails, "synthetic",
                                            f"{scenario}/{n}/{seed}/{bm}",
                                            f"{ob.status}: {ob.error}")
                                rows.append({
                                    "scenario": scenario, "n": n,
                                    "noise_regime": noise_name, "seed": seed,
                                    "method": bm, "kind": "baseline", "d": d,
                                    "n_selected": 0,
                                    "precision": np.nan, "recall": np.nan,
                                    "f1": np.nan, "jaccard": np.nan,
                                    "all_selected": np.nan,
                                    "interaction_precision": np.nan,
                                    "interaction_recall": np.nan,
                                    "interaction_all_selected": np.nan,
                                    "screening_recall_support": np.nan,
                                    "screening_recall_interaction": np.nan,
                                    "f_screened": np.nan, "accuracy": np.nan,
                                    "fs_seconds": ob.seconds, "note": ob.notes or "",
                                    "error": (ob.error or ob.status)[:300],
                                })
                                continue
                            ranking = np.asarray(ob.ranking, dtype=int)
                            m = min(manuscript_m(d), len(ranking))
                            sel = ranking[:m]
                            rec = recovery_metrics(sel, support)
                            rec_inter = recovery_metrics(sel, inter) if len(inter) else {
                                "precision": np.nan, "recall": np.nan, "f1": np.nan,
                                "jaccard": np.nan, "all_selected": np.nan}
                            ev = evaluate_selected(Xtr, ytr, Xte, yte, sel, "RF", split_seed)
                            rows.append({
                                "scenario": scenario, "n": n,
                                "noise_regime": noise_name, "seed": seed,
                                "method": bm, "kind": "baseline", "d": d,
                                "n_selected": rec["n_selected"],
                                "precision": rec["precision"], "recall": rec["recall"],
                                "f1": rec["f1"], "jaccard": rec["jaccard"],
                                "all_selected": rec["all_selected"],
                                "interaction_precision": rec_inter["precision"],
                                "interaction_recall": rec_inter["recall"],
                                "interaction_all_selected": rec_inter["all_selected"],
                                "screening_recall_support": np.nan,
                                "screening_recall_interaction": np.nan,
                                "f_screened": np.nan, "accuracy": ev.accuracy,
                                "fs_seconds": ob.seconds, "note": ob.notes or "",
                                "error": ev.error or "",
                            })
                        except Exception as exc:
                            failure_log(fails, "synthetic",
                                        f"{scenario}/{n}/{seed}/{bm}",
                                        f"{type(exc).__name__}: {exc}")
                    if rows:
                        append_rows(raw_path, rows,
                                    ["scenario", "n", "noise_regime", "seed", "method"])
                        syn_keys.update(
                            (r["scenario"], str(r["n"]), r["noise_regime"], str(r["seed"]),
                             r["method"]) for r in rows)
                        rows = []

    if rows:
        append_rows(raw_path, rows, ["scenario", "n", "noise_regime", "seed", "method"])
    if gt_rows:
        pd.DataFrame(gt_rows).drop_duplicates(
            subset=["scenario", "n", "noise_regime", "seed"]
        ).to_csv(out / "ground_truth_support.csv", index=False)

    # marginal-weakness assertion evidence for XOR/parity
    weak_rows = []
    for scenario in ("xor2", "parity3"):
        ds_syn = generate(scenario, n=200, seed=0, n_noise=0)
        for j in ds_syn.support:
            weak_rows.append({
                "scenario": scenario, "feature": int(j),
                "marginal_nmi_with_y": marginal_association_check(ds_syn.X[:, j], ds_syn.y),
            })
    pd.DataFrame(weak_rows).to_csv(out / "xor_parity_marginal_association.csv", index=False)

    export_interaction_example_table(out / "deterministic_interaction_example.csv", seed=0)

    raw = read_csv_or_none(raw_path)
    paths = {"raw": raw_path}
    if raw is not None and not raw.empty:
        # Reporting-only builders shared with sifhfam.reporting_only;
        # the experimental sample size stays in `n`, the seed count is
        # `n_seeds` (Issue A schema rule).
        from .reporting_only import build_synthetic_summary, build_xor_parity_summary
        s = build_synthetic_summary(raw)
        s.to_csv(out / "synthetic_summary.csv", index=False)
        paths["summary"] = out / "synthetic_summary.csv"
        xor = build_xor_parity_summary(raw)
        if not xor.empty:
            xor.to_csv(out / "xor_parity_recovery_summary.csv", index=False)
            paths["xor_summary"] = out / "xor_parity_recovery_summary.csv"
    return paths


# --------------------------- runtime/scalability ------------------------- #

def run_runtime(ctx: RunContext) -> Dict[str, Path]:
    out = ctx.out("08_runtime_scalability")
    fails = out / "failure_log.csv"
    from .runtime_audit import environment_snapshot
    write_json(environment_snapshot(), out / "environment.json")

    # ---- stage-level timings aggregated from equal-budget raw (if present) ----
    eb_raw = read_csv_or_none(ctx.evidence / "03_equal_budget" / "equal_budget_per_run.csv")
    stage_rows = []
    if eb_raw is not None and not eb_raw.empty:
        canon = eb_raw[eb_raw["method"] == "SIFHFAM_canonical"]
        for ds, g in canon.groupby("dataset"):
            stage_rows.append({
                "dataset": ds, "source": "equal_budget_canonical",
                "fs_seconds_mean": g["fs_seconds"].mean(),
                "fs_seconds_std": g["fs_seconds"].std(),
                "eval_seconds_mean": g["eval_seconds"].mean(),
                "n_rows": len(g),
            })
    pd.DataFrame(stage_rows).to_csv(out / "stage_timing_from_main_runs.csv", index=False)

    # ---- synthetic scaling sweep ----
    d_grid = list(ctx.grids.get("scaling_d", [500, 1000]))
    K_grid = list(ctx.grids.get("scaling_K_grid", [2, 3]))
    if ctx.profile == "smoke":
        K_grid = [3]
    f_grid = list(ctx.grids.get("scaling_f_grid", []))
    edge_grid = list(ctx.grids.get("scaling_edge_grid", []))
    n_fixed = 200
    rows: List[Dict[str, object]] = []
    raw_path = out / "scaling_per_config.csv"
    write_json({"n_fixed": n_fixed, "d_grid": d_grid, "K_grid": K_grid,
                "f_grid": f_grid, "edge_grid": edge_grid,
                "timeout_note": "configs exceeding 600 s recorded as TIMEOUT"},
               out / "scaling_config.json")

    for d in d_grid:
        for K in K_grid:
            try:
                rng = np.random.default_rng(12345)
                X = rng.normal(size=(n_fixed, d))
                y = (X[:, :5].sum(axis=1) + 0.5 * rng.normal(size=n_fixed) > 0).astype(int)
                cfg = RevisionConfig(random_state=ctx.seed, K=int(K),
                                     name="scaling")
                t0 = time.time()
                res = fit(X, y, cfg, max_path_budget=20)
                dt = time.time() - t0
                if dt > 600:
                    rows.append({"kind": "d_K", "d": d, "K": K, "n": n_fixed,
                                 "status": "TIMEOUT", "total_seconds": dt})
                    continue
                rows.append({
                    "kind": "d_K", "d": d, "K": K, "n": n_fixed, "status": "OK",
                    "total_seconds": res.timings.total_fs_sec,
                    "screen_score_sec": res.timings.screening_score_sec,
                    "discretization_sec": res.timings.discretization_sec,
                    "vertex_relevance_sec": res.timings.vertex_relevance_sec,
                    "vertex_redundancy_sec": res.timings.vertex_redundancy_sec,
                    "hyperedge_gen_sec": res.timings.hyperedge_generation_sec,
                    "hyperedge_score_sec": res.timings.hyperedge_scoring_sec,
                    "edge_weight_sec": res.timings.edge_weight_sec,
                    "greedy_sec": res.timings.greedy_sec,
                    "f": res.f, "n_edges": res.ops.n_edges,
                    "rss_peak_delta": res.memory.rss_peak_delta_bytes,
                    "tracemalloc_peak": res.memory.tracemalloc_peak_bytes,
                    "error": "",
                })
            except Exception as exc:
                failure_log(fails, "scaling", f"d={d},K={K}",
                            f"{type(exc).__name__}: {exc}")
                rows.append({"kind": "d_K", "d": d, "K": K, "n": n_fixed,
                             "status": "FAILED",
                             "error": f"{type(exc).__name__}: {exc}"})

    # f and edge-budget sweeps (full profile; on a fixed d)
    if f_grid or edge_grid:
        d_fix = 2000
        rng = np.random.default_rng(7)
        X = rng.normal(size=(n_fixed, d_fix))
        y = (X[:, :5].sum(axis=1) > 0).astype(int)
        for fv in f_grid:
            for ev in (edge_grid or [5000]):
                cfg = RevisionConfig(random_state=ctx.seed, f=int(fv),
                                     max_edges=int(ev), name="scaling_f_edges")
                t0 = time.time()
                res = fit(X, y, cfg, max_path_budget=20)
                rows.append({
                    "kind": "f_edges", "d": d_fix, "n": n_fixed, "f_setting": fv,
                    "max_edges": ev, "K": 3, "status": "OK",
                    "total_seconds": res.timings.total_fs_sec,
                    "hyperedge_score_sec": res.timings.hyperedge_scoring_sec,
                    "greedy_sec": res.timings.greedy_sec,
                    "f": res.f, "n_edges": res.ops.n_edges,
                    "n_positive_edges": res.ops.n_positive_edges,
                    "theoretical_edges": str(res.ops.theoretical_edges_by_order),
                    "error": "",
                })
    if rows:
        fixed_cols = [
            "kind", "d", "K", "n", "f_setting", "max_edges", "status",
            "total_seconds", "screen_score_sec", "discretization_sec",
            "vertex_relevance_sec", "vertex_redundancy_sec",
            "hyperedge_gen_sec", "hyperedge_score_sec", "edge_weight_sec",
            "greedy_sec", "f", "n_edges", "n_positive_edges",
            "theoretical_edges", "rss_peak_delta", "tracemalloc_peak", "error",
        ]
        norm_rows = []
        for r in rows:
            norm_rows.append({c: r.get(c, np.nan) for c in fixed_cols})
        append_rows(raw_path, norm_rows,
                    ["kind", "d", "K", "n", "f_setting", "max_edges"])

    # ---- complexity analysis document ----
    doc = """# COMPLEXITY ANALYSIS (derived from the implemented pipeline)

Notation: n = samples, d = original features, f = screened features,
K = maximum hyperedge order, E = number of evaluated hyperedges
(E <= max_edges, orders 2..K), m = selected cardinality, b = bins.

## Implemented stages and their costs

1. **Screening (univariate NMI)**: discretization O(n d log b) via vectorized
   percentiles + per-column NMI via bincount histograms O(n d).
   Operationally measured as `screening_score_sec`.
2. **Post-screen pairwise redundancy (vertex, default nmi_mean)**:
   O(n f + f^2) time with combined-key histograms
   (`pairwise_redundancy_evals = f(f-1)` directed evaluations, stored in
   `OpCounts`), O(f) memory.
3. **Hyperedge candidate count**: theoretical total
   sum_{k=2}^{K} C(f, k). With sampling, only E <= max_edges edges are
   evaluated (`evaluated_edges_by_order` records the realized counts; the
   sampled/capped nature must be stated whenever scalability is discussed).
4. **Entropy/MI scoring of hyperedges**: O(E * n * K) worst-case for joint
   histograms of order-K columns (`n_joint_entropy_calls ~ 2E`,
   `n_mi_calls ~ E + f + f(f-1)`).
5. **Greedy incidence-based selection**: building incidence O(E K); each of m
   rounds scans f candidates, each costing O(deg(candidate)):
   O(m * (f + E)) typical worst-case scanning all candidates each round:
   O(m f + m E).
6. **Edge weights**: O(E).

## Time and space complexity (implemented pipeline)

- Time: O(n d log b + f^2 * c_nmi + sum_{k<=K} C(f,k) sampled-to-E * (n k) + m(f + E))
- Space: O(n f + f^2???) -> implemented as O(n d) input + O(f) vectors + O(E K)
  edge structures + O(f^2)-time but O(f)-space redundancy (no f x f matrix is
  stored: pairwise loop), O(n b) histograms transient.

## Complete vs sampled hypergraphs

- Complete: requires E = sum_{k=2}^{K} C(f,k); infeasible beyond small f
  (e.g., f=30, K=3 gives 4,495 edges; f=100 gives 166,650; f=300 gives ~4.5M).
- Sampled/capped: E <= max_edges (default 5,000), uniform per-order budget
  max_edges / (#orders); seeds and realized edge counts are recorded in every
  run (`hyperedge_info`, `OpCounts.theoretical_edges_by_order` vs
  `evaluated_edges_by_order`).

Any empirical scaling claim must state the sampling/cap configuration.
"""
    (out / "COMPLEXITY_ANALYSIS.md").write_text(doc, encoding="utf-8")
    return {"scaling": raw_path,
            "complexity_md": out / "COMPLEXITY_ANALYSIS.md",
            "environment": out / "environment.json"}


# -------------------------------- stability ------------------------------ #

def run_stability(ctx: RunContext) -> Dict[str, Path]:
    out = ctx.out("09_stability")
    fails = out / "failure_log.csv"
    st = dict(ctx.grids.get("stability", {"n_splits": 6, "n_hyperedge_seeds": 4,
                                          "n_bootstrap": 10}))
    datasets = list(ctx.grids.get("sensitivity_datasets", REPRESENTATIVE_DATASETS_PRESELECTED))
    datasets = [d for d in datasets if d in ctx.datasets] or ctx.datasets[:1]
    n_splits = int(st.get("n_splits", 6))
    n_edge_seeds = int(st.get("n_hyperedge_seeds", 4))
    n_boot = int(st.get("n_bootstrap", 10))
    write_json({"datasets": datasets, "n_splits": n_splits,
                "n_hyperedge_seeds": n_edge_seeds, "n_bootstrap": n_boot,
                "factors": {"A": "split varies, hyperedge seed fixed",
                            "B": "same split, hyperedge seed varies",
                            "C": "both vary",
                            "D": "screened-set stability across splits",
                            "E": "determinism check"}},
               out / "config.json")

    sel_rows: List[Dict[str, object]] = []
    decomp_rows: List[Dict[str, object]] = []
    cons_rows: List[Dict[str, object]] = []

    for ds in datasets:
        try:
            X, y, meta = load_dataset(ds, ctx.load_dir())
        except Exception as exc:
            failure_log(fails, "stability", ds, str(exc))
            continue
        d = X.shape[1]
        m = min(manuscript_m(d), choose_f(d, RevisionConfig()))

        # A: split instability (fixed hyperedge seed)
        sels_A, screened_A = [], []
        for run in range(n_splits):
            seed_r = run_seed(ctx.seed, ds, run)
            sp = make_split(ds, X.shape[0], y, run, ctx.test_size, seed_r)
            cfg = RevisionConfig(random_state=ctx.seed, name="A")
            res = fit(X[sp.train_idx], y[sp.train_idx], cfg, max_path_budget=m)
            if res.error:
                failure_log(fails, "stability", f"{ds}/A/{run}", res.error[-400:])
                continue
            sels_A.append([int(i) for i in res.selected])
            screened_A.append([int(i) for i in res.screened_idx])
            sel_rows.append({"dataset": ds, "factor": "A_split", "rep": run,
                             "seed": seed_r, "hyperedge_seed": ctx.seed,
                             "selected": ";".join(map(str, res.selected))})
        if len(sels_A) >= 2:
            stats = pairwise_summary(sels_A, d)
            decomp_rows.append({"dataset": ds, "factor": "A_split_instability", **stats})
            stats_s = pairwise_summary(screened_A, d)
            decomp_rows.append({"dataset": ds, "factor": "D_screening_instability",
                                **stats_s})

        # B: hyperedge-seed instability (fixed split run0)
        sp0 = make_split(ds, X.shape[0], y, 0, ctx.test_size, run_seed(ctx.seed, ds, 0))
        sels_B = []
        for i in range(n_edge_seeds):
            cfg = RevisionConfig(random_state=ctx.seed + 1000 + i, name="B")
            res = fit(X[sp0.train_idx], y[sp0.train_idx], cfg, max_path_budget=m)
            if res.error:
                failure_log(fails, "stability", f"{ds}/B/{i}", res.error[-400:])
                continue
            sels_B.append([int(j) for j in res.selected])
            sel_rows.append({"dataset": ds, "factor": "B_hyperedge_seed", "rep": i,
                             "seed": sp0.seed, "hyperedge_seed": cfg.random_state,
                             "selected": ";".join(map(str, res.selected))})
        if len(sels_B) >= 2:
            stats = pairwise_summary(sels_B, d)
            decomp_rows.append({"dataset": ds, "factor": "B_hyperedge_sampling_instability",
                                **stats})

        # C: combined (both vary)
        sels_C = []
        for run in range(n_splits):
            seed_r = run_seed(ctx.seed + 7777, ds, run)
            sp = make_split(ds, X.shape[0], y, run, ctx.test_size, seed_r)
            cfg = RevisionConfig(random_state=ctx.seed + 7777 + run, name="C")
            res = fit(X[sp.train_idx], y[sp.train_idx], cfg, max_path_budget=m)
            if res.error:
                continue
            sels_C.append([int(j) for j in res.selected])
            sel_rows.append({"dataset": ds, "factor": "C_combined", "rep": run,
                             "seed": seed_r, "hyperedge_seed": cfg.random_state,
                             "selected": ";".join(map(str, res.selected))})
        if len(sels_C) >= 2:
            stats = pairwise_summary(sels_C, d)
            decomp_rows.append({"dataset": ds, "factor": "C_combined_instability", **stats})

        # E: determinism
        try:
            res1 = fit(X[sp0.train_idx], y[sp0.train_idx],
                       RevisionConfig(random_state=ctx.seed), max_path_budget=m)
            res2 = fit(X[sp0.train_idx], y[sp0.train_idx],
                       RevisionConfig(random_state=ctx.seed), max_path_budget=m)
            det = bool(np.array_equal(res1.selected, res2.selected))
            decomp_rows.append({"dataset": ds, "factor": "E_determinism",
                                "n_pairs": 1, "jaccard_mean": float(det),
                                "jaccard_std": 0.0, "kuncheva_mean": np.nan,
                                "kuncheva_std": np.nan, "nogueira": np.nan})
        except Exception as exc:
            failure_log(fails, "stability", f"{ds}/E", str(exc))

        # Consensus (training-only bootstrap) - full profile
        if st.get("consensus", False) or ctx.profile == "full":
            rng = np.random.default_rng(ctx.seed)
            boot_sels = []
            n_tr = len(sp0.train_idx)
            idx_tr = np.arange(n_tr)
            for bi in range(n_boot):
                bidx = rng.choice(idx_tr, size=n_tr, replace=True)
                Xb = X[sp0.train_idx][bidx]
                yb = y[sp0.train_idx][bidx]
                if len(np.unique(yb)) < 2:
                    continue
                cfg = RevisionConfig(random_state=ctx.seed + bi, name="consensus")
                res = fit(Xb, yb, cfg, max_path_budget=m)
                if not res.error:
                    boot_sels.append([int(j) for j in res.selected])
            if boot_sels:
                cons, freq = consensus_select(boot_sels, m, d)
                # accuracy on untouched outer test
                ev = evaluate_selected(X[sp0.train_idx], y[sp0.train_idx],
                                        X[sp0.test_idx], y[sp0.test_idx],
                                        list(cons), "RF", sp0.seed)
                cons_rows.append({
                    "dataset": ds, "n_bootstrap": len(boot_sels), "m": int(m),
                    "accuracy": ev.accuracy, "balanced_accuracy": ev.balanced_accuracy,
                    "macro_f1": ev.macro_f1,
                    "consensus_stability_vs_split_sets": float(np.mean(
                        [jaccard(cons, s) for s in sels_A])) if sels_A else np.nan,
                    "mean_selection_frequency": float(np.mean(freq[cons])) if len(cons) else np.nan,
                    "consensus_indices": ";".join(map(str, cons.tolist())),
                })

    if sel_rows:
        append_rows(out / "stability_selections.csv", sel_rows,
                    ["dataset", "factor", "rep"])
    if decomp_rows:
        pd.DataFrame(decomp_rows).to_csv(out / "stability_decomposition.csv", index=False)
    if cons_rows:
        pd.DataFrame(cons_rows).to_csv(out / "consensus_stability_tradeoff.csv", index=False)
    return {"selections": out / "stability_selections.csv",
            "decomposition": out / "stability_decomposition.csv",
            "consensus": out / "consensus_stability_tradeoff.csv"}


# ------------------------------- statistics ------------------------------ #

def run_statistics(ctx: RunContext) -> Dict[str, Path]:
    from .statistics import plot_rank_distribution, run_statistics as _run_stats, save_stats_inputs
    out = ctx.out("10_statistics")
    eb = read_csv_or_none(ctx.evidence / "03_equal_budget" / "equal_budget_per_run.csv")
    paths: Dict[str, Path] = {}
    if eb is None or eb.empty:
        (out / "STATS_SKIPPED.txt").write_text("equal_budget_per_run.csv missing/empty",
                                               encoding="utf-8")
        return paths
    # dataset-level paired unit: mean over runs at the manuscript budget
    man = eb[eb["manuscript_budget"] == True]  # noqa: E712
    if man.empty:
        man = eb[eb["budget"] == eb.groupby("dataset")["budget"].transform("max")]
    ref = "SIFHFAM_canonical"
    try:
        wilcoxon_df, friedman_df, ranks_df, pivot = _run_stats(man, ref_method=ref)
        wilcoxon_df.to_csv(out / "wilcoxon_holm_two_sided.csv", index=False)
        friedman_df.to_csv(out / "friedman.csv", index=False)
        ranks_df.to_csv(out / "average_ranks.csv", index=False)
        save_stats_inputs(pivot, out / "stats_raw_input_pivot.csv")
        paths.update({"wilcoxon": out / "wilcoxon_holm_two_sided.csv",
                      "friedman": out / "friedman.csv",
                      "ranks": out / "average_ranks.csv",
                      "raw": out / "stats_raw_input_pivot.csv"})
        p = plot_rank_distribution(ranks_df, pivot, out / "rank_distribution.png")
        if p:
            paths["rank_plot"] = p
        stale = out / "rank_boxplot_cd.png"
        if stale.exists():
            stale.unlink()
    except Exception as exc:
        failure_log(out / "failure_log.csv", "stats", "main", f"{type(exc).__name__}: {exc}")

    # Empirical objective <-> accuracy association (NOT a theorem)
    assoc_rows = []
    datasets = list(ctx.grids.get("sensitivity_datasets", REPRESENTATIVE_DATASETS_PRESELECTED))
    datasets = [d for d in datasets if d in ctx.datasets] or ctx.datasets[:1]
    n_assoc_runs = min(3, ctx.n_runs)
    try:
        from scipy.stats import spearmanr
        for ds in datasets:
            X, y, meta = load_dataset(ds, ctx.load_dir())
            d = X.shape[1]
            f_ref = choose_f(d, RevisionConfig())
            budgets = [b for b in budget_grid(d) if b <= f_ref]
            for run in range(n_assoc_runs):
                seed_r = run_seed(ctx.seed, ds, run)
                sp = make_split(ds, X.shape[0], y, run, ctx.test_size, seed_r)
                Xtr, ytr = X[sp.train_idx], y[sp.train_idx]
                # inner validation split (train-only)
                inner = make_split(ds + "_inner", Xtr.shape[0], ytr, 0, 0.25,
                                   run_seed(seed_r, "inner", 0))
                cfg = RevisionConfig(random_state=ctx.seed)
                res = fit(Xtr, ytr, cfg, max_path_budget=max(budgets) if budgets else 20)
                if res.error:
                    continue
                order = res.screened_idx[res.greedy_order_screened]
                obj_vals, acc_vals, rand_acc_vals, info_vals = [], [], [], []
                from .information import feature_nmi_scores
                from .discretization import Discretizer
                Xd_full = Discretizer().fit_transform(Xtr)
                mi_full = feature_nmi_scores(Xd_full, ytr)
                rng = np.random.default_rng(seed_r)
                for b in budgets:
                    sel = order[:b]
                    if len(sel) < 2:
                        continue
                    # objective value of prefix from stored path (scaled per prefix)
                    o = Objective(res.vertex_mu, res.edge_weights,
                                  alpha=cfg.alpha, beta=cfg.beta, g_type=cfg.g_type)
                    local_order = [int(np.where(res.screened_idx == j)[0][0]) for j in sel
                                   if j in set(res.screened_idx.tolist())]
                    obj_vals.append(o.value(local_order))
                    ev = evaluate_selected(Xtr[inner.train_idx], ytr[inner.train_idx],
                                           Xtr[inner.test_idx], ytr[inner.test_idx],
                                           list(sel), "RF", seed_r)
                    acc_vals.append(ev.accuracy)
                    # random subset baseline of same size
                    rsel = rng.choice(d, size=min(b, d), replace=False)
                    ev_r = evaluate_selected(Xtr[inner.train_idx], ytr[inner.train_idx],
                                             Xtr[inner.test_idx], ytr[inner.test_idx],
                                             list(rsel), "RF", seed_r)
                    rand_acc_vals.append(ev_r.accuracy)
                    info_vals.append(float(np.mean(mi_full[sel])) if len(sel) else np.nan)
                if len(obj_vals) >= 3:
                    rho_acc, p_acc = spearmanr(obj_vals, acc_vals)
                    rho_rand, p_rand = spearmanr(obj_vals, rand_acc_vals)
                    rho_info, p_info = spearmanr(obj_vals, info_vals)
                    assoc_rows.append({
                        "dataset": ds, "run": run, "n_points": len(obj_vals),
                        "spearman_obj_vs_val_accuracy": rho_acc, "p_val": p_acc,
                        "spearman_obj_vs_random_accuracy": rho_rand, "p_rand": p_rand,
                        "spearman_obj_vs_retained_mi": rho_info, "p_info": p_info,
                        "interpretation": "empirical association only; not a bound",
                    })
    except Exception as exc:
        failure_log(out / "failure_log.csv", "stats", "association",
                    f"{type(exc).__name__}: {exc}")
    if assoc_rows:
        pd.DataFrame(assoc_rows).to_csv(
            out / "objective_accuracy_association.csv", index=False)
        paths["association"] = out / "objective_accuracy_association.csv"

    # theoretical-scope tiny greedy-vs-optimum records (empirical ratio)
    ratio_rows = []
    rng = np.random.default_rng(0)
    for trial in range(20):
        f = 7
        m_budget = max(2, f // 2)  # non-trivial budget (< f) so optimum != everything
        mu = rng.random(f)
        edges = [tuple(sorted(rng.choice(f, size=2, replace=False).tolist()))
                 for _ in range(8)]
        w = {e: float(rng.random()) for e in edges}
        obj = Objective(mu, w, g_type="binary")
        path = greedy_path(obj, budget=m_budget, stop_nonpositive=False)
        greedy_val = path.prefix_values[-1] if path.prefix_values else 0.0
        _, opt_val = exhaustive_greedy_ratio(obj, budget=m_budget)
        mono = is_monotone(obj, list(range(f)), max_sets=300, seed=trial)
        sub = is_submodular(obj, list(range(f)), max_checks=400, seed=trial)
        ratio_rows.append({
            "trial": trial, "f": f, "budget": m_budget,
            "greedy_value": greedy_val,
            "optimum_value": opt_val,
            "ratio": greedy_val / opt_val if opt_val > 0 else np.nan,
            "monotone": mono, "submodular": sub,
            "note": "empirical record; classical guarantee scope in THEORETICAL_SCOPE_AUDIT.md",
        })
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(out / "greedy_ratio_records.csv", index=False)
    if not ratio_df["monotone"].all() or not ratio_df["submodular"].all():
        failure_log(out / "failure_log.csv", "stats", "submodularity",
                    "monotonicity/submodularity check failed on a random tiny instance")
    paths["ratios"] = out / "greedy_ratio_records.csv"

    # theoretical scope doc lives in method audit but references these records
    from .audit import write_theoretical_scope_audit
    write_theoretical_scope_audit(ctx.evidence / "01_method_audit", ratio_df)
    return paths


# --------------------------- reporting validation ------------------------ #

def run_reporting_validation(ctx: RunContext) -> Dict[str, Path]:
    out = ctx.out("12_final_gate")
    reports = []
    checks_all = []

    eb = read_csv_or_none(ctx.evidence / "03_equal_budget" / "equal_budget_per_run.csv")
    if eb is not None and not eb.empty:
        summary = read_csv_or_none(ctx.evidence / "03_equal_budget" / "equal_budget_summary.csv")
        rep, chk = run_validations(eb, summary, ("dataset", "method", "budget"),
                                   "accuracy")
        reports.append(("equal_budget", rep))
        chk.insert(0, "table", "equal_budget")
        checks_all.append(chk)

    cg = read_csv_or_none(ctx.evidence / "05_classifier_generality" / "classifier_generality_per_run.csv")
    if cg is not None and not cg.empty:
        s = read_csv_or_none(ctx.evidence / "05_classifier_generality" / "classifier_generality_summary.csv")
        rep, chk = run_validations(cg, s, ("dataset", "method", "classifier"), "accuracy")
        reports.append(("classifier_generality", rep))
        chk.insert(0, "table", "classifier_generality")
        checks_all.append(chk)

    ab = read_csv_or_none(ctx.evidence / "07_ablation" / "ablation_per_run.csv")
    if ab is not None and not ab.empty:
        s = read_csv_or_none(ctx.evidence / "07_ablation" / "ablation_summary.csv")
        rep, chk = run_validations(ab, s, ("dataset", "variant"), "accuracy")
        reports.append(("ablation", rep))
        chk.insert(0, "table", "ablation")
        checks_all.append(chk)

    # mixed-unit detector on any formatted percent columns produced later
    overall = {
        "all_passed": all(r["all_passed"] for _, r in reports) if reports else False,
        "tables": {name: rep for name, rep in reports},
        "n_failed_tables": sum(0 if r["all_passed"] else 1 for _, r in reports),
    }
    checks = pd.concat(checks_all, ignore_index=True) if checks_all else pd.DataFrame(
        {"check": [], "status": [], "detail": [], "table": []})
    write_reporting_validation(overall, checks, out)
    paths = {"json": out / "reporting_validation.json",
             "csv": out / "table_consistency_checks.csv"}
    if not overall["all_passed"]:
        failure_log(out / "failure_log.csv", "reporting", "validation",
                    f"failed tables: {overall['n_failed_tables']}")
    return paths
