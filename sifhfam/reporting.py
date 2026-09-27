"""Numerical reporting validation (prompt section 15).

Rules enforced:
- accuracy stored internally in [0,1]
- percent display multiplies mean AND SD by 100
- never display `95.45 +- 0.0552` (mixed units)
- mean rounded to 100.00% must retain precision to explain nonzero SD
- selected feature count integer per run
- feature reduction == 1 - k/d
- table means recompute exactly from raw rows within tolerance
- no manually entered aggregates
Failed checks BLOCK final report generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def check_accuracy_range(df: pd.DataFrame, col: str = "accuracy") -> Tuple[bool, str]:
    if col not in df.columns:
        return False, f"missing column {col}"
    v = df[col].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return False, "no finite accuracy values"
    if v.min() < 0.0 or v.max() > 1.0:
        return False, f"accuracy outside [0,1]: min={v.min()}, max={v.max()}"
    return True, "ok"


def check_percent_units(mean: float, sd: float, as_percent: bool,
                        raw_mean: float, raw_sd: float) -> Tuple[bool, str]:
    """If displayed as percent, BOTH mean and SD must be *100 of fraction values."""
    if not as_percent:
        return True, "fraction display"
    exp_mean, exp_sd = raw_mean * 100.0, raw_sd * 100.0
    if abs(mean - exp_mean) > 1e-6 or abs(sd - exp_sd) > 1e-6:
        return False, (f"unit mismatch: displayed mean={mean}, sd={sd}; "
                       f"expected {exp_mean}, {exp_sd}")
    return True, "ok"


def check_no_mixed_units(mean_pct: float, sd_fraction: float) -> Tuple[bool, str]:
    """Detect the classic `95.45 +- 0.0552` pattern (percent mean, fraction SD)."""
    if 1.0 < mean_pct <= 100.0 and 0.0 <= sd_fraction < 1.0 and sd_fraction > 0:
        # could still be legit if sd also displayed in percent (>=1 typically for small sd)
        # treat sd < 1 together with mean > 1 as suspicious mixed units
        return False, (f"suspicious mixed units: mean={mean_pct} looks like percent "
                       f"while sd={sd_fraction} looks like a fraction")
    return True, "ok"


def check_mean_sd_rounding(mean_pct: float, sd_pct: float, decimals: int = 2) -> Tuple[bool, str]:
    """If mean rounds to 100.00%, SD must still be explainable at that precision."""
    if round(mean_pct, decimals) == 100.0 and sd_pct > 0 and round(sd_pct, decimals) == 0.0:
        return False, (f"mean rounds to 100.00% but SD={sd_pct} rounds to 0.00 at "
                       f"{decimals} decimals; keep more precision or lower rounding")
    return True, "ok"


def check_cardinality_integer(df: pd.DataFrame, col: str = "n_selected") -> Tuple[bool, str]:
    if col not in df.columns:
        return False, f"missing column {col}"
    v = df[col].to_numpy(dtype=float)
    if not np.all(np.isfinite(v)):
        return False, "non-finite n_selected"
    if not np.allclose(v, np.round(v)):
        return False, "n_selected not integer in every run"
    return True, "ok"


def check_reduction_arithmetic(df: pd.DataFrame, k_col: str = "n_selected",
                               d_col: str = "d",
                               red_col: str = "feature_reduction") -> Tuple[bool, str]:
    for c in (k_col, d_col, red_col):
        if c not in df.columns:
            return False, f"missing column {c}"
    k = df[k_col].to_numpy(dtype=float)
    d = df[d_col].to_numpy(dtype=float)
    red = df[red_col].to_numpy(dtype=float)
    # rows with NaN accuracy/reduction are failed runs; arithmetic is checked
    # on completed rows only
    mask = np.isfinite(k) & np.isfinite(d) & np.isfinite(red)
    if not mask.any():
        return True, "no finite rows to check"
    expected = 1.0 - k[mask] / d[mask]
    if not np.allclose(expected, red[mask], atol=1e-9):
        return False, "feature_reduction != 1 - k/d"
    return True, "ok"


def check_summary_recomputes(raw: pd.DataFrame, summary: pd.DataFrame,
                             group_cols: Sequence[str], value_col: str,
                             atol: float = 1e-9) -> Tuple[bool, str]:
    if summary is None or summary.empty:
        return False, "empty summary"
    recomputed = raw.groupby(list(group_cols))[value_col].agg(["mean", "std"]).reset_index()
    # locate summary mean/std columns (supports 'mean' and '<value>_mean' styles)
    if f"{value_col}_mean" in summary.columns and f"{value_col}_std" in summary.columns:
        sm = summary[list(group_cols) + [f"{value_col}_mean", f"{value_col}_std"]].copy()
        sm = sm.rename(columns={f"{value_col}_mean": "mean", f"{value_col}_std": "std"})
    elif "mean" in summary.columns and "std" in summary.columns:
        sm = summary[list(group_cols) + ["mean", "std"]].copy()
    else:
        return False, "summary lacks mean/std (or <value>_mean/<value>_std) columns"
    rec = recomputed.rename(columns={"mean": "mean_rec", "std": "std_rec"})
    merged = sm.merge(rec, on=list(group_cols), how="inner")
    if merged.empty:
        return False, "empty merge between summary and raw"
    for col in ("mean", "std"):
        a = merged[col].to_numpy(dtype=float)
        b = merged[f"{col}_rec"].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.any() and not np.allclose(a[mask], b[mask], atol=atol):
            return False, f"summary {col} does not recompute from raw rows"
    return True, "ok"


def run_validations(raw: pd.DataFrame,
                    summary: Optional[pd.DataFrame] = None,
                    group_cols: Sequence[str] = ("dataset", "method"),
                    value_col: str = "accuracy") -> Tuple[Dict[str, object], pd.DataFrame]:
    """Run all checks; returns (report_dict, checks_csv_rows)."""
    rows: List[Dict[str, object]] = []

    def add(name: str, ok: bool, detail: str):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    ok, d = check_accuracy_range(raw, value_col)
    add("accuracy_in_0_1", ok, d)
    if "n_selected" in raw.columns:
        ok, d = check_cardinality_integer(raw, "n_selected")
        add("n_selected_integer", ok, d)
    if {"n_selected", "d", "feature_reduction"} <= set(raw.columns):
        ok, d = check_reduction_arithmetic(raw)
        add("reduction_arithmetic", ok, d)
    if summary is not None:
        ok, d = check_summary_recomputes(raw, summary, group_cols, value_col)
        add("summary_recomputes_from_raw", ok, d)
        if "accuracy_mean" in summary.columns and "accuracy_std" in summary.columns:
            # stored as fractions
            am = summary["accuracy_mean"].to_numpy(dtype=float)
            if np.nanmax(am) > 1.0:
                add("summary_mean_fraction_scale", False,
                    f"summary mean > 1: max={np.nanmax(am)}")
            else:
                add("summary_mean_fraction_scale", True, "ok")
    # mixed-unit detector on any formatted columns if present
    if "formatted_percent_mean" in raw.columns and "formatted_percent_sd" in raw.columns:
        for _, r in raw.iterrows():
            ok, d = check_no_mixed_units(float(r["formatted_percent_mean"]),
                                         float(r["formatted_percent_sd"]))
            if not ok:
                add("no_mixed_units", False, d)
                break
        else:
            add("no_mixed_units", True, "ok")

    checks = pd.DataFrame(rows)
    all_ok = bool((checks["status"] == "PASS").all()) if not checks.empty else False
    report = {
        "all_passed": all_ok,
        "n_checks": int(len(checks)),
        "n_failed": int((checks["status"] == "FAIL").sum()),
        "failed": checks.loc[checks["status"] == "FAIL", "check"].tolist() if not checks.empty else [],
    }
    return report, checks


def write_reporting_validation(report: Dict[str, object], checks: pd.DataFrame,
                               out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "reporting_validation.json"
    csv_path = out_dir / "table_consistency_checks.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    checks.to_csv(csv_path, index=False)
    return json_path, csv_path


def assert_reporting_ok(report: Dict[str, object]) -> None:
    """Any failed check must stop final report generation."""
    if not report.get("all_passed", False):
        failed = report.get("failed", [])
        raise AssertionError(
            "REPORTING VALIDATION FAILED: " + ", ".join(map(str, failed))
        )


def format_mean_sd(mean: float, sd: float, as_percent: bool = False,
                   decimals: int = 2) -> str:
    """Consistent 'mean +- sd' formatting with matched units."""
    if as_percent:
        m, s = mean * 100.0, sd * 100.0
        # keep extra precision when 2-decimals would hide a nonzero SD
        if round(s, decimals) == 0.0 and s > 0:
            return f"{m:.4f} +- {s:.4f} %"
        return f"{m:.{decimals}f} +- {s:.{decimals}f} %"
    if round(sd, decimals) == 0.0 and sd > 0:
        return f"{mean:.4f} +- {sd:.4f}"
    return f"{mean:.{decimals}f} +- {sd:.{decimals}f}"
