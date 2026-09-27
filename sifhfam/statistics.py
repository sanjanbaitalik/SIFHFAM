"""Across-dataset statistics with dataset as the paired unit (prompt section 16).

All tests are two-sided; no one-sided alternative is used anywhere.
Raw inputs to each test are saved alongside the results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def holm_correction(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        val = (n - i) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def wilcoxon_vs(ref_col: np.ndarray, other: np.ndarray) -> Dict[str, float]:
    """Two-sided paired Wilcoxon on the finite intersection only."""
    from scipy.stats import wilcoxon
    mask = np.isfinite(ref_col) & np.isfinite(other)
    a, b = ref_col[mask], other[mask]
    if a.size < 2:
        return {"n": int(a.size), "statistic": float("nan"), "p_value": float("nan"),
                "mean_diff": float("nan"), "median_diff": float("nan")}
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        stat, p = float("nan"), 1.0
    diff = a - b
    return {
        "n": int(a.size),
        "statistic": float(stat),
        "p_value": float(p),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
    }


def paired_comparison(ref_col: np.ndarray, other: np.ndarray, *,
                      method: str, ref_method: str,
                      dataset_labels: Optional[Sequence[str]] = None,
                      alpha: float = 0.05) -> Dict[str, object]:
    """One baseline comparison whose test AND descriptives use the exact same
    paired dataset intersection.

    Every mean reported here is computed only over datasets where BOTH the
    reference and the comparator have finite scores (``paired_datasets``).
    Legacy column aliases (``n``, ``ref_mean``, ``baseline_mean``, ... ) are
    retained and now also carry the *paired* quantities.
    """
    ref_col = np.asarray(ref_col, dtype=float)
    other = np.asarray(other, dtype=float)
    if ref_col.shape != other.shape:
        raise ValueError("paired_comparison requires aligned arrays")
    mask = np.isfinite(ref_col) & np.isfinite(other)
    a, b = ref_col[mask], other[mask]
    labels: List[str] = []
    if dataset_labels is not None:
        labels = [str(x) for x, m in zip(dataset_labels, mask) if bool(m)]

    base: Dict[str, object] = {
        "method": method,
        "baseline": method,                      # legacy alias
        "reference": ref_method,
        "n_pairs": int(a.size),
        "n": int(a.size),                        # legacy alias
        "paired_datasets": ";".join(labels),
    }
    if a.size == 0:
        base.update({
            "reference_mean_paired": float("nan"),
            "comparator_mean_paired": float("nan"),
            "mean_difference_paired": float("nan"),
            "median_difference": float("nan"),
            "wilcoxon_statistic": float("nan"),
            "statistic": float("nan"),
            "p_raw": float("nan"),
            "p_value": float("nan"),
            "ref_mean": float("nan"),
            "baseline_mean": float("nan"),
            "mean_diff": float("nan"),
            "median_diff": float("nan"),
            "effect_size_rank_biserial": float("nan"),
        })
        return base

    ref_paired = float(np.mean(a))
    cmp_paired = float(np.mean(b))
    diff = a - b
    if a.size < 2:
        stat, p = float("nan"), float("nan")
    else:
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            stat, p = float(stat), float(p)
        except Exception:
            stat, p = float("nan"), 1.0
    base.update({
        "reference_mean_paired": ref_paired,
        "comparator_mean_paired": cmp_paired,
        "mean_difference_paired": float(np.mean(diff)),
        "median_difference": float(np.median(diff)),
        "wilcoxon_statistic": stat,
        "statistic": stat,                        # legacy alias
        "p_raw": p,
        "p_value": p,                             # legacy alias
        "ref_mean": ref_paired,                   # legacy alias -> PAIRED mean
        "baseline_mean": cmp_paired,              # legacy alias -> PAIRED mean
        "mean_diff": float(np.mean(diff)),        # legacy alias
        "median_diff": float(np.median(diff)),    # legacy alias
        "effect_size_rank_biserial": paired_rank_biserial(ref_col, other),
        "alpha": alpha,
    })
    return base


def paired_rank_biserial(ref_col: np.ndarray, other: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation effect size (two-sided sign-free)."""
    mask = np.isfinite(ref_col) & np.isfinite(other)
    d = ref_col[mask] - other[mask]
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    from scipy.stats import rankdata
    ranks = rankdata(np.abs(d))
    r_pos = float(np.sum(ranks[d > 0]))
    r_neg = float(np.sum(ranks[d < 0]))
    return (r_pos - r_neg) / (r_pos + r_neg)


def run_statistics(df: pd.DataFrame, ref_method: str = "SIFHFAM_canonical",
                   value_col: str = "accuracy") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dataset-level paired comparisons.

    Two distinct analyses are produced and must be reported separately:

    1. **Global complete-case rank analysis** (Friedman omnibus + average
       ranks): computed only on datasets where EVERY retained method has a
       score.  The omnibus test indicates that at least some methods differ;
       it does NOT establish that the reference method is superior.
    2. **Baseline-specific paired comparisons** (Wilcoxon + Holm): for each
       baseline B, the test AND every descriptive mean use exactly the
       dataset intersection where both the reference and B are present.

    Returns (wilcoxon_df, friedman_df, ranks_df, raw_pivot).
    """
    from scipy.stats import friedmanchisquare, rankdata

    ds_method = df.groupby(["dataset", "method"], as_index=False)[value_col].mean()
    pivot = ds_method.pivot(index="dataset", columns="method", values=value_col)
    if ref_method not in pivot.columns:
        # fall back to the first available method name containing 'SIFHFAM'
        cands = [c for c in pivot.columns if "SIFHFAM" in str(c)]
        if not cands:
            raise ValueError(f"Reference method {ref_method} not in data")
        ref_method = cands[0]

    dataset_labels = [str(x) for x in pivot.index]
    rows: List[Dict[str, object]] = []
    for method in pivot.columns:
        if method == ref_method:
            continue
        row = paired_comparison(
            pivot[ref_method].to_numpy(dtype=float),
            pivot[method].to_numpy(dtype=float),
            method=str(method), ref_method=str(ref_method),
            dataset_labels=dataset_labels,
        )
        rows.append(row)
    wilcoxon_df = pd.DataFrame(rows)
    if not wilcoxon_df.empty:
        p_raw = wilcoxon_df["p_raw"].to_numpy(dtype=float)
        p_holm = holm_correction(p_raw)
        wilcoxon_df["p_holm"] = p_holm
        wilcoxon_df["significant_holm_0.05"] = [
            bool(np.isfinite(pv) and pv < 0.05) for pv in p_holm
        ]

    # --- global complete-case analysis (Friedman + ranks) ---
    complete = pivot.dropna(axis=1, how="any")
    complete = complete.dropna(axis=0, how="any")
    friedman_rows = []
    ranks_df = pd.DataFrame()
    if complete.shape[0] >= 3 and complete.shape[1] >= 3:
        stat, p = friedmanchisquare(*[complete[c].to_numpy() for c in complete.columns])
        excluded = [str(c) for c in pivot.columns if c not in complete.columns]
        friedman_rows.append({
            "n_datasets": int(complete.shape[0]),
            "n_methods": int(complete.shape[1]),
            "friedman_chi_square": float(stat),
            "p_value": float(p),
            "test": "friedman two-sided / omnibus",
            "analysis": "global complete-case rank analysis",
            "complete_case_methods": ";".join(str(c) for c in complete.columns),
            "excluded_methods": ";".join(excluded),
            "interpretation": ("omnibus only: at least some methods differ; "
                               "not a claim of reference-method superiority"),
        })
        rr = []
        for ds, row in complete.iterrows():
            ranks = rankdata(-row.to_numpy(), method="average")
            for method, rank in zip(complete.columns, ranks):
                rr.append({"dataset": ds, "method": method, "rank": float(rank)})
        ranks_df = (pd.DataFrame(rr).groupby("method", as_index=False)["rank"].mean()
                    .rename(columns={"rank": "average_rank"})
                    .sort_values("average_rank"))
        ranks_df["analysis"] = "complete_case"
    return wilcoxon_df, pd.DataFrame(friedman_rows), ranks_df, pivot


def save_stats_inputs(pivot: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(path)


def plot_rank_boxplot(ranks_df: pd.DataFrame, pivot: pd.DataFrame, path: Path) -> Optional[Path]:
    """Descriptive per-dataset rank distribution + average-rank panel.

    This figure is purely descriptive.  It deliberately does NOT draw any
    inferential post-hoc rank interval: inferential evidence comes from
    the Friedman omnibus test and the paired Wilcoxon tests with Holm
    correction reported in the statistics CSVs.
    """
    if ranks_df.empty:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import rankdata
    except Exception:
        return None

    complete = pivot.dropna(axis=1, how="any").dropna(axis=0, how="any")
    if complete.shape[1] < 2:
        return None
    rank_mat = pd.DataFrame(
        [rankdata(-row.to_numpy(), method="average") for _, row in complete.iterrows()],
        index=complete.index, columns=complete.columns,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    rank_mat[list(rank_mat.columns)].boxplot(ax=ax, rot=30)
    ax.set_ylabel("rank (1 = best)")
    ax.set_title(f"Per-dataset rank distribution (N={complete.shape[0]} datasets)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    order = ranks_df.sort_values("average_rank")
    ax.barh(order["method"], order["average_rank"], color="#4C72B0")
    avg = float(order["average_rank"].mean())
    ax.axvline(avg, color="gray", linestyle="--", linewidth=1,
               label=f"mean rank = {avg:.2f}")
    ax.set_title("Average ranks (descriptive)")
    ax.set_xlabel("average rank across complete-case datasets")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Rank summaries are descriptive; see statistics CSVs for tests",
                 fontsize=9, color="#444444")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# Backwards-compatible alias: descriptive rank figure under its new name.
plot_rank_distribution = plot_rank_boxplot
