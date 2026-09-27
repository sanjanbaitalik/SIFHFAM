"""Code-generated figures: method schematic + plots derived from raw CSVs only."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def method_schematic(out_dir: Path) -> Dict[str, Path]:
    """Reproducible, non-decorative method schematic (reviewer 4.6).

    Steps: train-fold data -> relevance screening -> IFS vertex degrees ->
    sampled/constructed hyperedges -> group relevance/redundancy ->
    non-negative hyperedge weights -> representative/soft coverage objective ->
    greedy selection -> held-out evaluation.
    """
    plt = _plt()
    steps = [
        ("1. Train-fold data\n(X_train, y_train only)", "#DDEBF7"),
        ("2. Relevance screening\n(univariate NMI, top-f)", "#DDEBF7"),
        ("3. IFS vertex degrees\n(mu_V, nu_V, pi_V)", "#E2EFDA"),
        ("4. Sampled / constructed\nhyperedges (orders 2..K)", "#FFF2CC"),
        ("5. Group relevance &\nredundancy  R_e = TC/(sum H)", "#FFF2CC"),
        ("6. Non-negative hyperedge\nweights w_e >= 0", "#FCE4D6"),
        ("7. Coverage objective\ng binary / soft / fraction", "#FCE4D6"),
        ("8. Greedy selection\n(prefix path, |S| = m)", "#E4DFEC"),
        ("9. Held-out evaluation\n(fixed RF/SVM/kNN/LR)", "#D9D9D9"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    w, h = 6.4, 1.15
    y_positions = np.linspace(11.0, 0.7, len(steps))
    for i, ((label, color), y) in enumerate(zip(steps, y_positions)):
        x = 1.4 if i % 2 == 0 else 2.4
        ax.add_patch(plt.Rectangle((x, y - h / 2), w, h, facecolor=color,
                                   edgecolor="#333333", linewidth=1.2))
        ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=10.5)
        if i < len(steps) - 1:
            y2 = y_positions[i + 1]
            x2 = 1.4 if (i + 1) % 2 == 0 else 2.4
            ax.annotate("", xy=(x2 + w / 2, y2 + h / 2),
                        xytext=(x + w / 2, y - h / 2),
                        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.4))
    ax.text(5.4, 12.6, "Revision-canonical feature-selection pipeline (code-generated schematic)",
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(5.4, 12.1, "All stages fit on the training fold; held-out test used once for evaluation",
            ha="center", va="center", fontsize=9.5, color="#444444")
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("png", "pdf", "svg"):
        p = out_dir / f"method_schematic.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths[ext] = p
    plt.close(fig)
    return paths


def budget_plots(evidence: Path) -> Dict[str, Path]:
    """Accuracy-vs-cardinality plots derived ONLY from raw CSVs."""
    plt = _plt()
    avc_path = evidence / "03_equal_budget" / "accuracy_vs_cardinality.csv"
    out_dir = evidence / "03_equal_budget" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    if not avc_path.exists():
        return paths
    avc = pd.read_csv(avc_path)
    for ds, g in avc.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, gm in g.groupby("method"):
            gm = gm.sort_values("n_selected_mean")
            ax.errorbar(gm["n_selected_mean"], gm["accuracy_mean"],
                        yerr=gm["accuracy_std"].fillna(0), marker="o", capsize=3,
                        label=method)
        ax.set_xlabel("number of selected features (mean)")
        ax.set_ylabel("test accuracy (mean $\\pm$ sd)")
        ax.set_title(f"Accuracy vs cardinality: {ds}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / f"accuracy_vs_cardinality_{ds}.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths[f"avc_{ds}"] = p

    # pareto frontier scatter (all datasets in one figure if small)
    pareto_path = evidence / "03_equal_budget" / "accuracy_sparsity_pareto.csv"
    if pareto_path.exists():
        pareto = pd.read_csv(pareto_path)
        fig, ax = plt.subplots(figsize=(7, 5))
        for ds, g in pareto.groupby("dataset"):
            ax.scatter(g["n_selected_mean"], g["accuracy_mean"], s=12, label=ds, alpha=0.7)
        ax.set_xlabel("mean n_selected")
        ax.set_ylabel("mean accuracy")
        ax.set_title("Accuracy-sparsity points (per dataset/method/budget)")
        ax.grid(alpha=0.3)
        if pareto["dataset"].nunique() <= 14:
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = out_dir / "accuracy_sparsity_pareto_all.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths["pareto"] = p
    return paths


def run_figures(evidence: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    paths.update({f"schematic_{k}": v for k, v in method_schematic(
        evidence / "11_reproducibility" / "figures").items()})
    paths.update(budget_plots(evidence))
    # keep schematic generator source alongside figures for reproducibility
    src = Path(__file__)
    target = evidence / "11_reproducibility" / "figures" / "method_schematic_source.py"
    target.write_text(
        "# Source-of-truth for method_schematic.* figures\n"
        "# Generated copy of sifhfam/figures.py::method_schematic\n"
        + src.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    paths["schematic_source"] = target
    return paths
