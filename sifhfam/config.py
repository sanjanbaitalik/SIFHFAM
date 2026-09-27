"""Configuration objects for the Neurocomputing revision experiments."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Sequence

PAPER14_DATASETS: List[str] = [
    "ALLAML", "CLL_SUB_111", "GLIOMA", "GLI_85", "Prostate_GE",
    "SMK_CAN_187", "TOX_171", "arcene", "colon", "leukemia",
    "lung", "lymphoma", "madelon", "gisette",
]

# Representative datasets chosen BEFORE observing any new revision results
# (prompt section 11).  One small-n microarray, one medium, one larger-n,
# plus one high-dimension (madelon) dataset.
REPRESENTATIVE_DATASETS_PRESELECTED: List[str] = ["colon", "GLI_85", "ALLAML", "madelon"]

DEFAULT_SEED = 42


@dataclass(frozen=True)
class RevisionConfig:
    """Full configuration of the revision_canonical SIFHFAM pipeline.

    Evidence-to-degree mapping
    --------------------------
    raw evidence values are optionally passed through a sigmoid with a fixed
    midpoint (evidence = 0.5 maps to sigmoid(0) = 0.5 when b = -a/2), then the
    IFS-safe normalization
        mu = raw_mu / (1 + raw_mu + raw_nu), etc.
    guarantees mu + nu + pi = 1 and mu + nu <= 1.
    """

    # ---- sigmoid mapping (vertex) ----
    a1: float = 5.0   # relevance slope
    b1: float = -2.5  # relevance intercept (midpoint 0.5)
    a2: float = 5.0   # redundancy slope
    b2: float = -2.5

    # ---- sigmoid mapping (hyperedge) ----
    a3: float = 5.0
    b3: float = -2.5
    a4: float = 5.0
    b4: float = -2.5

    # ---- objective ----
    lam: float = 0.5
    alpha: float = 1.0
    beta: float = 1.0
    g_type: str = "binary"          # binary | soft | fraction
    soft_rho: float = 0.5

    # ---- discretization / hypergraph ----
    bins: int = 10
    discretizer: str = "quantile"   # quantile | uniform
    K: int = 3
    edge_orders: str = "range_2_K"   # range_2_K | exact_K (ablation)
    max_edges: int = 5000
    f: Optional[int] = None         # screening budget; None -> heuristic
    min_f: int = 50
    max_f: int = 300
    screening_strategy: str = "mi"  # mi | anova_f | relieff | interaction_union | none
    screening_union_share: float = 0.5  # share of f given to MI half in interaction_union

    # ---- entropy estimator ----
    entropy_estimator: str = "plugin"     # plugin | miller_madow

    # ---- cardinality ----
    m: Optional[int] = None         # None -> manuscript rule max(10, round(sqrt(d)))
    cardinality_rule: str = "manuscript_sqrt"  # manuscript_sqrt | fixed | efficiency_ratio | threshold

    # ---- ablation switches ----
    use_hyperedges: bool = True
    use_redundancy: bool = True           # redundancy penalty in edge weight (lam)
    use_hesitation_weight: bool = True
    use_nonmembership_degree: bool = True  # False -> nu evidence forced to 0 BEFORE IFS norm
    hyperedge_degrees: str = "independent"  # independent | strict_strong
    vertex_redundancy: str = "nmi_mean"     # nmi_mean | corr_mean
    evidence_map: str = "sigmoid"           # sigmoid | direct

    # ---- protocol ----
    random_state: int = DEFAULT_SEED
    n_runs: int = 10
    test_size: float = 0.25
    mode: str = "revision_canonical"  # revision_canonical | legacy_repro
    name: str = "SIFHFAM"

    # thread control (hard constraint 14)
    n_jobs: int = 1

    def with_updates(self, **kwargs) -> "RevisionConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def choose_f(d: int, cfg: RevisionConfig) -> int:
    """Screening budget.  Identical heuristic to the legacy config."""
    if cfg.f is not None:
        return int(max(1, min(cfg.f, d)))
    heuristic = int(math.floor(math.sqrt(max(d, 2)) * math.log(max(d, 3))))
    return int(max(1, min(d, max(cfg.min_f, min(cfg.max_f, heuristic)))))


def manuscript_m(d: int) -> int:
    """Manuscript SIFHFAM cardinality rule: max(10, round(sqrt(d)))."""
    return int(max(10, round(math.sqrt(max(d, 1)))))


def choose_m(d: int, f: int, cfg: RevisionConfig) -> int:
    if cfg.m is not None:
        return int(max(1, min(cfg.m, f, d)))
    if cfg.cardinality_rule == "manuscript_sqrt":
        return int(max(1, min(manuscript_m(d), f, d)))
    if cfg.cardinality_rule == "efficiency_ratio":
        # ratio-based rule used by legacy sif_hfam greedy_select_ratio_based
        return int(max(1, min(max(5, int(math.sqrt(f))), f, d)))
    if cfg.cardinality_rule == "threshold":
        return int(max(1, min(f, d)))
    return int(max(1, min(manuscript_m(d), f, d)))


def budget_grid(d: int, m_manuscript: Optional[int] = None) -> List[int]:
    """Common equal-budget grid per dataset (prompt section 8).

    unique(sorted(clipped([10, 25, 50, 100, manuscript_m])))
    with values > d removed.
    """
    base = [10, 25, 50, 100]
    mm = m_manuscript if m_manuscript is not None else manuscript_m(d)
    vals = sorted(set(b for b in base + [mm] if b <= d))
    return [int(v) for v in vals if v >= 1]


def default_canonical_methods() -> List[str]:
    return [
        "SIFHFAM_canonical",
        "SIFFAM_vertex_only",
        "mRMR",
        "ReliefF",
        "CMIM",
        "JMIM",
    ]


def profile_grids(profile: str) -> Dict[str, object]:
    """Reduced/standard/expanded experiment grids per CLI profile."""
    smoke = {
        "n_runs": 2,
        "synthetic_seeds": 10,
        "sensitivity_datasets": ["colon"],
        "lambda_grid": [0.0, 0.5, 2.0],
        "ab_grid": [(1.0, 0.0), (1.0, 1.0)],
        "K_grid": [2, 3],
        "max_edges_grid": [1000, 5000],
        "bins_grid": [5, 10],
        "discretizer_grid": ["quantile", "uniform"],
        "slope_grid": [2.5, 10.0],
        "screening_budget_multipliers": [0.5, 1.0, 2.0],
        "scaling_d": [500, 1000],
        "ablation_subset": [
            "full", "vertex_only", "K2", "K3", "no_redundancy_penalty",
            "no_hesitation", "binary_coverage", "soft_coverage",
            "no_screen", "strict_strong",
        ],
        "stability": {"n_splits": 4, "n_hyperedge_seeds": 3, "n_bootstrap": 5},
        "classifiers": ["RF", "SVM", "KNN", "LR"],
        "expensive_baselines": [],
        "quickselection_epochs": 2,
    }
    core = {
        "n_runs": 10,
        "synthetic_seeds": 20,
        "sensitivity_datasets": REPRESENTATIVE_DATASETS_PRESELECTED,
        "lambda_grid": [0.0, 0.25, 0.5, 1.0, 2.0],
        "ab_grid": [(1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0), (0.0, 1.0)],
        "K_grid": [2, 3, 4],
        "max_edges_grid": [1000, 5000, 10000],
        "bins_grid": [5, 7, 10, 15],
        "discretizer_grid": ["quantile", "uniform"],
        "slope_grid": [2.5, 5.0, 10.0],
        "screening_budget_multipliers": [0.5, 1.0, 2.0],
        "scaling_d": [500, 1000, 2000],
        "ablation_subset": [
            "full", "vertex_only", "K2", "K3", "K4",
            "no_nonmembership_degree", "no_redundancy_penalty", "no_hesitation",
            "binary_coverage", "soft_coverage", "fraction_coverage",
            "alternative_hyperedges", "alternative_discretization",
            "alternative_estimator", "screen_budget_low", "screen_budget_high",
            "screen_no", "strict_strong", "independent_group",
            "screening_anova", "screening_interaction_union",
        ],
        "stability": {"n_splits": 6, "n_hyperedge_seeds": 4, "n_bootstrap": 10},
        "classifiers": ["RF", "SVM", "KNN", "LR"],
        "expensive_baselines": ["FRFS", "PPFS"],
        "quickselection_epochs": 3,
    }
    full = {
        **core,
        "n_runs": 10,
        "synthetic_seeds": 20,
        "scaling_d": [500, 1000, 2000, 5000, 10000, 20000],
        "expensive_baselines": ["FRFS", "PPFS", "QuickSelection", "FCBF"],
        "quickselection_epochs": 5,
        "stability": {"n_splits": 8, "n_hyperedge_seeds": 5, "n_bootstrap": 20,
                      "consensus": True},
        "scaling_f_grid": [50, 100, 200, 300],
        "scaling_edge_grid": [1000, 5000, 10000],
        "scaling_K_grid": [2, 3, 4],
        "quickselection_epochs": 3,
    }
    if profile == "smoke":
        return smoke
    if profile == "core":
        return core
    if profile == "full":
        return full
    raise ValueError(f"Unknown profile: {profile}")
