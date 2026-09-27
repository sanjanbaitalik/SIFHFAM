"""Canonical revision selector: screening -> IFS -> hypergraph -> coverage greedy.

Modes
-----
revision_canonical : corrected/transparent pipeline implemented here.
legacy_repro       : wraps ``reviewer_revision.sifhfam_selector.SIFHFAMSelector``
                     unchanged (the implementation behind Outputs/01_main_repeated).

The canonical object is intentionally NOT called "strong": the strongness
audit (prompt section 3) must be run before any strongness claim; a
``strict_strong`` hyperedge-degree variant exists for ablation only.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import RevisionConfig, choose_f, choose_m, manuscript_m
from .discretization import Discretizer
from .hypergraph import HyperedgeCollection, generate_hyperedges, theoretical_counts
from .ifs import (
    EdgeDegrees,
    VertexDegrees,
    compute_vertex_degrees,
    independent_edge_degrees,
    strict_strong_induced_edge_degrees,
    strong_relation_audit,
    validate_ifs,
)
from .information import EPS, entropy
from .objective import Objective, GreedyResult, greedy_path, select_by_efficiency_ratio
from .screening import ScreeningResult, screen


@dataclass
class StageTimings:
    preprocess_sec: float = 0.0
    screening_score_sec: float = 0.0
    screening_sort_sec: float = 0.0
    discretization_sec: float = 0.0
    vertex_relevance_sec: float = 0.0
    vertex_redundancy_sec: float = 0.0
    hyperedge_generation_sec: float = 0.0
    hyperedge_scoring_sec: float = 0.0
    edge_weight_sec: float = 0.0
    greedy_sec: float = 0.0
    total_fs_sec: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


@dataclass
class OpCounts:
    d: int = 0
    f: int = 0
    n_train: int = 0
    pairwise_redundancy_evals: int = 0
    theoretical_edges_by_order: Dict[int, int] = field(default_factory=dict)
    evaluated_edges_by_order: Dict[int, int] = field(default_factory=dict)
    n_edges: int = 0
    n_positive_edges: int = 0
    n_entropy_calls: int = 0
    n_mi_calls: int = 0
    n_joint_entropy_calls: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "d": self.d, "f": self.f, "n_train": self.n_train,
            "pairwise_redundancy_evals": self.pairwise_redundancy_evals,
            "theoretical_edges_by_order": {str(k): v for k, v in self.theoretical_edges_by_order.items()},
            "evaluated_edges_by_order": {str(k): v for k, v in self.evaluated_edges_by_order.items()},
            "n_edges": self.n_edges, "n_positive_edges": self.n_positive_edges,
            "n_entropy_calls": self.n_entropy_calls, "n_mi_calls": self.n_mi_calls,
            "n_joint_entropy_calls": self.n_joint_entropy_calls,
        }


@dataclass
class MemoryRecord:
    rss_before_bytes: Optional[int] = None
    rss_after_bytes: Optional[int] = None
    rss_peak_delta_bytes: Optional[int] = None
    tracemalloc_peak_bytes: Optional[int] = None
    backend: str = "none"

    def to_dict(self) -> Dict[str, object]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SelectionResult:
    method: str
    selected: np.ndarray                 # original feature indices
    greedy_order_screened: np.ndarray    # ranking in screened space
    prefix_values: List[float]
    screened_idx: np.ndarray
    scores: np.ndarray                   # full-d screening scores
    vertex_mu: np.ndarray
    vertex_nu: np.ndarray
    vertex_pi: np.ndarray
    edge_weights: Dict[Tuple[int, ...], float]
    edge_mu: Dict[Tuple[int, ...], float]
    edge_nu: Dict[Tuple[int, ...], float]
    edge_pi: Dict[Tuple[int, ...], float]
    timings: StageTimings
    ops: OpCounts
    memory: MemoryRecord
    ifs_checks: Dict[str, bool]
    strongness_rows: List[Dict[str, object]]
    strongness_summary: Dict[str, object]
    hyperedge_info: Dict[str, object]
    config: RevisionConfig
    m: int
    f: int
    d: int
    objective_value: float
    notes: str = ""
    error: Optional[str] = None

    @property
    def fs_time_sec(self) -> float:
        return self.timings.total_fs_sec


def _rss() -> Optional[int]:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def fit_canonical(X_train: np.ndarray, y_train: np.ndarray, cfg: RevisionConfig,
                  max_path_budget: Optional[int] = None) -> SelectionResult:
    """Run the revision_canonical feature-selection pipeline on the train fold."""
    t_start = time.perf_counter()
    timings = StageTimings()
    ops = OpCounts()
    mem = MemoryRecord(rss_before_bytes=_rss())

    use_tm = False
    try:
        import tracemalloc
        tracemalloc.start()
        use_tm = True
    except Exception:
        use_tm = False

    try:
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train).ravel()
        n, d = X.shape
        ops.n_train = n
        ops.d = d

        # ---------------- screening (train only) ----------------
        if cfg.screening_strategy == "none":
            f_budget = d
        else:
            f_budget = choose_f(d, cfg)
        scr: ScreeningResult = screen(
            X, y, f_budget, strategy=cfg.screening_strategy, bins=cfg.bins,
            discretizer=cfg.discretizer, seed=cfg.random_state,
            union_share=cfg.screening_union_share,
        )
        timings.screening_score_sec = scr.fit_seconds
        timings.screening_sort_sec = scr.sort_seconds
        screened = scr.screened_idx
        f = len(screened)
        ops.f = f

        # ---------------- discretization (train only) ----------------
        t0 = time.perf_counter()
        disc = Discretizer(method=cfg.discretizer, bins=cfg.bins).fit(X[:, screened])
        X_disc = disc.transform(X[:, screened])
        timings.discretization_sec = time.perf_counter() - t0

        # ---------------- vertex degrees ----------------
        from .ifs import ifs_from_evidences, map_evidence
        from .information import nmi as _nmi
        from .ifs import vertex_redundancy_scores

        t0 = time.perf_counter()
        from .information import feature_nmi_scores
        rel = feature_nmi_scores(X_disc, y, estimator=cfg.entropy_estimator)
        ops.n_mi_calls += f
        ops.n_entropy_calls += 3 * f  # H(X), H(Y), H(X,Y) inside each NMI
        timings.vertex_relevance_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        if cfg.vertex_redundancy == "nmi_mean":
            ops.pairwise_redundancy_evals = f * (f - 1)
            ops.n_mi_calls += f * (f - 1)
            ops.n_entropy_calls += 3 * f * (f - 1)
        elif cfg.vertex_redundancy == "corr_mean":
            ops.pairwise_redundancy_evals = f * (f - 1) // 2
        red = vertex_redundancy_scores(X_disc, kind=cfg.vertex_redundancy,
                                       estimator=cfg.entropy_estimator)
        red_e = red if cfg.use_nonmembership_degree else np.zeros_like(red)
        raw_mu = map_evidence(rel, cfg.evidence_map, cfg.a1, cfg.b1)
        raw_nu = map_evidence(red_e, cfg.evidence_map, cfg.a2, cfg.b2)
        mu_v, nu_v, pi_v = ifs_from_evidences(raw_mu, raw_nu)
        vd = VertexDegrees(mu=mu_v, nu=nu_v, pi=pi_v, relevance=rel, redundancy=red)
        timings.vertex_redundancy_sec = time.perf_counter() - t0

        # ---------------- hypergraph ----------------
        t0 = time.perf_counter()
        if cfg.use_hyperedges and cfg.K >= 2 and f >= 2:
            rng = np.random.default_rng(cfg.random_state)
            coll: HyperedgeCollection = generate_hyperedges(
                f, cfg.K, cfg.max_edges, rng=rng, seed=cfg.random_state,
                orders_mode=cfg.edge_orders,
            )
        else:
            coll = generate_hyperedges(f, min(cfg.K, 1), 0, seed=cfg.random_state)
        timings.hyperedge_generation_sec = time.perf_counter() - t0
        ops.theoretical_edges_by_order = theoretical_counts(f, cfg.K if cfg.use_hyperedges else 1)
        ops.evaluated_edges_by_order = dict(coll.per_order_sampled)
        ops.n_edges = len(coll.edges)

        # ---------------- hyperedge scoring ----------------
        t0 = time.perf_counter()
        marginals = [entropy(X_disc[:, j], estimator=cfg.entropy_estimator) for j in range(f)]
        ops.n_entropy_calls += f
        if coll.edges:
            if cfg.hyperedge_degrees == "strict_strong":
                ed = strict_strong_induced_edge_degrees(vd.mu, vd.nu, coll.edges)
                ops.n_mi_calls += len(coll.edges)
                ops.n_joint_entropy_calls += 0
            else:
                ed = independent_edge_degrees(
                    X_disc, y, coll.edges,
                    estimator=cfg.entropy_estimator,
                    evidence_map=cfg.evidence_map,
                    a3=cfg.a3, b3=cfg.b3, a4=cfg.a4, b4=cfg.b4,
                    use_nonmembership=cfg.use_nonmembership_degree,
                    marginals=marginals,
                )
                ops.n_mi_calls += len(coll.edges)
                ops.n_joint_entropy_calls += 2 * len(coll.edges)
        else:
            ed = EdgeDegrees(mu={}, nu={}, pi={}, group_relevance={},
                             group_redundancy={}, tc={})
        timings.hyperedge_scoring_sec = time.perf_counter() - t0

        # ---------------- edge weights ----------------
        t0 = time.perf_counter()
        lam = cfg.lam if cfg.use_redundancy else 0.0
        edge_weights: Dict[Tuple[int, ...], float] = {}
        for e, mu_e in ed.mu.items():
            score = mu_e - lam * ed.nu[e]
            w = max(0.0, score)
            if w > 0 and cfg.use_hesitation_weight:
                w *= (1.0 - ed.pi[e])
            edge_weights[e] = float(max(0.0, w))
        ops.n_positive_edges = sum(1 for w in edge_weights.values() if w > 0)
        timings.edge_weight_sec = time.perf_counter() - t0

        # ---------------- greedy ----------------
        t0 = time.perf_counter()
        m = choose_m(d, f, cfg)
        if max_path_budget is None:
            max_path_budget = m
        budget = int(max(m, min(max_path_budget, f)))
        obj = Objective(
            vertex_mu=vd.mu,
            edge_weights=edge_weights,
            alpha=cfg.alpha,
            beta=cfg.beta,
            g_type=cfg.g_type,
            soft_rho=cfg.soft_rho,
        )
        path: GreedyResult = greedy_path(obj, budget)
        if cfg.cardinality_rule == "efficiency_ratio":
            k_min = max(5, int(np.sqrt(f)))
            selected_local = select_by_efficiency_ratio(path, k_min)
        else:
            selected_local = path.order[:m]
        timings.greedy_sec = time.perf_counter() - t0

        selected = screened[np.asarray(selected_local, dtype=int)] if len(selected_local) else np.array([], dtype=int)
        obj_val = obj.value(list(selected_local))

        # ---------------- audits ----------------
        ifs_checks = {
            **{f"vertex_{k}": v for k, v in validate_ifs(vd.mu, vd.nu, vd.pi).items()},
        }
        if ed.mu:
            emu = np.array([ed.mu[e] for e in ed.mu])
            enu = np.array([ed.nu[e] for e in ed.nu])
            epi = np.array([ed.pi[e] for e in ed.pi])
            ifs_checks.update({f"edge_{k}": v for k, v in validate_ifs(emu, enu, epi).items()})
        strong_rows, strong_sum = strong_relation_audit(vd.mu, vd.nu, ed, coll.edges)
        strong_sum["K"] = cfg.K
        strong_sum["hyperedge_degrees"] = cfg.hyperedge_degrees

        timings.total_fs_sec = time.perf_counter() - t_start
        mem.rss_after_bytes = _rss()
        if mem.rss_before_bytes and mem.rss_after_bytes:
            mem.rss_peak_delta_bytes = max(0, mem.rss_after_bytes - mem.rss_before_bytes)
        if use_tm:
            try:
                import tracemalloc as _tm
                mem.tracemalloc_peak_bytes = int(_tm.get_traced_memory()[1])
                _tm.stop()
            except Exception:
                pass
        mem.backend = "psutil+tracemalloc" if mem.rss_before_bytes is not None and use_tm else (
            "psutil" if mem.rss_before_bytes is not None else ("tracemalloc" if use_tm else "none"))

        return SelectionResult(
            method=cfg.name,
            selected=selected.astype(int),
            greedy_order_screened=np.asarray(path.order, dtype=int),
            prefix_values=path.prefix_values,
            screened_idx=screened.astype(int),
            scores=scr.scores,
            vertex_mu=vd.mu, vertex_nu=vd.nu, vertex_pi=vd.pi,
            edge_weights=edge_weights,
            edge_mu=ed.mu, edge_nu=ed.nu, edge_pi=ed.pi,
            timings=timings, ops=ops, memory=mem,
            ifs_checks=ifs_checks,
            strongness_rows=strong_rows,
            strongness_summary=strong_sum,
            hyperedge_info=coll.counts(),
            config=cfg, m=m, f=f, d=d,
            objective_value=float(obj_val),
            notes=scr.notes,
        )
    except Exception:
        timings.total_fs_sec = time.perf_counter() - t_start
        mem.rss_after_bytes = _rss()
        if use_tm:
            try:
                import tracemalloc as _tm
                mem.tracemalloc_peak_bytes = int(_tm.get_traced_memory()[1])
                _tm.stop()
            except Exception:
                pass
        d_guess = int(np.asarray(X_train).shape[1]) if X_train is not None else 0
        return SelectionResult(
            method=cfg.name,
            selected=np.array([], dtype=int),
            greedy_order_screened=np.array([], dtype=int),
            prefix_values=[],
            screened_idx=np.array([], dtype=int),
            scores=np.zeros(d_guess, dtype=float),
            vertex_mu=np.array([]), vertex_nu=np.array([]), vertex_pi=np.array([]),
            edge_weights={}, edge_mu={}, edge_nu={}, edge_pi={},
            timings=timings, ops=ops, memory=mem,
            ifs_checks={}, strongness_rows=[], strongness_summary={},
            hyperedge_info={}, config=cfg, m=0, f=0, d=d_guess, objective_value=0.0,
            notes="", error=traceback.format_exc(),
        )


def fit_legacy_repro(X_train: np.ndarray, y_train: np.ndarray,
                     cfg: RevisionConfig) -> SelectionResult:
    """Call the OLD reviewer-revision implementation without modifying it.

    Packaging note: the legacy ``reviewer_revision`` package (and the frozen
    legacy results it produced) is intentionally NOT distributed with this
    public source repository.  ``mode="legacy_repro"`` therefore raises an
    informative error here; the canonical ``mode="revision_canonical"``
    pipeline is unaffected.
    """
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from reviewer_revision.config import SIFHFAMConfig
        from reviewer_revision.sifhfam_selector import SIFHFAMSelector
    except ImportError as exc:
        raise RuntimeError(
            "mode='legacy_repro' requires the internal research repository "
            "(the legacy `reviewer_revision` package), which is not part of "
            "this public source distribution. Use the default "
            "mode='revision_canonical', or run legacy comparisons inside the "
            "research repository where that package exists."
        ) from exc

    t_start = time.perf_counter()
    old_cfg = SIFHFAMConfig(
        a1=cfg.a1, b1=cfg.b1, a2=cfg.a2, b2=cfg.b2,
        a3=cfg.a3, b3=cfg.b3, a4=cfg.a4, b4=cfg.b4,
        lam=cfg.lam, alpha=cfg.alpha, beta=cfg.beta,
        bins=cfg.bins, K=cfg.K, max_edges=cfg.max_edges,
        random_state=cfg.random_state, m=cfg.m,
        use_hyperedges=cfg.use_hyperedges,
        use_redundancy=cfg.use_redundancy,
        use_hesitation_weight=cfg.use_hesitation_weight,
        name="SIFHFAM_legacy_repro",
    )
    sel = SIFHFAMSelector(old_cfg)
    res = sel.fit_select(X_train, y_train)
    timings = StageTimings(total_fs_sec=time.perf_counter() - t_start)
    ops = OpCounts(d=X_train.shape[1], f=res.ifs_stats.n_screened_features,
                   n_train=X_train.shape[0],
                   n_edges=res.ifs_stats.n_edges_total,
                   n_positive_edges=res.ifs_stats.n_edges_positive)
    mem = MemoryRecord(rss_before_bytes=_rss(), rss_after_bytes=_rss(), backend="psutil")
    d = X_train.shape[1]
    return SelectionResult(
        method="SIFHFAM_legacy_repro",
        selected=np.asarray(res.selected_indices, dtype=int),
        greedy_order_screened=np.asarray(res.selected_screened_indices, dtype=int),
        prefix_values=[],
        screened_idx=np.asarray(res.screened_indices, dtype=int),
        scores=np.asarray(res.scores, dtype=float),
        vertex_mu=res.vertex_mu, vertex_nu=res.vertex_nu, vertex_pi=res.vertex_pi,
        edge_weights=res.edge_weights,
        edge_mu=res.edge_mu, edge_nu=res.edge_nu, edge_pi=res.edge_pi,
        timings=timings, ops=ops, memory=mem,
        ifs_checks={},
        strongness_rows=[],
        strongness_summary={},
        hyperedge_info={"legacy": True, "n_edges": res.ifs_stats.n_edges_total},
        config=cfg, m=len(res.selected_indices),
        f=res.ifs_stats.n_screened_features, d=d,
        objective_value=float("nan"),
        notes="legacy reviewer-revision selector (unchanged call)",
    )


def fit(X_train: np.ndarray, y_train: np.ndarray, cfg: RevisionConfig,
        max_path_budget: Optional[int] = None) -> SelectionResult:
    if cfg.mode == "legacy_repro":
        return fit_legacy_repro(X_train, y_train, cfg)
    return fit_canonical(X_train, y_train, cfg, max_path_budget=max_path_budget)
