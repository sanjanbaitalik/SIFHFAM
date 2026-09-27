"""Intuitionistic fuzzy sets: normalization, vertex/edge degrees, strongness audit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .information import EPS, group_redundancy, nmi, nmi_multivariate

TOL = 1e-9


def sigmoid(z: np.ndarray | float, slope: float = 1.0, midpoint: float = 0.5,
            intercept: float | None = None) -> np.ndarray | float:
    """Sigmoid with a clearly defined midpoint.

    If ``intercept`` is given: sigma(slope * z + intercept), midpoint at
    z = -intercept/slope.  Otherwise: sigma(slope * (z - midpoint)).
    Default slope=1, midpoint=0.5 gives sigma(0)=0.5 at z=0.5.
    """
    if intercept is None:
        b = -slope * midpoint
    else:
        b = intercept
    zz = np.clip(slope * np.asarray(z, dtype=float) + b, -60, 60)
    return 1.0 / (1.0 + np.exp(-zz))


def map_evidence(evidence: np.ndarray, mode: str, slope: float, intercept: float) -> np.ndarray:
    evidence = np.maximum(0.0, np.asarray(evidence, dtype=float))
    if mode == "sigmoid":
        return np.asarray(sigmoid(evidence, slope=slope, intercept=intercept), dtype=float)
    if mode == "direct":
        return evidence
    raise ValueError(f"Unknown evidence map: {mode}")


def ifs_from_evidences(raw_mu: np.ndarray, raw_nu: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IFS-safe normalization: mu+nu+pi = 1 exactly, mu+nu <= 1."""
    raw_mu = np.maximum(0.0, np.asarray(raw_mu, dtype=float))
    raw_nu = np.maximum(0.0, np.asarray(raw_nu, dtype=float))
    denom = 1.0 + raw_mu + raw_nu
    mu = raw_mu / denom
    nu = raw_nu / denom
    pi = 1.0 / denom
    return mu, nu, pi


def validate_ifs(mu: np.ndarray, nu: np.ndarray, pi: np.ndarray, tol: float = 1e-9) -> Dict[str, bool]:
    mu = np.asarray(mu, float); nu = np.asarray(nu, float); pi = np.asarray(pi, float)
    return {
        "mu_ge_0": bool(np.all(mu >= -tol)),
        "nu_ge_0": bool(np.all(nu >= -tol)),
        "pi_ge_0": bool(np.all(pi >= -tol)),
        "mu_nu_le_1": bool(np.all(mu + nu <= 1 + tol)),
        "sum_eq_1": bool(np.all(np.abs(mu + nu + pi - 1.0) <= tol)),
    }


@dataclass
class VertexDegrees:
    mu: np.ndarray
    nu: np.ndarray
    pi: np.ndarray
    relevance: np.ndarray
    redundancy: np.ndarray


def vertex_redundancy_scores(X_disc: np.ndarray, kind: str = "nmi_mean",
                             estimator: str = "plugin") -> np.ndarray:
    """Per-feature redundancy evidence on the *screened, discretized* train set.

    - ``nmi_mean`` (canonical): mean over i!=j of NMI(X_j; X_i)  -> information
      theoretic, bounded in [0,1].
    - ``corr_mean`` (legacy-compatible): mean |Pearson corr| with other
      features (used by both legacy implementations).
    """
    f = X_disc.shape[1]
    if f <= 1:
        return np.zeros(f, dtype=float)
    if kind == "corr_mean":
        Xf = X_disc.astype(float)
        corr = np.corrcoef(Xf, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)
        return np.mean(np.abs(corr), axis=1)
    if kind == "nmi_mean":
        from .information import pairwise_nmi_mean
        return pairwise_nmi_mean(X_disc, estimator=estimator)
    raise ValueError(f"Unknown vertex redundancy kind: {kind}")


def compute_vertex_degrees(
    X_disc: np.ndarray,
    y: np.ndarray,
    *,
    cfg_redundancy_kind: str = "nmi_mean",
    estimator: str = "plugin",
    evidence_map: str = "sigmoid",
    a1: float = 5.0, b1: float = -2.5,
    a2: float = 5.0, b2: float = -2.5,
    use_nonmembership: bool = True,
) -> VertexDegrees:
    f = X_disc.shape[1]
    rel = np.array([nmi(X_disc[:, j], y, estimator=estimator) for j in range(f)], dtype=float)
    red = vertex_redundancy_scores(X_disc, kind=cfg_redundancy_kind, estimator=estimator)
    if not use_nonmembership:
        red_evidence = np.zeros_like(red)
    else:
        red_evidence = red
    raw_mu = map_evidence(rel, evidence_map, a1, b1)
    raw_nu = map_evidence(red_evidence, evidence_map, a2, b2)
    mu, nu, pi = ifs_from_evidences(raw_mu, raw_nu)
    return VertexDegrees(mu=mu, nu=nu, pi=pi, relevance=rel, redundancy=red)


@dataclass
class EdgeDegrees:
    mu: Dict[Tuple[int, ...], float]
    nu: Dict[Tuple[int, ...], float]
    pi: Dict[Tuple[int, ...], float]
    group_relevance: Dict[Tuple[int, ...], float]
    group_redundancy: Dict[Tuple[int, ...], float]
    tc: Dict[Tuple[int, ...], float]


def independent_edge_degrees(
    X_disc: np.ndarray,
    y: np.ndarray,
    edges: Sequence[Tuple[int, ...]],
    *,
    estimator: str = "plugin",
    evidence_map: str = "sigmoid",
    a3: float = 5.0, b3: float = -2.5,
    a4: float = 5.0, b4: float = -2.5,
    use_nonmembership: bool = True,
    marginals: Sequence[float] | None = None,
) -> EdgeDegrees:
    """Independently estimated group degrees (the submitted construction)."""
    from .information import entropy  # local import to avoid cycle at module load

    f = X_disc.shape[1]
    if marginals is None:
        marginals = [entropy(X_disc[:, j], estimator) for j in range(f)]
    mu: Dict[Tuple[int, ...], float] = {}
    nu: Dict[Tuple[int, ...], float] = {}
    pi: Dict[Tuple[int, ...], float] = {}
    gr: Dict[Tuple[int, ...], float] = {}
    gn: Dict[Tuple[int, ...], float] = {}
    gt: Dict[Tuple[int, ...], float] = {}
    for e in edges:
        cols = X_disc[:, list(e)]
        rel = nmi_multivariate(cols, y, estimator=estimator)
        red, tc = group_redundancy(cols, estimator=estimator,
                                   marginals=[marginals[j] for j in e])
        red_e = 0.0 if not use_nonmembership else red
        raw_mu = float(map_evidence(np.array([rel]), evidence_map, a3, b3)[0])
        raw_nu = float(map_evidence(np.array([red_e]), evidence_map, a4, b4)[0])
        m, n, p = ifs_from_evidences(np.array([raw_mu]), np.array([raw_nu]))
        mu[e] = float(m[0]); nu[e] = float(n[0]); pi[e] = float(p[0])
        gr[e] = float(rel); gn[e] = float(red); gt[e] = float(tc)
    return EdgeDegrees(mu=mu, nu=nu, pi=pi, group_relevance=gr,
                       group_redundancy=gn, tc=gt)


def strict_strong_induced_edge_degrees(
    vertex_mu: np.ndarray,
    vertex_nu: np.ndarray,
    edges: Sequence[Tuple[int, ...]],
) -> EdgeDegrees:
    """Natural strong extension (ablation only):

        mu_strong(e) = min_{j in e} mu_V(j)
        nu_strong(e) = max_{j in e} nu_V(j)
        pi_strong(e) = 1 - mu - nu
    """
    mu: Dict[Tuple[int, ...], float] = {}
    nu: Dict[Tuple[int, ...], float] = {}
    pi: Dict[Tuple[int, ...], float] = {}
    for e in edges:
        m = float(min(vertex_mu[j] for j in e))
        n = float(max(vertex_nu[j] for j in e))
        # normalization can be violated only by float error; clip
        if m + n > 1.0:
            n = max(0.0, 1.0 - m)
        mu[e] = m
        nu[e] = n
        pi[e] = 1.0 - m - n
    return EdgeDegrees(mu=mu, nu=nu, pi=pi, group_relevance={}, group_redundancy={}, tc={})


def strong_relation_audit(
    vertex_mu: np.ndarray,
    vertex_nu: np.ndarray,
    edge_deg: EdgeDegrees,
    edges: Sequence[Tuple[int, ...]],
    tol: float = 1e-9,
) -> Tuple[list, Dict[str, object]]:
    """Compare independent group degrees against the natural strong extension."""
    rows = []
    n = 0
    n_ok_mu = 0
    n_ok_nu = 0
    n_ok_both = 0
    abs_mu_devs = []
    abs_nu_devs = []
    rel_mu_devs = []
    rel_nu_devs = []
    for e in edges:
        mu_s = float(min(vertex_mu[j] for j in e))
        nu_s = float(max(vertex_nu[j] for j in e))
        mu_i = edge_deg.mu[e]
        nu_i = edge_deg.nu[e]
        d_mu = abs(mu_i - mu_s)
        d_nu = abs(nu_i - nu_s)
        ok_mu = d_mu <= tol
        ok_nu = d_nu <= tol
        rows.append({
            "edge": ";".join(map(str, e)),
            "order": len(e),
            "mu_independent": mu_i,
            "nu_independent": nu_i,
            "pi_independent": edge_deg.pi[e],
            "mu_strong": mu_s,
            "nu_strong": nu_s,
            "pi_strong": 1.0 - mu_s - nu_s,
            "abs_dev_mu": d_mu,
            "abs_dev_nu": d_nu,
            "rel_dev_mu": d_mu / (abs(mu_s) + EPS),
            "rel_dev_nu": d_nu / (abs(nu_s) + EPS),
            "satisfies_mu": ok_mu,
            "satisfies_nu": ok_nu,
            "satisfies_both": ok_mu and ok_nu,
        })
        n += 1
        n_ok_mu += int(ok_mu)
        n_ok_nu += int(ok_nu)
        n_ok_both += int(ok_mu and ok_nu)
        abs_mu_devs.append(d_mu)
        abs_nu_devs.append(d_nu)
        rel_mu_devs.append(d_mu / (abs(mu_s) + EPS))
        rel_nu_devs.append(d_nu / (abs(nu_s) + EPS))

    summary = {
        "n_edges": n,
        "pct_satisfies_mu": 100.0 * n_ok_mu / n if n else float("nan"),
        "pct_satisfies_nu": 100.0 * n_ok_nu / n if n else float("nan"),
        "pct_satisfies_both": 100.0 * n_ok_both / n if n else float("nan"),
        "mean_abs_dev_mu": float(np.mean(abs_mu_devs)) if n else float("nan"),
        "mean_abs_dev_nu": float(np.mean(abs_nu_devs)) if n else float("nan"),
        "max_abs_dev_mu": float(np.max(abs_mu_devs)) if n else float("nan"),
        "max_abs_dev_nu": float(np.max(abs_nu_devs)) if n else float("nan"),
        "mean_rel_dev_mu": float(np.mean(rel_mu_devs)) if n else float("nan"),
        "mean_rel_dev_nu": float(np.mean(rel_nu_devs)) if n else float("nan"),
        "tol": tol,
    }
    return rows, summary
