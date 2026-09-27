import numpy as np
import pytest

from sifhfam.ifs import (
    ifs_from_evidences,
    strict_strong_induced_edge_degrees,
    strong_relation_audit,
    validate_ifs,
)


def test_ifs_axioms_random():
    rng = np.random.default_rng(0)
    raw_mu = rng.random(100) * 5
    raw_nu = rng.random(100) * 5
    mu, nu, pi = ifs_from_evidences(raw_mu, raw_nu)
    checks = validate_ifs(mu, nu, pi)
    assert all(checks.values()), checks
    assert np.all(mu + nu <= 1.0 + 1e-12)


def test_ifs_nonnegative_evidence_clipped():
    mu, nu, pi = ifs_from_evidences(np.array([-1.0, 0.0]), np.array([-2.0, 3.0]))
    assert np.all(mu >= 0) and np.all(nu >= 0) and np.all(pi > 0)
    assert np.allclose(mu + nu + pi, 1.0)


def test_strict_strong_induced_satisfies_relation():
    rng = np.random.default_rng(1)
    mu = rng.random(30)
    nu = rng.random(30)
    scale = np.maximum(1.0, mu + nu)
    mu, nu = mu / scale, nu / scale
    edges = [tuple(sorted(rng.choice(30, size=int(rng.integers(2, 6)),
                                     replace=False).tolist()))
             for _ in range(80)]
    ed = strict_strong_induced_edge_degrees(mu, nu, edges)
    rows, summary = strong_relation_audit(mu, nu, ed, edges, tol=1e-12)
    assert summary["pct_satisfies_both"] == 100.0
    assert summary["n_edges"] == 80


def test_independent_group_degrees_are_audited_not_asserted():
    """Independent degrees are expected NOT to match the strong relation in general;
    the audit must quantify this rather than assume it."""
    rng = np.random.default_rng(2)
    mu_v = rng.random(20)
    nu_v = rng.random(20)
    scale = np.maximum(1.0, mu_v + nu_v)
    mu_v, nu_v = mu_v / scale, nu_v / scale
    edges = [tuple(sorted(rng.choice(20, size=3, replace=False).tolist()))
             for _ in range(40)]
    from sifhfam.ifs import EdgeDegrees
    indep_mu = {e: float(rng.random() * 0.5) for e in edges}
    indep_nu = {e: float(rng.random() * 0.3) for e in edges}
    ed = EdgeDegrees(mu=indep_mu, nu=indep_nu,
                     pi={e: 1 - indep_mu[e] - indep_nu[e] for e in edges},
                     group_relevance={}, group_redundancy={}, tc={})
    rows, summary = strong_relation_audit(mu_v, nu_v, ed, edges, tol=1e-9)
    # audit produced numeric deviations for every edge
    assert summary["n_edges"] == len(edges)
    assert np.isfinite(summary["mean_abs_dev_mu"])
