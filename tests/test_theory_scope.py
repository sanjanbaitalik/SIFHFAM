"""Theory-scope tests: non-negative weights, monotone/submodular, greedy ratio."""
import numpy as np
import pytest

from sifhfam.config import RevisionConfig
from sifhfam.hypergraph import generate_hyperedges, theoretical_counts
from sifhfam.ifs import independent_edge_degrees, compute_vertex_degrees
from sifhfam.objective import Objective, exhaustive_greedy_ratio, greedy_path, is_monotone, is_submodular
from sifhfam.discretization import Discretizer
from sifhfam.information import nmi
from sifhfam.selector import fit_canonical


def _build_obj(seed=0, f=8, g_type="binary"):
    rng = np.random.default_rng(seed)
    mu = rng.random(f)
    edges = [tuple(sorted(rng.choice(f, size=3, replace=False).tolist()))
             for _ in range(12)]
    w = {e: float(rng.random()) for e in edges}  # >= 0
    return Objective(mu, w, g_type=g_type)


def test_weights_nonnegative_in_canonical_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 25))
    y = (X[:, :3].sum(axis=1) > 0).astype(int)
    cfg = RevisionConfig(random_state=0, f=15, max_edges=200, min_f=10, max_f=15)
    res = fit_canonical(X, y, cfg, max_path_budget=10)
    assert res.error is None
    assert len(res.edge_weights) > 0
    assert all(w >= 0 for w in res.edge_weights.values())


def test_theoretical_edge_counts():
    counts = theoretical_counts(10, 3)
    assert counts[2] == 45
    assert counts[3] == 120
    coll = generate_hyperedges(10, 3, 10000, seed=0)
    assert len(coll.edges) == 165  # exhaustive: 45 + 120
    assert coll.sampled is False


@pytest.mark.parametrize("g_type", ["binary", "soft", "fraction"])
def test_tiny_monotone_submodular_and_ratio(g_type):
    obj = _build_obj(g_type=g_type)
    assert is_monotone(obj, list(range(8)), max_sets=300, seed=1)
    assert is_submodular(obj, list(range(8)), max_checks=400, seed=1)
    path = greedy_path(obj, budget=4, stop_nonpositive=False)  # budget < f: non-trivial
    greedy_val = path.prefix_values[-1]
    _, opt_val = exhaustive_greedy_ratio(obj, budget=4)
    assert greedy_val <= opt_val + 1e-9
    ratio = greedy_val / opt_val if opt_val > 0 else float("nan")
    # classical greedy on a monotone submodular non-negative objective with
    # cardinality m: ratio >= 1 - 1/e is CLASSICAL (not claimed as new here);
    # we only record the observed ratio.
    assert np.isnan(ratio) or (0.0 <= ratio <= 1.0 + 1e-9)


def test_strict_strong_variant_edges_match_vertex_relation():
    from sifhfam.ifs import strict_strong_induced_edge_degrees, strong_relation_audit
    rng = np.random.default_rng(3)
    mu = rng.random(12)
    nu = rng.random(12)
    scale = np.maximum(1.0, mu + nu)
    mu, nu = mu / scale, nu / scale
    edges = [tuple(sorted(rng.choice(12, size=3, replace=False).tolist()))
             for _ in range(25)]
    ed = strict_strong_induced_edge_degrees(mu, nu, edges)
    _, summ = strong_relation_audit(mu, nu, ed, edges, tol=1e-12)
    assert summ["pct_satisfies_both"] == 100.0
