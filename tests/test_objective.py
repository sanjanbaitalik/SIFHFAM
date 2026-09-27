import numpy as np
import pytest

from sifhfam.objective import (
    Objective,
    exhaustive_greedy_ratio,
    g_binary,
    g_fraction,
    g_soft,
    greedy_path,
    is_monotone,
    is_submodular,
    make_g,
    select_by_efficiency_ratio,
)


def _random_objective(rng, f=7, n_edges=10, g_type="binary"):
    mu = rng.random(f)
    edges = [tuple(sorted(rng.choice(f, size=int(rng.integers(2, 4)),
                                     replace=False).tolist()))
             for _ in range(n_edges)]
    w = {e: float(rng.random()) for e in edges}  # non-negative
    return Objective(mu, w, alpha=1.0, beta=1.0, g_type=g_type)


def test_binary_coverage_behavior():
    assert g_binary(0) == 0.0
    assert g_binary(1) == 1.0
    assert g_binary(5) == 1.0


def test_soft_coverage_diminishing_returns():
    vals = [g_soft(k, rho=0.5) for k in range(0, 6)]
    diffs = np.diff(vals)
    assert np.all(diffs >= -1e-12)          # monotone
    assert np.all(np.diff(diffs) <= 1e-12)  # diminishing (concave)
    assert g_soft(0) == 0.0


def test_fraction_coverage_diminishing():
    vals = [g_fraction(k, 4) for k in range(0, 6)]
    diffs = np.diff(vals)
    assert np.all(diffs >= -1e-12)
    assert np.all(np.diff(diffs) <= 1e-12)
    assert g_fraction(4, 4) == 1.0
    assert g_fraction(10, 4) == 1.0


@pytest.mark.parametrize("g_type", ["binary", "soft", "fraction"])
def test_monotonicity_and_submodularity(g_type):
    rng = np.random.default_rng(0)
    for trial in range(5):
        obj = _random_objective(rng, g_type=g_type)
        assert is_monotone(obj, list(range(7)), max_sets=250, seed=trial)
        assert is_submodular(obj, list(range(7)), max_checks=350, seed=trial)


def test_nonnegative_weights_required_and_greedy_prefixes():
    rng = np.random.default_rng(1)
    obj = _random_objective(rng)
    assert all(w >= 0 for w in obj.edge_weights.values())
    path = greedy_path(obj, budget=7, stop_nonpositive=False)
    assert len(path.order) <= 7
    assert len(set(path.order)) == len(path.order)
    # prefix values are non-decreasing (non-negative marginal gains)
    diffs = np.diff(path.prefix_values)
    assert np.all(diffs >= -1e-12)


def test_greedy_vs_exhaustive_ratio_recorded_not_theorem():
    rng = np.random.default_rng(2)
    obj = _random_objective(rng, f=6, n_edges=8)
    path = greedy_path(obj, budget=6, stop_nonpositive=False)
    greedy_val = path.prefix_values[-1] if path.prefix_values else 0.0
    _, opt_val = exhaustive_greedy_ratio(obj, budget=6)
    assert opt_val + 1e-9 >= greedy_val
    ratio = greedy_val / opt_val if opt_val > 0 else float("nan")
    if np.isfinite(ratio):
        assert 0.0 <= ratio <= 1.0 + 1e-9


def test_efficiency_ratio_selection():
    rng = np.random.default_rng(3)
    obj = _random_objective(rng, f=10)
    path = greedy_path(obj, budget=10, stop_nonpositive=False)
    sel = select_by_efficiency_ratio(path, k_min=3)
    assert 1 <= len(sel) <= len(path.order)
    assert sel == path.order[: len(sel)]


def test_make_g_unknown_raises():
    with pytest.raises(ValueError):
        make_g("synergy_full_group")
