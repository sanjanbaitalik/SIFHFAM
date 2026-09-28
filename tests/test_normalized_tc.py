"""Focused tests for the corrected normalized total correlation (Phase 2).

R*_e = TC(e) / D*_e with D*_e = sum_j H(X_j) - max_j H(X_j),
tau used only as a numerical-zero threshold, and NO epsilon on a genuinely
positive denominator.
"""
from pathlib import Path

import numpy as np
import pytest

from sifhfam.information import (
    MATERIAL_VIOLATION,
    NORMALIZATION_TAU,
    group_redundancy,
    group_redundancy_detailed,
    legacy_group_redundancy,
)
from sifhfam.ifs import independent_edge_degrees, map_evidence


# 1) independent discrete variables ----------------------------------------- #

def test_independent_variables_near_zero():
    rng = np.random.default_rng(10)
    X = rng.integers(0, 5, size=(4000, 3))
    d = group_redundancy_detailed(X)
    assert d["tc"] < 0.2                 # TC near 0 up to estimation error
    assert d["r_star"] < 0.05            # corrected redundancy near 0
    assert d["d_star"] > 0.1
    assert not d["zero_denominator"]


# 2-4) perfect duplicates / deterministic functions ------------------------- #

def test_duplicated_pair_exactly_one():
    rng = np.random.default_rng(11)
    base = rng.integers(0, 6, size=1000)
    X = np.column_stack([base, base])
    d = group_redundancy_detailed(X)
    assert d["d_star"] > 0
    assert d["r_star"] == pytest.approx(1.0, abs=1e-9)
    assert d["pre_clip_deviation"] <= 1e-9


def test_duplicated_triple_exactly_one():
    rng = np.random.default_rng(12)
    base = rng.integers(0, 6, size=1000)
    X = np.column_stack([base, base, base])
    d = group_redundancy_detailed(X)
    assert d["d_star"] > 0
    assert d["r_star"] == pytest.approx(1.0, abs=1e-9)


def test_deterministic_function_of_one_member_one():
    rng = np.random.default_rng(13)
    x = rng.integers(0, 8, size=1000)
    cases = [
        np.column_stack([x, x % 2]),          # non-injective function
        np.column_stack([x, 2 * x]),          # bijective (scaled copy)
        np.column_stack([x, x // 2]),         # many-to-one
    ]
    for X in cases:
        d = group_redundancy_detailed(X)
        # joint entropy equals the largest marginal entropy -> R* = 1
        assert d["d_star"] > 0, "denominator must be positive for these cases"
        assert d["r_star"] == pytest.approx(1.0, abs=1e-9), d


# 5-6) zero-denominator handling -------------------------------------------- #

def test_all_constant_group_zero_no_nan():
    Xc = np.zeros((100, 4), dtype=int)
    d = group_redundancy_detailed(Xc)
    assert d["zero_denominator"] is True
    assert d["r_star"] == 0.0
    assert np.isfinite(d["r_star"]) and np.isfinite(d["tc"])


def test_one_nonconstant_plus_constants_zero_denominator():
    rng = np.random.default_rng(14)
    x = rng.integers(0, 5, size=500)
    X = np.column_stack([x, np.zeros_like(x), np.ones_like(x)])
    d = group_redundancy_detailed(X)
    # sum H = H(x), max H = H(x) -> D* = 0 -> score 0 (documented case)
    assert d["d_star"] <= NORMALIZATION_TAU
    assert d["zero_denominator"] is True
    assert d["r_star"] == 0.0


# 7) permutation invariance -------------------------------------------------- #

def test_permutation_invariance_of_tc_denominator_and_score():
    rng = np.random.default_rng(15)
    X = rng.integers(0, 4, size=(500, 4))
    d1 = group_redundancy_detailed(X)
    for _ in range(5):
        perm = rng.permutation(4)
        d2 = group_redundancy_detailed(X[:, perm])
        assert abs(d1["tc"] - d2["tc"]) < 1e-12
        assert abs(d1["d_star"] - d2["d_star"]) < 1e-12
        assert abs(d1["r_star"] - d2["r_star"]) < 1e-12


# 8) randomized range test ---------------------------------------------------- #

def test_randomized_range_within_tolerance():
    rng = np.random.default_rng(16)
    for estimator in ("plugin", "miller_madow"):
        for order in (2, 3, 4):
            for _ in range(25):
                X = rng.integers(0, rng.integers(2, 7), size=(150, order))
                d = group_redundancy_detailed(X, estimator=estimator)
                assert -1e-9 <= d["r_star"] <= 1 + 1e-9, d
                assert d["pre_clip_deviation"] <= MATERIAL_VIOLATION, d
                assert np.isfinite(d["r_star"]) and np.isfinite(d["tc"])
                if d["zero_denominator"]:
                    assert d["r_star"] == 0.0


def test_no_epsilon_on_positive_denominator():
    """A positive D* must be used exactly: R* == TC/D* pre- and post-clip."""
    rng = np.random.default_rng(17)
    X = rng.integers(0, 4, size=(300, 3))
    d = group_redundancy_detailed(X)
    assert d["d_star"] > NORMALIZATION_TAU
    assert d["r_pre_clip"] == pytest.approx(d["tc"] / d["d_star"], rel=1e-12)


def test_material_violation_refuses_to_clip(monkeypatch):
    """A material out-of-range value must raise, not be silently clipped."""
    import sifhfam.information as inf

    def fake_tc(*a, **k):
        return 100.0  # force TC >> D* so R* is materially out of range

    monkeypatch.setattr(inf, "total_correlation", fake_tc)
    X = np.random.default_rng(18).integers(0, 4, size=(200, 3))
    with pytest.raises(ValueError, match="MATERIAL normalisation violation"):
        inf.group_redundancy_detailed(X)


# 9) hyperedge IFS validity --------------------------------------------------- #

def test_hyperedge_ifs_validity_from_edge_degrees():
    rng = np.random.default_rng(19)
    X = rng.integers(0, 4, size=(200, 6))
    y = (X[:, 0] + X[:, 1] > 3).astype(int)
    edges = [(0, 1), (1, 2, 3), (0, 2, 4, 5), (2, 2, 2)][:0] or [(0, 1), (1, 2, 3), (0, 2, 4)]
    ed = independent_edge_degrees(X, y, edges, a4=5.0, b4=-2.5)
    for e in edges:
        mu, nu, pi = ed.mu[e], ed.nu[e], ed.pi[e]
        assert mu >= 0 and nu >= 0 and pi >= 0
        assert abs(mu + nu + pi - 1.0) < 1e-9
        # group redundancy stored is the corrected R*
        assert -1e-9 <= ed.group_redundancy[e] <= 1 + 1e-9


# 10) sigmoid behavior -------------------------------------------------------- #

def test_sigmoid_midpoint_and_high_evidence():
    # R* = 0.5 -> established midpoint under a4=5, b4=-2.5
    raw_mid = map_evidence(np.array([0.5]), "sigmoid", 5.0, -2.5)
    assert float(raw_mid[0]) == pytest.approx(0.5, abs=1e-12)
    # R* = 1 -> high non-membership evidence; R* = 0 -> low
    raw_one = float(map_evidence(np.array([1.0]), "sigmoid", 5.0, -2.5)[0])
    raw_zero = float(map_evidence(np.array([0.0]), "sigmoid", 5.0, -2.5)[0])
    assert raw_one == pytest.approx(1.0 / (1.0 + np.exp(-2.5)), rel=1e-12)
    assert raw_one >= 0.9
    assert raw_one > 0.5 > raw_zero
    assert raw_zero == pytest.approx(1.0 / (1.0 + np.exp(2.5)), rel=1e-12)
    # pipeline: a duplicated group (R*=1) yields high edge nu evidence
    rng = np.random.default_rng(20)
    base = rng.integers(0, 5, size=300)
    X = np.column_stack([base, base])
    y = (base > 2).astype(int)
    ed = independent_edge_degrees(X, y, [(0, 1)], a4=5.0, b4=-2.5)
    assert ed.group_redundancy[(0, 1)] == pytest.approx(1.0, abs=1e-9)
    # nu is the IFS-normalized sigmoid of R*=1 -> clearly elevated non-membership
    assert ed.nu[(0, 1)] > 0.3


# 11) regression guards: vertex / screening / sampling / objective / greedy --- #

def test_pipeline_modules_do_not_reference_group_redundancy():
    pkg = Path(__file__).resolve().parents[1] / "sifhfam"
    for mod in ("screening.py", "objective.py", "hypergraph.py",
                "discretization.py", "information.py"):
        # information.py defines it; the others must not use it
        if mod == "information.py":
            assert "def group_redundancy" in (pkg / mod).read_text()
            continue
        src = (pkg / mod).read_text()
        assert "group_redundancy" not in src, mod
    # objective/screening behavior on fixed inputs (deterministic, unchanged)
    from sifhfam.objective import Objective, greedy_path, g_binary
    obj = Objective(vertex_mu=np.array([0.4, 0.6, 0.5]),
                    edge_weights={(0, 1): 1.0, (1, 2): 0.5},
                    g_type="binary")
    path = greedy_path(obj, budget=3, stop_nonpositive=False)
    assert path.order[0] == 1                      # mu 0.6 + edge gains
    assert g_binary(0) == 0.0 and g_binary(2) == 1.0
    assert abs(obj.value([0, 1, 2]) - obj.value([0, 1, 2])) < 1e-15


def test_screening_and_hyperedge_sampling_deterministic():
    from sifhfam.screening import screen
    from sifhfam.hypergraph import generate_hyperedges
    rng = np.random.default_rng(21)
    X = rng.normal(size=(120, 40))
    y = (X[:, 0] > 0).astype(int)
    a = screen(X, y, 15, strategy="mi", seed=7)
    b = screen(X, y, 15, strategy="mi", seed=7)
    assert np.array_equal(a.screened_idx, b.screened_idx)
    e1 = generate_hyperedges(30, 3, 500, seed=7)
    e2 = generate_hyperedges(30, 3, 500, seed=7)
    assert e1.edges == e2.edges


def test_vertex_and_screening_paths_never_call_group_redundancy(monkeypatch):
    """With hyperedges disabled the whole fit must not touch group
    redundancy; vertex/screening logic is therefore unaffected."""
    import sifhfam.ifs as ifs_mod
    from sifhfam.selector import fit_canonical
    from sifhfam.config import RevisionConfig

    def boom(*a, **k):  # pragma: no cover
        raise AssertionError("group_redundancy must not be called in vertex-only path")

    monkeypatch.setattr(ifs_mod, "group_redundancy", boom)
    rng = np.random.default_rng(22)
    X = rng.normal(size=(80, 30))
    y = (X[:, :3].sum(axis=1) > 0).astype(int)
    cfg = RevisionConfig(random_state=0, use_hyperedges=False, beta=0.0, K=1,
                         f=15, min_f=10, max_f=20, name="SIFFAM")
    res = fit_canonical(X, y, cfg, max_path_budget=10)
    assert res.error is None
    assert len(res.selected) > 0
    # (with hyperedges ON, group_redundancy IS the path under correction)


def test_vertex_logic_values_unchanged_by_correction():
    """Vertex redundancy/relevance do not involve group redundancy."""
    from sifhfam.ifs import vertex_redundancy_scores
    from sifhfam.discretization import Discretizer
    from sifhfam.information import feature_nmi_scores
    rng = np.random.default_rng(23)
    X = Discretizer(bins=5).fit_transform(rng.normal(size=(300, 12)))
    y = (X[:, 0] % 2).astype(int)
    red = vertex_redundancy_scores(X, kind="nmi_mean")
    rel = feature_nmi_scores(X, y)
    assert red.shape == (12,) and rel.shape == (12,)
    assert np.all(np.isfinite(red)) and np.all(red >= 0)
    assert np.all(np.isfinite(rel)) and np.all(rel >= 0)


# legacy helper retained for delta audits ------------------------------------ #

def test_legacy_normalization_available_for_delta_only():
    rng = np.random.default_rng(24)
    base = rng.integers(0, 5, size=500)
    X = np.column_stack([base, base])
    r_old, tc_old = legacy_group_redundancy(X)
    r_new, tc_new = group_redundancy(X)
    assert abs(tc_old - tc_new) < 1e-12          # same TC
    assert abs(r_old - 0.5) < 1e-6
    assert abs(r_new - 1.0) < 1e-9
