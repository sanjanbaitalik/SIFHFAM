import numpy as np
import pytest

from sifhfam.discretization import Discretizer
from sifhfam.information import (
    ENTROPY_BASE,
    entropy,
    group_redundancy,
    nmi,
    pairwise_nmi_mean,
    feature_nmi_scores,
)


def test_entropy_base_documented():
    assert "natural" in ENTROPY_BASE.lower() or "nats" in ENTROPY_BASE.lower()


def test_redundancy_bounded_random_integer_sets():
    rng = np.random.default_rng(0)
    for _ in range(40):
        X = rng.integers(0, 4, size=(250, 3))
        r, tc = group_redundancy(X)
        assert np.isfinite(r) and np.isfinite(tc)
        assert -1e-12 <= r <= 1 + 1e-12
        assert tc >= 0.0


def test_redundancy_independence_near_zero():
    rng = np.random.default_rng(1)
    X = Discretizer(bins=4).fit_transform(rng.normal(size=(4000, 3)))
    r, _ = group_redundancy(X)
    assert r < 0.1


def test_redundancy_duplicates_attain_one():
    """Corrected normalization: maximally redundant groups attain exactly 1
    (D* > 0).  The historical normalization capped them at (k-1)/k and is
    retained only for delta audits."""
    rng = np.random.default_rng(2)
    base = rng.integers(0, 5, size=2000)
    X = Discretizer(bins=6).fit_transform(np.column_stack([base, base]))
    r, _ = group_redundancy(X)
    assert abs(r - 1.0) < 1e-9
    X3 = Discretizer(bins=6).fit_transform(np.column_stack([base, base, base]))
    r3, _ = group_redundancy(X3)
    assert abs(r3 - 1.0) < 1e-9
    # documented historical behavior (delta audit only)
    from sifhfam.information import legacy_group_redundancy
    r_old_pair, _ = legacy_group_redundancy(X)
    r_old_triple, _ = legacy_group_redundancy(X3)
    assert abs(r_old_pair - 0.5) < 1e-6
    assert abs(r_old_triple - 2 / 3) < 1e-6


def test_redundancy_permutation_invariance():
    rng = np.random.default_rng(3)
    X = rng.integers(0, 4, size=(300, 4))
    r1, tc1 = group_redundancy(X)
    perm = rng.permutation(4)
    r2, tc2 = group_redundancy(X[:, perm])
    assert abs(r1 - r2) < 1e-12
    assert abs(tc1 - tc2) < 1e-12


def test_redundancy_zero_entropy_safe():
    Xc = np.zeros((50, 3), dtype=int)
    r, tc = group_redundancy(Xc)
    assert r == 0.0 and tc == 0.0


def test_miller_madow_in_range():
    rng = np.random.default_rng(4)
    X = rng.integers(0, 3, size=(80, 3))
    r, _ = group_redundancy(X, estimator="miller_madow")
    assert 0.0 <= r <= 1.0
    # documented estimators exist
    assert entropy(X[:, 0], estimator="plugin") >= 0
    assert entropy(X[:, 0], estimator="miller_madow") >= 0


def test_pairwise_and_feature_nmi_fast_paths_match_generic():
    rng = np.random.default_rng(5)
    X = Discretizer(bins=5).fit_transform(rng.normal(size=(400, 8)))
    y = (rng.random(400) > 0.5).astype(int)
    fast_pair = pairwise_nmi_mean(X)
    # generic reference for feature 0
    ref0 = np.mean([nmi(X[:, 0], X[:, j]) for j in range(1, 8)])
    assert abs(fast_pair[0] - ref0) < 1e-9
    fast_feat = feature_nmi_scores(X, y)
    ref_f0 = nmi(X[:, 0], y)
    assert abs(fast_feat[0] - ref_f0) < 1e-9
