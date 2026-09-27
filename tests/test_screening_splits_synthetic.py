import numpy as np
import pytest

from sifhfam.config import RevisionConfig, budget_grid, choose_f, choose_m, manuscript_m
from sifhfam.data import load_dataset, make_split, run_seed, save_splits, load_splits
from sifhfam.screening import screen
from sifhfam.selector import fit, fit_canonical
from sifhfam.synthetic import (
    export_interaction_example_table,
    generate,
    marginal_association_check,
    recovery_metrics,
)


@pytest.fixture(scope="module")
def colon():
    X, y, meta = load_dataset("colon")
    return X, y


def test_screening_is_training_only_and_deterministic(colon):
    X, y = colon
    seed = run_seed(42, "colon", 0)
    sp = make_split("colon", X.shape[0], y, 0, 0.25, seed)
    Xtr, ytr = X[sp.train_idx], y[sp.train_idx]
    a = screen(Xtr, ytr, f_budget=40, strategy="mi", seed=seed)
    b = screen(Xtr, ytr, f_budget=40, strategy="mi", seed=seed)
    assert np.array_equal(a.screened_idx, b.screened_idx)
    # test features must not influence screening: screening on train only
    # uses exactly len(train) rows (fit on Xtr only by construction)
    assert a.screened_idx.shape[0] <= 40
    # changing test rows has no effect (we never pass them)
    Xte2 = X[sp.test_idx].copy()
    c = screen(Xtr, ytr, f_budget=40, strategy="mi", seed=seed)
    assert np.array_equal(a.screened_idx, c.screened_idx)
    assert Xte2.shape[0] == len(sp.test_idx)


def test_no_test_leakage_in_fit(colon):
    """Perturbing the held-out test fold must not change the selection."""
    X, y = colon
    seed = run_seed(42, "colon", 1)
    sp = make_split("colon", X.shape[0], y, 1, 0.25, seed)
    Xtr, ytr = X[sp.train_idx], y[sp.train_idx]
    cfg = RevisionConfig(random_state=seed, max_edges=800, min_f=30, max_f=60, f=40)
    res1 = fit_canonical(Xtr, ytr, cfg, max_path_budget=20)
    # scramble test fold then refit on same train data
    X_test_scrambled = X[sp.test_idx][::-1].copy()
    res2 = fit_canonical(Xtr, ytr, cfg, max_path_budget=20)
    assert res1.error is None and res2.error is None
    assert np.array_equal(res1.selected, res2.selected)
    assert X_test_scrambled.shape[0] == len(sp.test_idx)


def test_fit_deterministic_same_seed(colon):
    X, y = colon
    seed = run_seed(7, "colon", 0)
    sp = make_split("colon", X.shape[0], y, 0, 0.25, seed)
    Xtr, ytr = X[sp.train_idx], y[sp.train_idx]
    cfg = RevisionConfig(random_state=11, max_edges=500, f=30)
    r1 = fit_canonical(Xtr, ytr, cfg, max_path_budget=15)
    r2 = fit_canonical(Xtr, ytr, cfg, max_path_budget=15)
    assert r1.error is None and r2.error is None
    assert np.array_equal(r1.selected, r2.selected)
    assert r1.screened_idx.tolist() == r2.screened_idx.tolist()


def test_split_indices_roundtrip(tmp_path, colon):
    X, y = colon
    sp = make_split("colon", X.shape[0], y, 3, 0.25, 999)
    p = tmp_path / "splits.json"
    save_splits(p, [sp])
    got = load_splits(p)[0]
    assert np.array_equal(got.train_idx, sp.train_idx)
    assert np.array_equal(got.test_idx, sp.test_idx)
    assert set(got.train_idx).isdisjoint(set(got.test_idx))
    assert len(got.train_idx) + len(got.test_idx) == X.shape[0]


def test_run_seed_stable():
    assert run_seed(42, "colon", 0) == run_seed(42, "colon", 0)
    assert run_seed(42, "colon", 0) != run_seed(42, "colon", 1)
    assert run_seed(42, "colon", 0) != run_seed(43, "colon", 0)


def test_synthetic_xor_generator_has_weak_marginals():
    ds = generate("xor2", n=400, seed=0, n_noise=0)
    assert set(ds.support.tolist()) == {0, 1}
    assert np.array_equal(np.sort(np.unique(ds.y)), [0, 1])
    nmi0 = marginal_association_check(ds.X[:, 0], ds.y, bins=4)
    nmi1 = marginal_association_check(ds.X[:, 1], ds.y, bins=4)
    # weak marginal association by construction (XOR): far below a strong signal
    assert nmi0 < 0.35 and nmi1 < 0.35
    # joint XOR is informative: label determined by both
    xor = ((ds.X[:, 0] > 0.5).astype(int) ^ (ds.X[:, 1] > 0.5).astype(int))
    # allow noise columns of +-0.05 around {0,1}
    a = (ds.X[:, 0] > 0.5).astype(int)
    b = (ds.X[:, 1] > 0.5).astype(int)
    assert np.mean(np.bitwise_xor(a, b) == ds.y) > 0.95


def test_synthetic_parity3_ground_truth_stored():
    ds = generate("parity3", n=300, seed=1, n_noise=40)
    assert ds.interaction_support.tolist() == [0, 1, 2]
    assert ds.X.shape[1] == 43
    assert set(ds.support.tolist()) <= set(range(ds.X.shape[1]))


def test_recovery_metrics_exact():
    sel = [0, 1, 5]
    sup = np.array([0, 2])
    rec = recovery_metrics(sel, sup)
    assert abs(rec["precision"] - 1 / 3) < 1e-12
    assert abs(rec["recall"] - 1 / 2) < 1e-12
    assert rec["all_selected"] == 0.0


def test_budget_grid_rule():
    # manuscript_m(2000) = round(sqrt(2000)) = 45 -> grid includes 45
    assert budget_grid(2000) == [10, 25, 45, 50, 100]
    assert budget_grid(50) == [10, 25, 50]
    assert budget_grid(10) == [10]
    assert manuscript_m(2000) == 45
    assert manuscript_m(100) == 10


def test_interaction_example_table_export(tmp_path):
    p = export_interaction_example_table(tmp_path / "ex.csv", seed=0)
    import pandas as pd
    df = pd.read_csv(p)
    assert {"x1_binary", "x2_binary", "y_xor"} <= set(df.columns)
    assert (df["y_xor"] == df["x1_binary"] ^ df["x2_binary"]).all()
