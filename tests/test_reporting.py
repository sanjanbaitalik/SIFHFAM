import numpy as np
import pandas as pd
import pytest

from sifhfam.reporting import (
    assert_reporting_ok,
    check_accuracy_range,
    check_cardinality_integer,
    check_mean_sd_rounding,
    check_no_mixed_units,
    check_percent_units,
    check_reduction_arithmetic,
    check_summary_recomputes,
    format_mean_sd,
    run_validations,
)


def test_accuracy_range():
    ok, _ = check_accuracy_range(pd.DataFrame({"accuracy": [0.1, 0.9]}))
    assert ok
    ok, msg = check_accuracy_range(pd.DataFrame({"accuracy": [1.5]}))
    assert not ok


def test_percent_units_both_scaled():
    ok, _ = check_percent_units(mean=95.45, sd=5.52, as_percent=True,
                                raw_mean=0.9545, raw_sd=0.0552)
    assert ok
    ok, msg = check_percent_units(mean=95.45, sd=0.0552, as_percent=True,
                                  raw_mean=0.9545, raw_sd=0.0552)
    assert not ok


def test_mixed_units_detector_flags_classic_error():
    ok, _ = check_no_mixed_units(95.45, 0.0552)
    assert not ok          # percent mean with fraction sd
    ok, _ = check_no_mixed_units(95.45, 5.52)
    assert ok


def test_rounding_rule_when_mean_is_100():
    ok, _ = check_mean_sd_rounding(mean_pct=100.0, sd_pct=0.004, decimals=2)
    assert not ok          # sd rounds to 0.00 -> hidden variance
    ok, _ = check_mean_sd_rounding(mean_pct=99.99, sd_pct=0.004, decimals=2)
    assert ok


def test_cardinality_integer_and_reduction():
    df = pd.DataFrame({"n_selected": [10, 10], "d": [100, 100],
                       "feature_reduction": [0.9, 0.9]})
    ok, _ = check_cardinality_integer(df)
    assert ok
    ok, _ = check_reduction_arithmetic(df)
    assert ok
    df2 = df.copy()
    df2.loc[0, "feature_reduction"] = 0.8
    ok, _ = check_reduction_arithmetic(df2)
    assert not ok
    df3 = pd.DataFrame({"n_selected": [10.5]})
    ok, _ = check_cardinality_integer(df3)
    assert not ok


def test_summary_recomputes_from_raw():
    raw = pd.DataFrame({
        "dataset": ["a", "a", "b", "b"],
        "method": ["m"] * 4,
        "accuracy": [0.5, 0.7, 0.6, 0.8],
    })
    summary = raw.groupby(["dataset", "method"])["accuracy"].agg(["mean", "std"]).reset_index()
    ok, _ = check_summary_recomputes(raw, summary, ("dataset", "method"), "accuracy")
    assert ok
    bad = summary.copy()
    bad.loc[0, "mean"] = 0.99
    ok, _ = check_summary_recomputes(raw, bad, ("dataset", "method"), "accuracy")
    assert not ok


def test_run_validations_and_assert_blocks_on_failure():
    raw = pd.DataFrame({
        "dataset": ["a", "a"], "method": ["m", "m"],
        "accuracy": [0.5, 0.7], "n_selected": [10, 10],
        "d": [100, 100], "feature_reduction": [0.9, 0.9],
    })
    summary = raw.groupby(["dataset", "method"])["accuracy"].agg(["mean", "std"]).reset_index()
    rep, checks = run_validations(raw, summary, ("dataset", "method"), "accuracy")
    assert rep["all_passed"]
    assert_reporting_ok(rep)

    bad_raw = raw.copy()
    bad_raw.loc[0, "accuracy"] = 95.4
    rep2, _ = run_validations(bad_raw, summary, ("dataset", "method"), "accuracy")
    assert not rep2["all_passed"]
    with pytest.raises(AssertionError):
        assert_reporting_ok(rep2)


def test_format_mean_sd_units_matched():
    s = format_mean_sd(0.9545, 0.0552, as_percent=True, decimals=2)
    assert "%" in s
    mean_part = float(s.split(" ")[0])
    sd_part = float(s.split(" ")[2])
    # both are on the same (percent) scale
    assert mean_part > 1 and sd_part > 1
