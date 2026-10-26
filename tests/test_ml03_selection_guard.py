"""Tests for candidate selection protocol, high-load constraints, and fallback guards using production helpers."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from household_power.experiments import (
    align_baseline_test_predictions,
    compute_high_load_threshold,
    confirm_validation_candidate,
    evaluate_candidate_cv,
)


def test_compute_high_load_threshold_production_helper():
    """Verify compute_high_load_threshold calculates exact percentile and rejects invalid inputs."""
    y = np.linspace(0.0, 10.0, 101)
    p90 = compute_high_load_threshold(y, 90.0)
    assert np.isclose(p90, 9.0)

    # Rejects empty or all-nonfinite
    with pytest.raises(ValueError, match="empty or non-finite"):
        compute_high_load_threshold(np.array([]))
    with pytest.raises(ValueError, match="empty or non-finite"):
        compute_high_load_threshold(np.array([np.nan, np.inf]))


def test_evaluate_candidate_cv_1_percent_overall_guard():
    """Verify 1% overall MAE constraint boundary in evaluate_candidate_cv."""
    base_folds = [
        {"overall_mae": 0.400, "high_load_mae": 1.000, "normal_load_mae": 0.300},
        {"overall_mae": 0.400, "high_load_mae": 1.000, "normal_load_mae": 0.300},
        {"overall_mae": 0.400, "high_load_mae": 1.000, "normal_load_mae": 0.300},
    ]

    # Exactly 1.01 ratio (0.404 / 0.400 = 1.01) -> passes
    cand_pass_folds = [
        {"overall_mae": 0.404, "high_load_mae": 0.950, "normal_load_mae": 0.300},
        {"overall_mae": 0.404, "high_load_mae": 0.950, "normal_load_mae": 0.300},
        {"overall_mae": 0.404, "high_load_mae": 0.950, "normal_load_mae": 0.300},
    ]
    res_pass = evaluate_candidate_cv(cand_pass_folds, base_folds, "enhanced_features")
    assert res_pass["overall_constraint_pass"]
    assert res_pass["eligible_for_promotion"]

    # 1.015 ratio (0.406 / 0.400 = 1.015 > 1.01) -> rejected
    cand_fail_folds = [
        {"overall_mae": 0.406, "high_load_mae": 0.850, "normal_load_mae": 0.300},
        {"overall_mae": 0.406, "high_load_mae": 0.850, "normal_load_mae": 0.300},
        {"overall_mae": 0.406, "high_load_mae": 0.850, "normal_load_mae": 0.300},
    ]
    res_fail = evaluate_candidate_cv(cand_fail_folds, base_folds, "enhanced_weighted")
    assert not res_fail["overall_constraint_pass"]
    assert not res_fail["eligible_for_promotion"]


def test_evaluate_candidate_cv_folds_non_regression_boundary():
    """Candidate must match or improve high load in at least 2 of 3 folds."""
    base_folds = [
        {"overall_mae": 0.400, "high_load_mae": 1.000, "normal_load_mae": 0.300},
        {"overall_mae": 0.400, "high_load_mae": 1.100, "normal_load_mae": 0.300},
        {"overall_mae": 0.400, "high_load_mae": 1.050, "normal_load_mae": 0.300},
    ]

    # Only 1 fold improved (fold 1: 0.900 <= 1.000; fold 2: 1.150 > 1.100; fold 3: 1.100 > 1.050)
    cand_1fold = [
        {"overall_mae": 0.395, "high_load_mae": 0.900, "normal_load_mae": 0.300},
        {"overall_mae": 0.395, "high_load_mae": 1.150, "normal_load_mae": 0.300},
        {"overall_mae": 0.395, "high_load_mae": 1.100, "normal_load_mae": 0.300},
    ]
    res_1fold = evaluate_candidate_cv(cand_1fold, base_folds, "cand_1fold")
    assert res_1fold["high_load_non_regression_folds"] == 1
    assert not res_1fold["high_load_folds_pass"]
    assert not res_1fold["eligible_for_promotion"]

    # 2 folds improved (fold 1 and fold 2)
    cand_2folds = [
        {"overall_mae": 0.395, "high_load_mae": 0.900, "normal_load_mae": 0.300},
        {"overall_mae": 0.395, "high_load_mae": 1.080, "normal_load_mae": 0.300},
        {"overall_mae": 0.395, "high_load_mae": 1.100, "normal_load_mae": 0.300},
    ]
    res_2folds = evaluate_candidate_cv(cand_2folds, base_folds, "cand_2folds")
    assert res_2folds["high_load_non_regression_folds"] == 2
    assert res_2folds["high_load_folds_pass"]
    assert res_2folds["eligible_for_promotion"]


def test_confirm_validation_candidate_fallback_and_promotion():
    """Verify confirm_validation_candidate strict fallback when high load regresses."""
    base_val = {"overall_mae": 0.3706, "high_load_mae": 0.8359}

    # Case A: High load regresses (0.8409 > 0.8359) even though overall improves (0.3696 < 0.3706)
    cand_val_revert = {"overall_mae": 0.3696, "high_load_mae": 0.8409}
    confirmed, promoted = confirm_validation_candidate(cand_val_revert, base_val, "enhanced_features")
    assert not confirmed
    assert promoted == "baseline"

    # Case B: Both overall and high-load satisfy constraints
    cand_val_promote = {"overall_mae": 0.3696, "high_load_mae": 0.8300}
    confirmed, promoted = confirm_validation_candidate(cand_val_promote, base_val, "enhanced_features")
    assert confirmed
    assert promoted == "enhanced_features"


def test_align_baseline_test_predictions_reindexing_and_validation(tmp_path):
    """Verify align_baseline_test_predictions correctly reindexes and rejects corrupted files."""
    ts = pd.date_range("2020-01-01 00:00:00", periods=5, freq="1h")
    actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    preds = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

    # Save out-of-order CSV
    reordered_idx = [ts[2], ts[0], ts[4], ts[1], ts[3]]
    csv_file = tmp_path / "baseline_preds.csv"
    pd.DataFrame({
        "timestamp": reordered_idx,
        "actual": [actuals[2], actuals[0], actuals[4], actuals[1], actuals[3]],
        "XGBoost": [preds[2], preds[0], preds[4], preds[1], preds[3]],
    }).to_csv(csv_file, index=False)

    aligned = align_baseline_test_predictions(csv_file, test_timestamps=ts, expected_actual=actuals)
    np.testing.assert_allclose(aligned, preds)

    # Corrupt actual -> raise ValueError
    corrupt_csv = tmp_path / "corrupt_preds.csv"
    pd.DataFrame({
        "timestamp": ts,
        "actual": actuals + 9.0,
        "XGBoost": preds,
    }).to_csv(corrupt_csv, index=False)
    with pytest.raises(ValueError, match="differs"):
        align_baseline_test_predictions(corrupt_csv, test_timestamps=ts, expected_actual=actuals)
