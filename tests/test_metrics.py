"""Tests for evaluation metrics and common-mask parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from household_power.evaluation import calculate_metrics, compute_common_evaluation_mask


def test_metrics_calculation_correctness():
    """Verify MAE, RMSE, R2, and relative improvement calculations on synthetic inputs."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.2, 1.8, 3.1, 3.9])
    # abs errors: 0.2, 0.2, 0.1, 0.1 -> MAE = 0.6 / 4 = 0.15
    # sq errors: 0.04, 0.04, 0.01, 0.01 -> MSE = 0.10 / 4 = 0.025 -> RMSE = sqrt(0.025)

    persistence_mae = 0.30
    seasonal_mae = 0.60

    m = calculate_metrics(
        y_true,
        y_pred,
        persistence_mae=persistence_mae,
        seasonal_naive_mae=seasonal_mae,
        model_name="TestModel",
    )

    assert np.isclose(m.mae, 0.15, atol=1e-4)
    assert np.isclose(m.rmse, np.sqrt(0.025), atol=1e-4)
    # Improvement over persistence: (0.30 - 0.15) / 0.30 * 100% = 50.0%
    assert np.isclose(m.mae_improvement_vs_persistence_pct, 50.0, atol=1e-4)
    # Improvement over seasonal: (0.60 - 0.15) / 0.60 * 100% = 75.0%
    assert np.isclose(m.mae_improvement_vs_seasonal_naive_pct, 75.0, atol=1e-4)


def test_persistence_improvement_vs_seasonal_naive_calculated():
    """Verify that Persistence improvement over Seasonal Naive is computed accurately, not hardcoded to 0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    persistence_pred = np.array([1.2, 1.9, 3.1, 3.8])  # MAE = 0.15
    seasonal_pred = np.array([1.5, 2.5, 3.5, 4.5])     # MAE = 0.50

    p_mae = 0.15
    s_mae = 0.50

    m_persist = calculate_metrics(
        y_true,
        persistence_pred,
        persistence_mae=p_mae,
        seasonal_naive_mae=s_mae,
        model_name="Persistence",
    )

    # Persistence vs itself is 0%
    assert np.isclose(m_persist.mae_improvement_vs_persistence_pct, 0.0, atol=1e-4)
    # Persistence vs Seasonal Naive: (0.50 - 0.15) / 0.50 * 100% = 70.0%
    assert np.isclose(m_persist.mae_improvement_vs_seasonal_naive_pct, 70.0, atol=1e-4)


def test_calculate_metrics_rejects_nonfinite():
    """Verify calculate_metrics raises ValueError when inputs contain NaN or Inf."""
    y_true = np.array([1.0, np.nan, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Inputs contain non-finite"):
        calculate_metrics(y_true, y_pred)

    y_true2 = np.array([1.0, 2.0, 3.0])
    y_pred2 = np.array([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="Inputs contain non-finite"):
        calculate_metrics(y_true2, y_pred2)


def test_calculate_metrics_rejects_shape_mismatch():
    """Verify calculate_metrics raises ValueError on mismatched lengths or multi-dimensional arrays."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_metrics(y_true, y_pred)

    y_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="Inputs must be 1D"):
        calculate_metrics(y_2d, y_2d)


def test_calculate_metrics_rejects_empty():
    """Verify calculate_metrics raises ValueError on empty input arrays."""
    with pytest.raises(ValueError, match="Inputs must not be empty"):
        calculate_metrics(np.array([]), np.array([]))


def test_common_mask_parity():
    """Verify that common mask filters out rows where baseline lag is missing."""
    timestamps = pd.date_range("2020-01-01", periods=10, freq="1h")
    df = pd.DataFrame({
        "target": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
        "is_valid_target": [True, True, True, True, True, False, True, True, True, True],
        "lag_1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "lag_24": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0],
    }, index=timestamps)

    mask = pd.Series(True, index=timestamps)
    common_mask, stats = compute_common_evaluation_mask(
        df, mask, lag_cols=["lag_1", "lag_24"], total_calendar_hours=10
    )

    # Row 2 has lag_1 = NaN -> excluded
    assert common_mask.iloc[2] is False or common_mask.iloc[2] == 0
    # Row 4 has lag_24 = NaN -> excluded
    assert common_mask.iloc[4] is False or common_mask.iloc[4] == 0
    # Row 5 has is_valid_target = False -> excluded
    assert common_mask.iloc[5] is False or common_mask.iloc[5] == 0

    # Total excluded = 3 rows (indices 2, 4, 5) -> 7 valid
    assert stats["common_valid_hours"] == 7
    assert stats["calendar_hours"] == 10
    assert stats["invalid_target_hours"] == 0
    assert stats["baseline_excluded_hours"] == 3
