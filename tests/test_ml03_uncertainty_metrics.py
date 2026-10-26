"""Tests for quantile loss, deterministic crossing repair, and uncertainty evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from household_power.uncertainty import (
    evaluate_uncertainty,
    pinball_loss,
    repair_and_clip_quantiles,
)


def test_pinball_loss_exact_values_and_validation():
    """Verify pinball loss against analytical mathematical definitions and check validation."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    q = np.array([1.5, 1.5, 3.5, 2.0])

    pb50 = pinball_loss(y, q, alpha=0.5)
    mae = np.mean(np.abs(y - q))
    assert np.isclose(pb50 * 2.0, mae)

    pb10 = pinball_loss(y, q, alpha=0.1)
    assert np.isclose(pb10, 0.2875)

    # Empty array returns nan
    assert np.isnan(pinball_loss(np.array([]), np.array([]), alpha=0.1))

    # Invalid alpha
    with pytest.raises(ValueError, match="alpha"):
        pinball_loss(y, q, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        pinball_loss(y, q, alpha=1.0)

    # Non-finite inputs
    with pytest.raises(ValueError, match="finite"):
        pinball_loss(np.array([1.0, np.nan]), np.array([1.0, 2.0]), alpha=0.5)

    # 2D inputs
    with pytest.raises(ValueError, match="1D"):
        pinball_loss(np.array([[1.0]]), np.array([[1.0]]), alpha=0.5)


def test_repair_and_clip_quantiles_enforces_monotonicity_and_non_negativity():
    """Verify that scrambled and negative predictions are cleanly repaired: 0 <= P10 <= P50 <= P90."""
    raw_p10 = np.array([2.5, -1.0, 4.0, -5.0])
    raw_p50 = np.array([1.0,  0.5, 3.0, -2.0])
    raw_p90 = np.array([0.5, -3.0, 5.0, -1.0])

    p10, p50, p90 = repair_and_clip_quantiles(raw_p10, raw_p50, raw_p90)

    assert (p10 >= 0.0).all()
    assert (p50 >= 0.0).all()
    assert (p90 >= 0.0).all()
    assert (p10 <= p50).all()
    assert (p50 <= p90).all()

    # Empty inputs supported consistently
    e10, e50, e90 = repair_and_clip_quantiles(np.array([]), np.array([]), np.array([]))
    assert len(e10) == len(e50) == len(e90) == 0

    # Non-finite rejected
    with pytest.raises(ValueError, match="finite"):
        repair_and_clip_quantiles([np.nan], [1.0], [2.0])

    # 2D rejected
    with pytest.raises(ValueError, match="1D"):
        repair_and_clip_quantiles(np.array([[1.0]]), np.array([[2.0]]), np.array([[3.0]]))


def test_evaluate_uncertainty_validation_and_empty_high_load():
    """Verify evaluate_uncertainty rejects non-finite, unordered, and handles empty high-load."""
    y = np.array([1.0, 1.2, 1.5, 0.8])
    p10 = np.array([0.5, 0.6, 0.8, 0.4])
    p50 = np.array([1.0, 1.1, 1.4, 0.7])
    p90 = np.array([1.5, 1.6, 2.0, 1.2])

    threshold = 10.0
    results = evaluate_uncertainty(y, p10, p50, p90, threshold_p90=threshold)

    assert results["overall"]["count"] == 4
    assert np.isclose(results["overall"]["coverage_pct"], 100.0)
    assert np.isclose(results["overall"]["below_lower_pct"], 0.0)
    assert np.isclose(results["overall"]["above_upper_pct"], 0.0)

    # Empty high load
    assert results["high_load"]["count"] == 0
    assert np.isnan(results["high_load"]["coverage_pct"])
    assert np.isnan(results["high_load"]["mean_width_kw"])

    # Rejects non-finite inputs
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_uncertainty(np.array([1.0, np.nan]), p10[:2], p50[:2], p90[:2], threshold_p90=2.0)

    # Rejects unordered quantiles (violating monotonicity)
    with pytest.raises(ValueError, match="monotonicity"):
        evaluate_uncertainty(y, p90, p50, p10, threshold_p90=2.0)

    # Rejects non-finite threshold
    with pytest.raises(ValueError, match="finite"):
        evaluate_uncertainty(y, p10, p50, p90, threshold_p90=np.nan)
