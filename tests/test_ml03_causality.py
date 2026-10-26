"""Tests for causal invariance of extended features under perturbation and continuous grid gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from household_power.features import EXTENDED_FEATURES, build_extended_features, build_features


def test_extended_features_causal_invariance_under_current_and_future_perturbations():
    """Verify that perturbing readings at hour s and future hours > s has ZERO effect on features at s."""
    n_hours = 250
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=n_hours, freq="1h")

    # Generate synthetic active power pattern
    np.random.seed(42)
    base_values = 1.5 + 0.8 * np.sin(2 * np.pi * np.arange(n_hours) / 24.0) + np.random.uniform(0, 0.5, n_hours)

    df_clean = pd.DataFrame(
        {
            "active_power_ffill3": base_values.copy(),
            "target": np.roll(base_values, -1),
        },
        index=timestamps,
    )

    # Compute baseline features at target hour s = 200
    target_idx = 200
    target_ts = timestamps[target_idx]

    X_clean, _, valid_clean = build_extended_features(df_clean)
    assert valid_clean.loc[target_ts]
    features_clean_s = X_clean.loc[target_ts].to_dict()

    # Perturb current reading at s and all future readings >= s
    df_perturbed = df_clean.copy()
    df_perturbed.loc[target_ts:, "active_power_ffill3"] += 999.0
    df_perturbed.loc[target_ts:, "target"] += 888.0

    X_perturbed, _, valid_perturbed = build_extended_features(df_perturbed)
    features_perturbed_s = X_perturbed.loc[target_ts].to_dict()

    # Every single extended feature at hour s must be strictly identical
    for col in EXTENDED_FEATURES:
        val_clean = features_clean_s[col]
        val_pert = features_perturbed_s[col]
        assert np.isclose(val_clean, val_pert, atol=1e-12, rtol=1e-12), (
            f"Causal leakage detected in extended feature '{col}' at hour {target_ts}: "
            f"clean={val_clean}, perturbed={val_pert}"
        )


def test_extended_features_requires_strict_continuous_grid():
    """Verify that gaps or duplicate timestamps are strictly rejected."""
    timestamps = pd.date_range("2020-01-01", periods=10, freq="2h")  # 2-hour spacing
    df = pd.DataFrame({"active_power_ffill3": 1.0, "target": 1.0}, index=timestamps)
    with pytest.raises(ValueError, match="1-hour spacing"):
        build_extended_features(df)

    # Monotonicity failure
    shuffled_idx = timestamps.tolist()
    shuffled_idx[1], shuffled_idx[2] = shuffled_idx[2], shuffled_idx[1]
    df_shuffled = pd.DataFrame({"active_power_ffill3": 1.0, "target": 1.0}, index=shuffled_idx)
    with pytest.raises(ValueError, match="monotonic"):
        build_extended_features(df_shuffled)
