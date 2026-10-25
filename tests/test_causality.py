"""Tests for causal integrity and future value invariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from household_power.features import build_features


def test_causal_feature_invariance():
    """Verify that features at target hour s are strictly invariant to any mutation of values at s or s + k (k > 0)."""
    # Create 200 hours of continuous synthetic data
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=200, freq="1h")
    np.random.seed(42)
    values = np.random.uniform(0.5, 3.5, size=200)

    df1 = pd.DataFrame(
        {
            "active_power_ffill3": values.copy(),
            "target": values.copy(),
            "is_valid_target": True,
        },
        index=timestamps,
    )

    X1, y1, _ = build_features(df1)

    # Pick a target hour s after warmup (e.g., index 175)
    s_idx = 175
    target_hour = timestamps[s_idx]
    features_before = X1.loc[target_hour].copy()

    # Create mutated series: mutate current hour s and all subsequent future hours
    df2 = df1.copy()
    mutated_values = values.copy()
    mutated_values[s_idx:] = mutated_values[s_idx:] * 100.0 + 999.0  # Drastic changes
    df2["active_power_ffill3"] = mutated_values
    df2["target"] = mutated_values

    X2, y2, _ = build_features(df2)
    features_after = X2.loc[target_hour].copy()

    # All features at target hour s MUST be 100% identical
    pd.testing.assert_series_equal(
        features_before,
        features_after,
        check_exact=True,
        obj="Features at hour s after future mutation",
    )
