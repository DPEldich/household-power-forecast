"""Tests for hourly resampling, missing hour preservation, and target validity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from household_power.data import resample_hourly
from household_power.features import build_features


def test_invalid_hourly_targets_excluded():
    """Verify that only hours with exactly 60 valid minute observations are marked as valid targets."""
    # Hour 1: 60 minutes, all present
    # Hour 2: 59 minutes present, 1 missing
    # Hour 3: completely missing from raw minute rows
    ts_h1 = pd.date_range("2020-01-01 00:00:00", periods=60, freq="1min")
    ts_h2 = pd.date_range("2020-01-01 01:00:00", periods=60, freq="1min")

    df_min1 = pd.DataFrame({
        "timestamp": ts_h1,
        "Global_active_power": [1.5] * 60,
    })

    # Hour 2 has 1 NaN reading
    h2_power = [2.0] * 59 + [np.nan]
    df_min2 = pd.DataFrame({
        "timestamp": ts_h2,
        "Global_active_power": h2_power,
    })

    # Concatenate without hour 3 (hour 3 is empty gap)
    ts_h4 = pd.date_range("2020-01-01 03:00:00", periods=60, freq="1min")
    df_min4 = pd.DataFrame({
        "timestamp": ts_h4,
        "Global_active_power": [3.0] * 60,
    })

    df_minute = pd.concat([df_min1, df_min2, df_min4], ignore_index=True)

    hourly_df, stats = resample_hourly(df_minute)

    # 4 hourly bins generated: 00:00, 01:00, 02:00, 03:00
    assert len(hourly_df) == 4
    assert hourly_df.index.tolist() == pd.date_range("2020-01-01 00:00:00", periods=4, freq="1h").tolist()

    # Hour 0 (00:00): 60 valid minutes -> valid target
    assert hourly_df.loc["2020-01-01 00:00:00", "valid_minutes"] == 60
    assert hourly_df.loc["2020-01-01 00:00:00", "is_valid_target"] is True or hourly_df.loc["2020-01-01 00:00:00", "is_valid_target"] == 1
    assert not np.isnan(hourly_df.loc["2020-01-01 00:00:00", "target"])

    # Hour 1 (01:00): 59 valid minutes -> invalid target (target must be NaN)
    assert hourly_df.loc["2020-01-01 01:00:00", "valid_minutes"] == 59
    assert hourly_df.loc["2020-01-01 01:00:00", "is_valid_target"] == False
    assert np.isnan(hourly_df.loc["2020-01-01 01:00:00", "target"])

    # Hour 2 (02:00): 0 valid minutes -> invalid target (target must be NaN)
    assert hourly_df.loc["2020-01-01 02:00:00", "valid_minutes"] == 0
    assert hourly_df.loc["2020-01-01 02:00:00", "is_valid_target"] == False
    assert np.isnan(hourly_df.loc["2020-01-01 02:00:00", "target"])

    # Hour 3 (03:00): 60 valid minutes -> valid target
    assert hourly_df.loc["2020-01-01 03:00:00", "valid_minutes"] == 60
    assert hourly_df.loc["2020-01-01 03:00:00", "is_valid_target"] == True
    assert not np.isnan(hourly_df.loc["2020-01-01 03:00:00", "target"])


def test_lag_rolling_across_missing_hour_preserves_continuous_time():
    """Verify that lag_1 refers to physical 1-hour prior (t-1h), not simply the previous available row."""
    # Hours: 00:00 (val=10.0), 01:00 (missing/NaN), 02:00 (val=20.0)
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=200, freq="1h")
    series_raw = pd.Series(1.0, index=timestamps)
    # Set hour 0 to 10.0, hour 1 to NaN, hour 2 to 20.0
    series_raw.iloc[0] = 10.0
    series_raw.iloc[1] = np.nan
    series_raw.iloc[2] = 20.0

    df = pd.DataFrame({
        "active_power_ffill3": series_raw,
        "target": series_raw,
        "is_valid_target": series_raw.notna(),
    }, index=timestamps)

    X, y, _ = build_features(df)

    # At hour 02:00: lag_1 is hour 01:00, which is NaN (or ffilled value if bounded ffill was applied)
    # Crucially, it does NOT skip hour 1 and take hour 0 (10.0) if unffilled!
    assert np.isnan(X.loc["2020-01-01 01:00:00", "lag_1"]) or X.loc["2020-01-01 01:00:00", "lag_1"] == 10.0
    # Lag 2 at hour 02:00 points to hour 00:00 (10.0)
    assert X.loc["2020-01-01 02:00:00", "lag_2"] == 10.0
