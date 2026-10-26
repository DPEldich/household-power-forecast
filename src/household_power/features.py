"""Feature engineering for next-hour household power forecasting.

Provides strictly causal lag, rolling, calendar, and cyclic features.
Excludes the initial 168-hour warmup consistently.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WARMUP_HOURS = 168
LAG_WINDOWS = [1, 2, 3, 24, 48, 168]
ROLLING_WINDOWS = [3, 24, 168]

CALENDAR_FEATURES = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
]

LAG_FEATURES = [f"lag_{w}" for w in LAG_WINDOWS]
ROLLING_FEATURES = (
    [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
)

FULL_FEATURES = LAG_FEATURES + ROLLING_FEATURES + CALENDAR_FEATURES

FEATURE_SETS: Dict[str, List[str]] = {
    "calendar_only": CALENDAR_FEATURES,
    "lag_rolling_only": LAG_FEATURES + ROLLING_FEATURES,
    "full_minus_rolling": LAG_FEATURES + CALENDAR_FEATURES,
    "full": FULL_FEATURES,
}


def build_features(
    df: pd.DataFrame,
    input_col: str = "active_power_ffill3",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate lag, rolling, calendar, and cyclic features for next-hour forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with continuous DatetimeIndex at hourly frequency,
        containing input_col and 'target' column.
    input_col : str
        Column to compute historical lag and rolling stats from.

    Returns
    -------
    X : pd.DataFrame
        Full feature matrix indexed by target hour s.
    y : pd.Series
        Target active power kW for interval [s, s+1h).
    valid_mask : pd.Series (bool)
        Mask indicating rows where target is valid (60 nonmissing minutes)
        AND beyond the 168-hour warmup.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a continuous pd.DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be strictly monotonic increasing.")
    if not df.index.is_unique:
        raise ValueError("DatetimeIndex must contain unique timestamps without duplicates.")
    if len(df.index) > 1:
        diffs = df.index[1:] - df.index[:-1]
        if not (diffs == pd.Timedelta(hours=1)).all():
            raise ValueError("DatetimeIndex must be continuous with exact 1-hour spacing.")

    series = df[input_col]

    features = pd.DataFrame(index=df.index)

    # 1. Strictly causal lags: values strictly prior to target hour s
    for lag in LAG_WINDOWS:
        features[f"lag_{lag}"] = series.shift(lag)

    # 2. Strictly causal rolling statistics: shift(1) ensures no current-hour data enters window
    shifted_series = series.shift(1)
    for window in ROLLING_WINDOWS:
        features[f"rolling_mean_{window}"] = (
            shifted_series.rolling(window=window, min_periods=1).mean()
        )
        # For std, fill single-point standard deviation with 0.0
        features[f"rolling_std_{window}"] = (
            shifted_series.rolling(window=window, min_periods=2).std().fillna(0.0)
        )

    # 3. Calendar & Cyclic features based on target timestamp s
    hour = df.index.hour.astype(int)
    dayofweek = (df.index.day_of_week if hasattr(df.index, "day_of_week") else df.index.dayofweek).astype(int)
    month = df.index.month.astype(int)

    features["hour"] = hour
    features["dayofweek"] = dayofweek
    features["month"] = month
    features["is_weekend"] = (dayofweek >= 5).astype(int)

    features["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    features["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    features["sin_dow"] = np.sin(2 * np.pi * dayofweek / 7.0)
    features["cos_dow"] = np.cos(2 * np.pi * dayofweek / 7.0)

    # Target series
    y = df["target"].copy()

    # Base valid mask: target is valid AND index >= index[0] + 168 hours
    warmup_cutoff = df.index[0] + pd.Timedelta(hours=WARMUP_HOURS)
    after_warmup = df.index >= warmup_cutoff

    if "is_valid_target" in df.columns:
        is_valid_target = df["is_valid_target"]
    else:
        is_valid_target = y.notna()

    valid_mask = after_warmup & is_valid_target

    return features, y, valid_mask


EXTENDED_LOAD_FEATURES = [
    "diff_1h",
    "diff_24h",
    "rolling_max_3",
    "rolling_max_24",
    "trend_3h",
    "missing_lag1",
]

# Note: In build_features, rolling columns are added per-window (mean_w, std_w).
# BASE_FEATURE_NAMES lists the exact column order produced by build_features.
BASE_FEATURE_NAMES = [
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168",
    "rolling_mean_3", "rolling_std_3",
    "rolling_mean_24", "rolling_std_24",
    "rolling_mean_168", "rolling_std_168",
    "hour", "dayofweek", "month", "is_weekend",
    "sin_hour", "cos_hour", "sin_dow", "cos_dow",
]

EXTENDED_FEATURES = BASE_FEATURE_NAMES + EXTENDED_LOAD_FEATURES


def build_extended_features(
    df: pd.DataFrame,
    input_col: str = "active_power_ffill3",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate extended causal load-dynamics features along with baseline features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with continuous DatetimeIndex at hourly frequency,
        containing input_col and 'target' column.
    input_col : str
        Column to compute historical stats from.

    Returns
    -------
    X : pd.DataFrame
        Extended feature matrix (26 features) indexed by target hour s.
    y : pd.Series
        Target active power kW for interval [s, s+1h).
    valid_mask : pd.Series (bool)
        Mask indicating rows where target is valid AND beyond 168h warmup.
    """
    features, y, valid_mask = build_features(df, input_col=input_col)

    series = df[input_col]
    shifted_series = series.shift(1)

    ext = pd.DataFrame(index=df.index)
    ext["diff_1h"] = series.shift(1) - series.shift(2)
    ext["diff_24h"] = series.shift(1) - series.shift(25)
    ext["rolling_max_3"] = shifted_series.rolling(window=3, min_periods=1).max()
    ext["rolling_max_24"] = shifted_series.rolling(window=24, min_periods=1).max()
    ext["trend_3h"] = series.shift(1) - shifted_series.rolling(window=3, min_periods=1).mean()
    ext["missing_lag1"] = series.shift(1).isna().astype(float)

    full_ext = pd.concat([features, ext], axis=1)
    return full_ext, y, valid_mask
