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
