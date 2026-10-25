"""Evaluation metrics, common-mask verification, and high-load breakdown."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    model_name: str
    mae: float
    rmse: float
    r2: float
    mae_improvement_vs_persistence_pct: float
    mae_improvement_vs_seasonal_naive_pct: float
    fit_time_seconds: float
    sample_count: int


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    persistence_mae: Optional[float] = None,
    seasonal_naive_mae: Optional[float] = None,
    fit_time_seconds: float = 0.0,
    model_name: str = "",
) -> ModelMetrics:
    """Compute MAE, RMSE, R2, and relative improvements against baselines.

    Requires finite, non-empty, same-shaped 1D arrays. Does not silently drop NaNs,
    guaranteeing evaluation strictly adheres to the common evaluation mask.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if y_t.ndim != 1 or y_p.ndim != 1:
        raise ValueError(
            f"Inputs must be 1D arrays, got y_true ndim={y_t.ndim}, y_pred ndim={y_p.ndim}"
        )
    if y_t.shape != y_p.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape={y_t.shape} != y_pred shape={y_p.shape}"
        )
    if len(y_t) == 0:
        raise ValueError("Inputs must not be empty.")
    if not (np.isfinite(y_t).all() and np.isfinite(y_p).all()):
        raise ValueError(
            "Inputs contain non-finite values (NaN or Inf). Common mask evaluation requires strictly valid arrays."
        )

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    if persistence_mae is not None and persistence_mae > 0:
        imp_p = float((persistence_mae - mae) / persistence_mae * 100.0)
    else:
        imp_p = 0.0

    if seasonal_naive_mae is not None and seasonal_naive_mae > 0:
        imp_s = float((seasonal_naive_mae - mae) / seasonal_naive_mae * 100.0)
    else:
        imp_s = 0.0

    return ModelMetrics(
        model_name=model_name,
        mae=mae,
        rmse=rmse,
        r2=r2,
        mae_improvement_vs_persistence_pct=imp_p,
        mae_improvement_vs_seasonal_naive_pct=imp_s,
        fit_time_seconds=float(fit_time_seconds),
        sample_count=len(y_t),
    )


def compute_common_evaluation_mask(
    df_or_features: pd.DataFrame,
    mask: pd.Series,
    lag_cols: List[str] = ("lag_1", "lag_24"),
    total_calendar_hours: Optional[int] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Determine the exact identical subset of timestamps where both target and baselines exist.

    Reports total calendar hours, invalid target count, valid target count,
    baseline excluded hours, and explicit coverage percentages.
    """
    valid_target_hours = int(mask.sum())
    common_mask = mask.copy()

    # If is_valid_target is present in the dataframe, enforce it
    if "is_valid_target" in df_or_features.columns:
        common_mask = common_mask & (df_or_features["is_valid_target"] == True)
    elif "target" in df_or_features.columns:
        common_mask = common_mask & df_or_features["target"].notna()

    # Baseline lags availability
    for col in lag_cols:
        if col in df_or_features.columns:
            common_mask = common_mask & df_or_features[col].notna()
        else:
            raise KeyError(f"Baseline column '{col}' missing from dataframe.")

    common_count = int(common_mask.sum())
    baseline_excluded = valid_target_hours - common_count

    cal_hours = total_calendar_hours if total_calendar_hours is not None else valid_target_hours
    invalid_target_hours = cal_hours - valid_target_hours

    stats = {
        "calendar_hours": cal_hours,
        "invalid_target_hours": invalid_target_hours,
        "valid_target_hours": valid_target_hours,
        "baseline_excluded_hours": baseline_excluded,
        "common_valid_hours": common_count,
        "valid_target_coverage_pct": (common_count / valid_target_hours * 100.0) if valid_target_hours > 0 else 0.0,
        "calendar_coverage_pct": (common_count / cal_hours * 100.0) if cal_hours > 0 else 0.0,
    }

    logger.info(
        f"Common evaluation mask: {common_count}/{cal_hours} calendar hours "
        f"({stats['calendar_coverage_pct']:.2f}% calendar coverage; "
        f"{common_count}/{valid_target_hours} valid target hours, "
        f"{stats['valid_target_coverage_pct']:.2f}% valid target coverage; "
        f"{baseline_excluded} baseline-excluded hours)."
    )
    return common_mask, stats


def evaluate_high_load(
    y_true: pd.Series,
    y_pred: pd.Series,
    threshold_p90: float,
) -> Dict[str, Dict[str, float]]:
    """Evaluate performance across high-load (>= p90 of train) vs other hours."""
    y_t = y_true.to_numpy()
    y_p = y_pred.to_numpy()

    high_mask = y_t >= threshold_p90
    other_mask = ~high_mask

    results = {}
    for subset_name, sub_mask in [("high_load", high_mask), ("normal_load", other_mask), ("overall", slice(None))]:
        sub_t = y_t[sub_mask]
        sub_p = y_p[sub_mask]
        if len(sub_t) > 0:
            results[subset_name] = {
                "count": int(len(sub_t)),
                "ratio_pct": float(len(sub_t) / len(y_t) * 100.0),
                "mae": float(mean_absolute_error(sub_t, sub_p)),
                "rmse": float(np.sqrt(mean_squared_error(sub_t, sub_p))),
                "mean_actual": float(np.mean(sub_t)),
                "mean_pred": float(np.mean(sub_p)),
            }
        else:
            results[subset_name] = {
                "count": 0,
                "ratio_pct": 0.0,
                "mae": np.nan,
                "rmse": np.nan,
                "mean_actual": np.nan,
                "mean_pred": np.nan,
            }
    return results


def metrics_to_dataframe(metrics_list: List[ModelMetrics]) -> pd.DataFrame:
    """Convert list of ModelMetrics to a clean pandas DataFrame."""
    rows = [asdict(m) for m in metrics_list]
    df = pd.DataFrame(rows)
    return df
