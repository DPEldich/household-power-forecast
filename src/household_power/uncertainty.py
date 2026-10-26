"""Uncertainty quantification utilities, pinball loss, and quantile crossing repair.

Shared between offline experiments and online serving.
Zero machine learning / training framework dependencies (pure NumPy).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, alpha: float) -> float:
    """Compute pinball (quantile) loss for quantile alpha in (0, 1).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth 1D array.
    y_pred_q : np.ndarray
        Quantile predictions 1D array.
    alpha : float
        Target quantile level in (0, 1).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Quantile alpha must be in (0, 1), got {alpha}")

    y_t = np.asarray(y_true, dtype=float)
    y_q = np.asarray(y_pred_q, dtype=float)

    if y_t.ndim != 1 or y_q.ndim != 1:
        raise ValueError(f"Inputs must be 1D arrays, got y_true ndim={y_t.ndim}, y_pred_q ndim={y_q.ndim}")
    if y_t.shape != y_q.shape:
        raise ValueError(f"Shape mismatch: y_true shape={y_t.shape} != y_pred_q shape={y_q.shape}")
    if len(y_t) == 0:
        return float("nan")
    if not (np.isfinite(y_t).all() and np.isfinite(y_q).all()):
        raise ValueError("pinball_loss requires finite inputs (no NaN or Inf)")

    diff = y_t - y_q
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


def repair_and_clip_quantiles(
    raw_p10: np.ndarray | list[float],
    raw_p50: np.ndarray | list[float],
    raw_p90: np.ndarray | list[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically clip to >= 0.0 kW and enforce 0 <= P10 <= P50 <= P90.

    Ensures no negative power predictions and repairs quantile crossing.
    """
    r10 = np.asarray(raw_p10, dtype=float)
    r50 = np.asarray(raw_p50, dtype=float)
    r90 = np.asarray(raw_p90, dtype=float)

    if r10.ndim != 1 or r50.ndim != 1 or r90.ndim != 1:
        raise ValueError("Quantile predictions must be 1D arrays")
    if not (len(r10) == len(r50) == len(r90)):
        raise ValueError(f"Array length mismatch: p10={len(r10)}, p50={len(r50)}, p90={len(r90)}")
    if len(r10) == 0:
        return np.array([]), np.array([]), np.array([])
    if not (np.isfinite(r10).all() and np.isfinite(r50).all() and np.isfinite(r90).all()):
        raise ValueError("Quantile predictions must be finite (no NaN or Inf)")

    c10 = np.clip(r10, 0.0, None)
    c50 = np.clip(r50, 0.0, None)
    c90 = np.clip(r90, 0.0, None)

    stacked = np.column_stack([c10, c50, c90])
    sorted_q = np.sort(stacked, axis=1)

    return sorted_q[:, 0], sorted_q[:, 1], sorted_q[:, 2]


def evaluate_uncertainty(
    y_true: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    threshold_p90: float,
) -> Dict[str, Dict[str, Any]]:
    """Compute pinball loss per quantile, empirical coverage, below/above rates, and width.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth 1D array.
    p10, p50, p90 : np.ndarray
        Deterministic, non-negative, monotonicity-repaired quantiles.
    threshold_p90 : float
        High-load cutoff computed strictly from training data.
    """
    y_t = np.asarray(y_true, dtype=float)
    p10_arr = np.asarray(p10, dtype=float)
    p50_arr = np.asarray(p50, dtype=float)
    p90_arr = np.asarray(p90, dtype=float)

    for name, arr in [("y_true", y_t), ("p10", p10_arr), ("p50", p50_arr), ("p90", p90_arr)]:
        if arr.ndim != 1:
            raise ValueError(f"Input '{name}' must be 1D, got ndim={arr.ndim}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Input '{name}' contains non-finite values (NaN or Inf)")

    if not (len(y_t) == len(p10_arr) == len(p50_arr) == len(p90_arr)):
        raise ValueError(
            f"Shape mismatch: y_true={len(y_t)}, p10={len(p10_arr)}, p50={len(p50_arr)}, p90={len(p90_arr)}"
        )
    if len(y_t) == 0:
        raise ValueError("Inputs must not be empty")

    if not np.isfinite(threshold_p90):
        raise ValueError(f"threshold_p90 must be a finite float, got {threshold_p90}")

    # Verify input quantiles are already repaired and monotonic
    if not ((p10_arr <= p50_arr + 1e-12).all() and (p50_arr <= p90_arr + 1e-12).all()):
        raise ValueError("Input quantiles violate monotonicity (repair_and_clip_quantiles required first)")

    high_mask = y_t >= threshold_p90
    normal_mask = ~high_mask

    subsets = {
        "overall": np.ones(len(y_t), dtype=bool),
        "high_load": high_mask,
        "normal_load": normal_mask,
    }

    results: Dict[str, Dict[str, Any]] = {}
    for name, mask in subsets.items():
        sub_y = y_t[mask]
        sub_10 = p10_arr[mask]
        sub_50 = p50_arr[mask]
        sub_90 = p90_arr[mask]
        n_samples = int(len(sub_y))

        if n_samples == 0:
            results[name] = {
                "count": 0,
                "coverage_pct": float("nan"),
                "below_lower_pct": float("nan"),
                "above_upper_pct": float("nan"),
                "undercoverage_pct": float("nan"),
                "overcoverage_pct": float("nan"),
                "mean_width_kw": float("nan"),
                "median_width_kw": float("nan"),
                "pinball_p10": float("nan"),
                "pinball_p50": float("nan"),
                "pinball_p90": float("nan"),
                "mean_pinball": float("nan"),
            }
            continue

        covered = (sub_y >= sub_10) & (sub_y <= sub_90)
        below = sub_y < sub_10
        above = sub_y > sub_90
        widths = sub_90 - sub_10

        pb10 = pinball_loss(sub_y, sub_10, 0.1)
        pb50 = pinball_loss(sub_y, sub_50, 0.5)
        pb90 = pinball_loss(sub_y, sub_90, 0.9)

        cov_pct = float(np.mean(covered) * 100.0)
        bel_pct = float(np.mean(below) * 100.0)
        abv_pct = float(np.mean(above) * 100.0)

        results[name] = {
            "count": n_samples,
            "coverage_pct": cov_pct,
            "below_lower_pct": bel_pct,
            "above_upper_pct": abv_pct,
            "undercoverage_pct": bel_pct,  # legacy alias
            "overcoverage_pct": abv_pct,   # legacy alias
            "mean_width_kw": float(np.mean(widths)),
            "median_width_kw": float(np.median(widths)),
            "pinball_p10": pb10,
            "pinball_p50": pb50,
            "pinball_p90": pb90,
            "mean_pinball": float((pb10 + pb50 + pb90) / 3.0),
        }

    return results
