"""Publication-quality plotting module for models, residuals, ablation, and error distributions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Aesthetic styling
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def plot_one_week_forecast(
    df_pred: pd.DataFrame,
    winner_col: str,
    output_path: str | Path,
    start_idx: int = 100,
    num_days: int = 7,
):
    """Plot 7-day (168 continuous hours) actual vs predicted series on continuous time."""
    if not isinstance(df_pred.index, pd.DatetimeIndex):
        if "timestamp" in df_pred.columns:
            df_pred = df_pred.set_index(pd.to_datetime(df_pred["timestamp"]))

    start_ts = df_pred.index[min(start_idx, len(df_pred) - 1)]
    end_ts = start_ts + pd.Timedelta(days=num_days)

    # Reindex on exact continuous hourly range so any missing hours appear as gaps
    continuous_hourly_range = pd.date_range(start=start_ts, end=end_ts - pd.Timedelta(hours=1), freq="1h")
    subset = df_pred.reindex(continuous_hourly_range)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        subset.index,
        subset["actual"],
        label="Actual Ground Truth (kW)",
        color="black",
        linewidth=2.0,
        alpha=0.85,
    )

    if winner_col in subset.columns:
        ax.plot(
            subset.index,
            subset[winner_col],
            label=f"Validation Winner ({winner_col})",
            color="#1f77b4",
            linewidth=1.8,
            linestyle="-",
        )

    if "Persistence" in subset.columns:
        ax.plot(
            subset.index,
            subset["Persistence"],
            label="Persistence (Lag 1)",
            color="#ff7f0e",
            linewidth=1.2,
            linestyle="--",
            alpha=0.75,
        )

    if "Seasonal Naive" in subset.columns:
        ax.plot(
            subset.index,
            subset["Seasonal Naive"],
            label="Seasonal Naive (Lag 24)",
            color="#2ca02c",
            linewidth=1.2,
            linestyle=":",
            alpha=0.75,
        )

    ax.set_title(f"7-Day Next-Hour Active Power Forecast vs Actual (Continuous Timeline: {start_ts.strftime('%Y-%m-%d')} to {end_ts.strftime('%Y-%m-%d')})")
    ax.set_xlabel("Time (Continuous Hourly interval [s, s+1h))")
    ax.set_ylabel("Active Power (kW)")
    ax.legend(loc="upper right", framealpha=0.9)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=200)
    plt.close(fig)
    logger.info(f"Saved 1-week forecast plot to {out_p}")


def plot_residuals_over_time(
    df_pred: pd.DataFrame,
    winner_col: str,
    output_path: str | Path,
):
    """Plot residual series (y - y_hat) over continuous test set timeline."""
    if not isinstance(df_pred.index, pd.DatetimeIndex):
        if "timestamp" in df_pred.columns:
            df_pred = df_pred.set_index(pd.to_datetime(df_pred["timestamp"]))

    # Reindex on continuous hourly range
    continuous_idx = pd.date_range(start=df_pred.index[0], end=df_pred.index[-1], freq="1h")
    df_cont = df_pred.reindex(continuous_idx)

    residuals = df_cont["actual"] - df_cont[winner_col]
    rolling_mean_res = residuals.rolling(window=168, min_periods=24).mean()
    rolling_std_res = residuals.rolling(window=168, min_periods=24).std()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    ax1.scatter(df_cont.index, residuals, alpha=0.20, s=5, color="#4a7ebb", label="Instantaneous Residual (y - ŷ)")
    ax1.plot(df_cont.index, rolling_mean_res, color="#d62728", linewidth=2.0, label="7-Day (168h) Rolling Mean Bias")
    ax1.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_ylabel("Residual (kW)")
    ax1.set_title(f"Continuous Residual Analysis Over Test Set — Validation Winner: {winner_col}")
    ax1.legend(loc="upper right")

    ax2.plot(df_cont.index, rolling_std_res, color="#9467bd", linewidth=1.5, label="7-Day (168h) Rolling Residual Std (Volatility)")
    ax2.set_xlabel("Test Set Continuous Timeline")
    ax2.set_ylabel("Residual Std (kW)")
    ax2.legend(loc="upper right")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=200)
    plt.close(fig)
    logger.info(f"Saved residuals plot to {out_p}")


def plot_mae_by_hour(
    df_pred: pd.DataFrame,
    model_cols: List[str],
    output_path: str | Path,
):
    """Plot diurnal MAE breakdown by hour of day (0 to 23)."""
    hours = df_pred.index.hour
    hourly_mae = {}

    for col in model_cols:
        err = np.abs(df_pred["actual"] - df_pred[col])
        hourly_mae[col] = err.groupby(hours).mean()

    df_h_mae = pd.DataFrame(hourly_mae)

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in model_cols:
        style = "-"
        width = 1.8
        alpha = 0.9
        if "Persistence" in col or "Seasonal" in col:
            style = "--"
            width = 1.3
            alpha = 0.7
        ax.plot(df_h_mae.index, df_h_mae[col], label=col, linestyle=style, linewidth=width, alpha=alpha, marker="o", markersize=3)

    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Hour of Day (Target interval [s, s+1h))")
    ax.set_ylabel("Mean Absolute Error (kW)")
    ax.set_title("Hourly Diurnal Forecast Error (MAE by Hour of Day)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), framealpha=0.9)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved diurnal MAE plot to {out_p}")


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    winner_name: Optional[str] = None,
):
    """Plot bar chart comparing MAE and RMSE across all seven models."""
    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(len(metrics_df))
    width = 0.35

    bar_colors_mae = ["#1f77b4" if m != winner_name else "#2ca02c" for m in metrics_df["model_name"]]
    bar_colors_rmse = ["#aec7e8" if m != winner_name else "#98df8a" for m in metrics_df["model_name"]]

    rects1 = ax.bar(x - width/2, metrics_df["mae"], width, label="MAE (kW)", color=bar_colors_mae)
    rects2 = ax.bar(x + width/2, metrics_df["rmse"], width, label="RMSE (kW)", color=bar_colors_rmse)

    ax.set_ylabel("Error (kW)")
    ax.set_title(f"Model Error Comparison on Held-Out Test Set (Validation-Selected Winner: {winner_name})")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["model_name"], rotation=25, ha="right")
    ax.legend()

    # Add numeric labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved model comparison plot to {out_p}")


def plot_ablation_comparison(
    ablation_df: pd.DataFrame,
    output_path: str | Path,
):
    """Plot ablation performance comparison."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    x = np.arange(len(ablation_df))
    width = 0.35

    rects1 = ax.bar(x - width/2, ablation_df["mae"], width, label="MAE (kW)", color="#e377c2")
    rects2 = ax.bar(x + width/2, ablation_df["rmse"], width, label="RMSE (kW)", color="#c5b0d5")

    ax.set_ylabel("Error (kW)")
    ax.set_title("Ablation Study: Feature Subset Impact on Validation Set")
    ax.set_xticks(x)
    labels = [c.replace("_", " ").title() for c in ablation_df["config_name"]]
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ablation plot to {out_p}")
