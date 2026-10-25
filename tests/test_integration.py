"""Deterministic end-to-end integration test on synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from household_power.data import RawDataStats
from household_power.pipeline import PipelineRunner


def test_end_to_end_pipeline_integration(tmp_path: Path, monkeypatch):
    """Exercise complete PipelineRunner execution on synthetic small data.

    Verifies all steps: feature generation, expanding CV, validation, refit,
    test evaluation, artifact creation, and common mask integrity.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"

    # Create 500 hours of continuous hourly data
    np.random.seed(42)
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=500, freq="1h")
    values = np.random.uniform(0.5, 3.5, size=500).astype("float32")

    hourly_df = pd.DataFrame(
        {
            "valid_minutes": [60] * 500,
            "hourly_mean": values,
            "is_valid_target": [True] * 500,
            "target": values,
            "raw_active_power": values,
            "active_power_ffill3": values,
        },
        index=timestamps,
    )

    stats = RawDataStats(
        total_minute_rows=30000,
        missing_minute_readings=0,
        duplicate_minute_timestamps=0,
        first_minute_timestamp=str(timestamps[0]),
        last_minute_timestamp=str(timestamps[-1]),
        total_hourly_intervals=500,
        complete_target_hours=500,
        incomplete_target_hours=0,
        valid_target_ratio=1.0,
    )

    # Monkeypatch data loading to return deterministic synthetic data
    monkeypatch.setattr(
        "household_power.pipeline.prepare_and_cache_hourly_data",
        lambda *args, **kwargs: (hourly_df, stats),
    )

    runner = PipelineRunner(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        force_download=False,
        force_process=False,
    )

    result = runner.run()

    # 1. Verification of returned result
    assert "validation_winner" in result
    assert isinstance(result["validation_metrics"], pd.DataFrame)
    assert isinstance(result["test_metrics"], pd.DataFrame)
    assert len(result["test_metrics"]) == 7

    # 2. Verification of saved artifacts on disk
    metrics_dir = artifacts_dir / "metrics"
    models_dir = artifacts_dir / "models"
    metadata_dir = artifacts_dir / "metadata"
    predictions_dir = artifacts_dir / "predictions"
    figures_dir = artifacts_dir / "figures"

    assert (metrics_dir / "cv_metrics.csv").exists()
    assert (metrics_dir / "validation_metrics.csv").exists()
    assert (metrics_dir / "test_metrics.csv").exists()
    assert (metrics_dir / "ablation_metrics.csv").exists()
    assert (metrics_dir / "high_load_metrics.csv").exists()

    assert (metadata_dir / "split_metadata.json").exists()
    assert (predictions_dir / "test_predictions.csv").exists()

    # Models saved
    assert (models_dir / "persistence.joblib").exists()
    assert (models_dir / "seasonal_naive.joblib").exists()
    assert (models_dir / "winner_model.joblib").exists()

    # Figures saved
    assert (figures_dir / "one_week_actual_vs_predicted.png").exists()
    assert (figures_dir / "residuals_over_time.png").exists()
    assert (figures_dir / "mae_by_hour.png").exists()
    assert (figures_dir / "model_comparison.png").exists()
    assert (figures_dir / "ablation_comparison.png").exists()

    # 3. Check test predictions format
    test_pred_df = pd.read_csv(predictions_dir / "test_predictions.csv")
    assert "actual" in test_pred_df.columns
    assert "Persistence" in test_pred_df.columns
    assert "Seasonal Naive" in test_pred_df.columns
    assert result["validation_winner"] in test_pred_df.columns
    assert not test_pred_df["actual"].isna().any()
