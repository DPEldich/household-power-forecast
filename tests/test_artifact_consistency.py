"""Verification of real artifact consistency, metrics parity, and live inference."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
PRED_PATH = ARTIFACTS_DIR / "predictions" / "test_predictions.csv"
METRICS_PATH = ARTIFACTS_DIR / "metrics" / "test_metrics.csv"
VAL_METRICS_PATH = ARTIFACTS_DIR / "metrics" / "validation_metrics.csv"
META_PATH = ARTIFACTS_DIR / "metadata" / "split_metadata.json"
DATA_PATH = ROOT_DIR / "data" / "processed" / "hourly_power.parquet"

# Availability check
artifacts_present = (
    PRED_PATH.exists()
    and METRICS_PATH.exists()
    and VAL_METRICS_PATH.exists()
    and META_PATH.exists()
    and DATA_PATH.exists()
)


@pytest.mark.skipif(not artifacts_present, reason="Real artifacts not yet generated.")
def test_test_metrics_recomputation_consistency():
    """Recompute MAE and RMSE from test_predictions.csv and verify exact match with test_metrics.csv."""
    df_pred = pd.read_csv(PRED_PATH)
    df_metrics = pd.read_csv(METRICS_PATH).set_index("model_name")

    actual = df_pred["actual"].to_numpy()
    assert len(actual) == 4900, f"Expected 4900 test samples on common mask, got {len(actual)}"

    for model_name in df_metrics.index:
        assert model_name in df_pred.columns, f"Model {model_name} missing from predictions"
        pred = df_pred[model_name].to_numpy()

        recomputed_mae = float(mean_absolute_error(actual, pred))
        recomputed_rmse = float(np.sqrt(mean_squared_error(actual, pred)))

        saved_mae = float(df_metrics.loc[model_name, "mae"])
        saved_rmse = float(df_metrics.loc[model_name, "rmse"])

        assert np.isclose(recomputed_mae, saved_mae, atol=1e-5), f"MAE mismatch for {model_name}"
        assert np.isclose(recomputed_rmse, saved_rmse, atol=1e-5), f"RMSE mismatch for {model_name}"


@pytest.mark.skipif(not artifacts_present, reason="Real artifacts not yet generated.")
def test_validation_winner_selection_consistency():
    """Verify that validation_selected_winner in metadata strictly corresponds to the lowest MAE ML model on val."""
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    val_df = pd.read_csv(VAL_METRICS_PATH)
    ml_val = val_df[~val_df["model_name"].isin(["Persistence", "Seasonal Naive"])]
    lowest_mae_model = ml_val.sort_values("mae").iloc[0]["model_name"]

    assert meta["validation_selected_winner"] == lowest_mae_model


@pytest.mark.skipif(not artifacts_present, reason="Real artifacts not yet generated.")
def test_live_predict_matches_saved_predictions():
    """Verify that live predict from saved joblib models exactly matches precomputed predictions."""
    from household_power.features import build_features

    hourly_df = pd.read_parquet(DATA_PATH)
    X_features, _, _ = build_features(hourly_df)

    df_pred = pd.read_csv(PRED_PATH)
    # Test on a specific row
    test_idx = 100
    target_time = pd.Timestamp(df_pred["timestamp"].iloc[test_idx])
    x_sample = X_features.loc[[target_time]]

    # 1. Test Persistence
    persist_model = joblib.load(ARTIFACTS_DIR / "models" / "persistence.joblib")
    persist_live = float(persist_model.predict(x_sample)[0])
    persist_saved = float(df_pred["Persistence"].iloc[test_idx])
    assert np.isclose(persist_live, persist_saved, atol=1e-5)

    # 2. Test Winner Model
    winner_model = joblib.load(ARTIFACTS_DIR / "models" / "winner_model.joblib")
    winner_live = float(winner_model.predict(x_sample)[0])
    with open(META_PATH, "r", encoding="utf-8") as f:
        winner_name = json.load(f)["validation_selected_winner"]
    winner_saved = float(df_pred[winner_name].iloc[test_idx])
    assert np.isclose(winner_live, winner_saved, atol=1e-5)
