"""API tests: Verify old /v1/predict path, new /v1/predict_interval path, and history parity."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from household_power.api import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    app = create_app(bundle_dir=ROOT / "deploy/bundle", api_key="")
    with TestClient(app) as test_client:
        yield test_client


def test_baseline_predict_preserves_7_models_and_parity(client):
    """Test that all 7 legacy models continue to function on /v1/predict."""
    dash = client.get("/v1/dashboard").json()
    version = dash["model_version"]
    preds = dash["predictions"]
    sample_row = preds[50]
    target_ts = sample_row["timestamp"]

    for model in ["Persistence", "Seasonal Naive", "Linear Regression", "Ridge", "Random Forest", "Gradient Boosting", "XGBoost"]:
        resp = client.post(
            "/v1/predict",
            json={
                "model": model,
                "target_timestamp": target_ts,
                "expected_model_version": version,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert np.isclose(data["prediction_kw"], sample_row[model], atol=1e-5)


def test_predict_interval_historical_replay(client):
    """Test /v1/predict_interval in historical replay mode."""
    dash = client.get("/v1/dashboard").json()
    version = dash["model_version"]

    assert "experiments" in dash
    exp_preds = dash["experiments"]["predictions"]
    sample_exp = exp_preds[50]
    target_ts = sample_exp["timestamp"]

    resp = client.post(
        "/v1/predict_interval",
        json={
            "target_timestamp": target_ts,
            "expected_model_version": version,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["model_version"] == version
    assert np.isclose(data["point_prediction_kw"], sample_exp["point_pred"], atol=1e-5)
    assert np.isclose(data["p10_kw"], sample_exp["p10"], atol=1e-5)
    assert np.isclose(data["p50_kw"], sample_exp["p50"], atol=1e-5)
    assert np.isclose(data["p90_kw"], sample_exp["p90"], atol=1e-5)
    assert data["p10_kw"] <= data["p50_kw"] <= data["p90_kw"]
    assert data["mode"] == "historical_replay"


def test_predict_interval_provided_history_parity(client):
    """Test /v1/predict_interval with provided history produces EXACT numerical parity with replay."""
    dash = client.get("/v1/dashboard").json()
    version = dash["model_version"]

    exp_preds = dash["experiments"]["predictions"]
    target_ts = exp_preds[50]["timestamp"]

    # Pull 168 hours of history from history endpoint
    hist_resp = client.get("/v1/history", params={"target_timestamp": target_ts, "hours": 168})
    assert hist_resp.status_code == 200, hist_resp.text
    history_items = hist_resp.json()["history"]

    # Prepare observation payloads for predict_interval
    observations = [
        {"timestamp": item["timestamp"], "active_power_kw": item["raw_active_power"]}
        for item in history_items
    ]

    # Call predict_interval with provided history
    resp = client.post(
        "/v1/predict_interval",
        json={
            "target_timestamp": target_ts,
            "expected_model_version": version,
            "history": observations,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Replay prediction
    replay_resp = client.post(
        "/v1/predict_interval",
        json={
            "target_timestamp": target_ts,
            "expected_model_version": version,
        },
    )
    replay_data = replay_resp.json()

    # Must match to numerical tolerance
    assert np.isclose(data["point_prediction_kw"], replay_data["point_prediction_kw"], atol=1e-5)
    assert np.isclose(data["p10_kw"], replay_data["p10_kw"], atol=1e-5)
    assert np.isclose(data["p50_kw"], replay_data["p50_kw"], atol=1e-5)
    assert np.isclose(data["p90_kw"], replay_data["p90_kw"], atol=1e-5)
    assert data["mode"] == "provided_history"


def test_predict_interval_validation_guards(client):
    """Test version mismatch, non-hourly timestamp, and insufficient history guards."""
    dash = client.get("/v1/dashboard").json()
    version = dash["model_version"]
    target_ts = dash["predictions"][0]["timestamp"]

    # 1. Version mismatch -> 409
    resp = client.post(
        "/v1/predict_interval",
        json={"target_timestamp": target_ts, "expected_model_version": "wrong-version"},
    )
    assert resp.status_code == 409

    # 2. Non-hourly timestamp -> 422
    resp = client.post(
        "/v1/predict_interval",
        json={"target_timestamp": "2010-05-01T12:30:00", "expected_model_version": version},
    )
    assert resp.status_code == 422

    # 3. Insufficient history (< 168 hours) -> 422
    resp = client.post(
        "/v1/predict_interval",
        json={
            "target_timestamp": target_ts,
            "expected_model_version": version,
            "history": [{"timestamp": "2010-05-01T11:00:00", "active_power_kw": 1.2}],
        },
    )
    assert resp.status_code == 422
