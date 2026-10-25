"""Inference parity, causal validation, auth and immutable release checks."""
import json
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from household_power.api import create_app
from household_power.serving import ForecastService

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(api_key="")) as client:
        yield client


def request_for(client, model="XGBoost"):
    snapshot = client.get("/v1/dashboard").json()
    return {"model": model, "target_timestamp": snapshot["predictions"][240]["timestamp"],
            "expected_model_version": snapshot["model_version"]}


@pytest.mark.parametrize("model", ["Persistence", "Seasonal Naive", "Linear Regression", "Ridge",
                                    "Random Forest", "Gradient Boosting", "XGBoost"])
def test_api_inference_matches_real_stored_predictions(client, model):
    response = client.post("/v1/predict", json=request_for(client, model))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction_kw"] == pytest.approx(result["stored_prediction_kw"], abs=1e-6)
    assert result["unit"] == "kW"


def test_version_mismatch_and_unknown_model_fail_explicitly(client):
    body = request_for(client)
    body["expected_model_version"] = "old-release"
    assert client.post("/v1/predict", json=body).status_code == 409
    assert client.post("/v1/predict", json=request_for(client, "../../any-file")).status_code == 404


@pytest.mark.parametrize("timestamp", ["2010-05-01T12:30:00", "2010-05-01T12:00:00+07:00"])
def test_timestamp_convention_enforced(client, timestamp):
    body = request_for(client)
    body["target_timestamp"] = timestamp
    assert client.post("/v1/predict", json=body).status_code == 422


def test_api_key_protects_data_and_prediction():
    with TestClient(create_app(api_key="test-only-key")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/dashboard").status_code == 401
        assert client.get("/v1/dashboard", headers={"X-API-Key": "incorrect"}).status_code == 401
        assert client.get("/v1/dashboard", headers={"X-API-Key": "test-only-key"}).status_code == 200


def test_history_is_causal_and_keeps_missing_hours(client):
    body = request_for(client)
    response = client.get("/v1/history", params={"target_timestamp": body["target_timestamp"], "hours": 72})
    history = response.json()["history"]
    times = pd.DatetimeIndex([row["timestamp"] for row in history])
    assert len(times) == 72
    assert times.max() == pd.Timestamp(body["target_timestamp"]) - pd.Timedelta(hours=1)
    assert ((times[1:] - times[:-1]) == pd.Timedelta(hours=1)).all()


def test_provided_history_matches_replay_and_rejects_future_or_incomplete_grid(client):
    body = request_for(client)
    hourly = pd.read_parquet(ROOT / "deploy/bundle/hourly.parquet")
    target = pd.Timestamp(body["target_timestamp"])
    raw = hourly.loc[(hourly.index >= target - pd.Timedelta(hours=336)) & (hourly.index < target), "raw_active_power"]
    history = [{"timestamp": t.isoformat(), "active_power_kw": None if pd.isna(v) else float(v)}
               for t, v in raw.items()]
    replay = client.post("/v1/predict", json=body).json()
    body["history"] = history
    custom = client.post("/v1/predict", json=body)
    assert custom.status_code == 200, custom.text
    assert custom.json()["mode"] == "provided_history"
    assert custom.json()["prediction_kw"] == pytest.approx(replay["prediction_kw"], abs=1e-6)
    body["history"][-1]["timestamp"] = target.isoformat()
    assert client.post("/v1/predict", json=body).status_code == 422
    body["history"] = history[:100]
    assert client.post("/v1/predict", json=body).status_code == 422


def test_cannot_pass_future_truth_or_unbounded_inputs(client):
    body = request_for(client)
    body["actual"] = 10
    assert client.post("/v1/predict", json=body).status_code == 422
    assert client.get("/v1/history", params={"target_timestamp": "2010-05-01T00:00:00", "hours": 999}).status_code == 422


def test_bundle_corruption_blocks_startup(tmp_path):
    manifest = {"schema_version": 1, "files": {"model.joblib": "bad-checksum"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "model.joblib").write_bytes(b"do not deserialize me")
    with pytest.raises(ValueError, match="checksum"):
        ForecastService(tmp_path)
