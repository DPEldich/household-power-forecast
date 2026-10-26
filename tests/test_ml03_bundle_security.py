"""Security & Integrity tests: Verify that corrupted checksums, missing roles, or invalid feature orders fail closed."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import pytest

from household_power.serving import ForecastService

ROOT = Path(__file__).resolve().parents[1]


def test_corrupted_optional_model_checksum_fails_closed(tmp_path):
    """Verify that tampering with an optional model file causes ForecastService to fail closed."""
    bundle_src = ROOT / "deploy/bundle"
    bundle_copy = tmp_path / "bundle"
    shutil.copytree(bundle_src, bundle_copy)

    target_model = bundle_copy / "experiments/models/quantile_p10.joblib"
    if not target_model.exists():
        pytest.skip("Experimental model does not exist in bundle")

    original_bytes = target_model.read_bytes()
    tampered_bytes = bytearray(original_bytes)
    tampered_bytes[10] ^= 0xFF
    target_model.write_bytes(tampered_bytes)

    with pytest.raises(ValueError, match="checksum mismatch"):
        ForecastService(bundle_copy)


def test_unregistered_optional_model_fails_closed(tmp_path):
    """Verify that an experimental model referenced in experiments but not in manifest['files'] fails closed."""
    bundle_src = ROOT / "deploy/bundle"
    bundle_copy = tmp_path / "bundle"
    shutil.copytree(bundle_src, bundle_copy)

    manifest_file = bundle_copy / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    if "experiments" not in manifest:
        pytest.skip("No experiments in bundle")

    rel_p10 = manifest["experiments"]["models"]["p10"]
    if rel_p10 in manifest["files"]:
        del manifest["files"][rel_p10]

    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Every optional model and artifact must be checksum-verified"):
        ForecastService(bundle_copy)


def test_path_traversal_in_experiments_fails_closed(tmp_path):
    """Verify that paths attempting to escape the bundle root fail closed."""
    bundle_src = ROOT / "deploy/bundle"
    bundle_copy = tmp_path / "bundle"
    shutil.copytree(bundle_src, bundle_copy)

    manifest_file = bundle_copy / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    if "experiments" not in manifest:
        pytest.skip("No experiments in bundle")

    manifest["experiments"]["models"]["p10"] = "../../escaped_model.joblib"
    manifest["files"]["../../escaped_model.joblib"] = "fakehash"

    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="stay within bundle directory|within the bundle directory"):
        ForecastService(bundle_copy)


def test_missing_required_model_role_fails_closed(tmp_path):
    """Verify omitting required role (e.g. 'p90') fails closed before deserialization."""
    bundle_src = ROOT / "deploy/bundle"
    bundle_copy = tmp_path / "bundle"
    shutil.copytree(bundle_src, bundle_copy)

    manifest_file = bundle_copy / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if "experiments" not in manifest:
        pytest.skip("No experiments in bundle")

    del manifest["experiments"]["models"]["p90"]
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required experimental model role"):
        ForecastService(bundle_copy)


def test_missing_required_metric_key_fails_closed(tmp_path):
    """Verify omitting required metric key (e.g. 'uncertainty') fails closed before deserialization."""
    bundle_src = ROOT / "deploy/bundle"
    bundle_copy = tmp_path / "bundle"
    shutil.copytree(bundle_src, bundle_copy)

    manifest_file = bundle_copy / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if "experiments" not in manifest:
        pytest.skip("No experiments in bundle")

    del manifest["experiments"]["metrics"]["uncertainty"]
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required experimental metric key"):
        ForecastService(bundle_copy)


def test_bad_feature_order_fails_closed(tmp_path):
    """Verify reversed feature order in manifest alone fails closed against metadata."""
    bundle_src = ROOT / "deploy/bundle"
    bundle_copy = tmp_path / "bundle"
    shutil.copytree(bundle_src, bundle_copy)

    manifest_file = bundle_copy / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if "experiments" not in manifest:
        pytest.skip("No experiments in bundle")

    reversed_cols = list(reversed(manifest["experiments"]["feature_names"]))
    manifest["experiments"]["feature_names"] = reversed_cols
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="feature order mismatch|mismatch"):
        ForecastService(bundle_copy)


def test_reversed_features_in_both_manifest_and_metadata_fails_closed(tmp_path):
    """Verify reversing feature_names in BOTH manifest and metadata with valid checksum fails against pipeline model."""
    import hashlib
    bundle_src = ROOT / "deploy/bundle"
    bundle_copy = tmp_path / "bundle"
    shutil.copytree(bundle_src, bundle_copy)

    manifest_file = bundle_copy / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if "experiments" not in manifest:
        pytest.skip("No experiments in bundle")

    # Reverse in metadata
    meta_rel = manifest["experiments"]["metadata"]
    meta_path = bundle_copy / meta_rel
    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
    reversed_feats = list(reversed(meta_data["selected_feature_names"]))
    meta_data["selected_feature_names"] = reversed_feats
    meta_bytes = json.dumps(meta_data, indent=2).encode("utf-8")
    meta_path.write_bytes(meta_bytes)

    # Recompute metadata checksum in manifest
    manifest["files"][meta_rel] = hashlib.sha256(meta_bytes).hexdigest()

    # Reverse in manifest as well (so metadata and manifest agree)
    manifest["experiments"]["feature_names"] = reversed_feats
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    # Must fail closed because pipeline model.feature_names_in_ does not match reversed order!
    with pytest.raises(ValueError, match="feature_names_in_ mismatch"):
        ForecastService(bundle_copy)


def test_empty_or_duplicate_feature_names_fails_closed(tmp_path):
    """Verify empty or duplicate feature names in manifest fail closed."""
    bundle_src = ROOT / "deploy/bundle"
    bundle_copy = tmp_path / "bundle"
    shutil.copytree(bundle_src, bundle_copy)

    manifest_file = bundle_copy / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if "experiments" not in manifest:
        pytest.skip("No experiments in bundle")

    # Empty features
    manifest["experiments"]["feature_names"] = []
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        ForecastService(bundle_copy)

    # Duplicate features
    valid_feats = json.loads((bundle_copy / manifest["experiments"]["metadata"]).read_text(encoding="utf-8"))["selected_feature_names"]
    manifest["experiments"]["feature_names"] = [valid_feats[0]] + valid_feats
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unique names"):
        ForecastService(bundle_copy)


def test_legacy_bundle_api_compatibility(tmp_path):
    """Verify FastAPI create_app on legacy bundle serves baseline and returns 501 on interval predict."""
    from fastapi.testclient import TestClient
    from household_power.api import create_app

    bundle_src = ROOT / "deploy/bundle"
    bundle_copy = tmp_path / "bundle"
    shutil.copytree(bundle_src, bundle_copy)

    manifest_file = bundle_copy / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if "experiments" in manifest:
        del manifest["experiments"]
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    legacy_app = create_app(bundle_dir=bundle_copy, api_key="")
    with TestClient(legacy_app) as client:
        dash_resp = client.get("/v1/dashboard")
        assert dash_resp.status_code == 200
        dash = dash_resp.json()
        assert "experiments" not in dash
        assert len(dash["predictions"]) > 0

        target_ts = dash["predictions"][0]["timestamp"]
        version = dash["model_version"]

        # Baseline predict works
        base_resp = client.post(
            "/v1/predict",
            json={
                "model": "XGBoost",
                "target_timestamp": target_ts,
                "expected_model_version": version,
            },
        )
        assert base_resp.status_code == 200
        assert "prediction_kw" in base_resp.json()

        # Interval predict returns HTTP 501 Not Implemented
        int_resp = client.post(
            "/v1/predict_interval",
            json={
                "target_timestamp": target_ts,
                "expected_model_version": version,
            },
        )
        assert int_resp.status_code == 501
        assert "not available" in int_resp.json()["detail"]


def test_legacy_bundle_without_experiments_works_and_rejects_predict_interval(tmp_path):
    """Verify legacy bundle with NO experiments works for 7 models and rejects predict_interval cleanly."""
    bundle_src = ROOT / "deploy/bundle"
    bundle_copy = tmp_path / "bundle"
    shutil.copytree(bundle_src, bundle_copy)

    manifest_file = bundle_copy / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if "experiments" in manifest:
        del manifest["experiments"]
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    legacy_service = ForecastService(bundle_copy)
    assert not legacy_service.has_experiments
    dash = legacy_service.dashboard()
    assert "experiments" not in dash
    assert len(legacy_service.model_names) == 7

    target_ts = legacy_service.predictions["timestamp"].iloc[0]
    pred = legacy_service.predict("XGBoost", target_ts, None)
    assert pred["prediction_kw"] is not None

    with pytest.raises(ValueError, match="not available"):
        legacy_service.predict_interval(target_ts, None)
