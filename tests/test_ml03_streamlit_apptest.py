"""Streamlit AppTest integration tests for ML-03 interval prediction and legacy bundle compatibility."""

from __future__ import annotations

import os
from pathlib import Path
import streamlit as st
from streamlit.testing.v1 import AppTest

import household_power.api_client as client_module

ROOT = Path(__file__).resolve().parents[1]


def clear_cache():
    st.cache_data.clear()
    st.cache_resource.clear()


def test_frontend_live_interval_predict_with_real_api(api_url, monkeypatch):
    """Test frontend interacts with real FastAPI backend for live interval prediction."""
    monkeypatch.setenv("API_BASE_URL", api_url)
    monkeypatch.setenv("API_KEY", "")
    clear_cache()

    at = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=25).run()
    assert not at.exception
    assert len(at.tabs) == 6

    # Test live interval predict button
    assert at.button(key="live_predict_interval")
    at.button(key="live_predict_interval").click().run()
    assert not at.exception
    assert any("XÁC MINH THÀNH CÔNG" in item.value for item in at.success)
    assert any("Khoảng dự báo P10–P90 (danh định 80%)" in item.value for item in at.markdown)

    # Test toggle for uncertainty shading
    assert at.toggle(key="show_p10_p90")
    at.toggle(key="show_p10_p90").set_value(True).run()
    assert not at.exception

    clear_cache()


def test_frontend_live_interval_predict_version_mismatch(api_url, monkeypatch):
    """Test version mismatch in interval prediction response triggers error and no success."""
    monkeypatch.setenv("API_BASE_URL", api_url)
    monkeypatch.setenv("API_KEY", "")
    clear_cache()

    orig_request_json = client_module.request_json

    def mock_request_json(base_url, path, *args, **kwargs):
        if path == "/v1/predict_interval":
            real_res = orig_request_json(base_url, path, *args, **kwargs)
            tampered = dict(real_res)
            tampered["model_version"] = "mismatched-version-v999"
            return tampered
        return orig_request_json(base_url, path, *args, **kwargs)

    monkeypatch.setattr(client_module, "request_json", mock_request_json)

    at = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=25).run()
    assert not at.exception
    assert at.button(key="live_predict_interval")

    at.button(key="live_predict_interval").click().run()
    assert not at.exception
    assert any("LỖI PHIÊN BẢN MÔ HÌNH" in item.value for item in at.error)
    assert not any("XÁC MINH THÀNH CÔNG" in item.value for item in at.success)

    clear_cache()


def test_frontend_live_interval_predict_stored_mismatch(api_url, monkeypatch):
    """Test stored prediction mismatch > 1e-5 in interval prediction triggers error and no success."""
    monkeypatch.setenv("API_BASE_URL", api_url)
    monkeypatch.setenv("API_KEY", "")
    clear_cache()

    orig_request_json = client_module.request_json

    def mock_request_json(base_url, path, *args, **kwargs):
        if path == "/v1/predict_interval":
            real_res = orig_request_json(base_url, path, *args, **kwargs)
            tampered = dict(real_res)
            # Add 0.5 kW to point prediction, violating 1e-5 tolerance
            tampered["point_prediction_kw"] = float(tampered["point_prediction_kw"]) + 0.5
            return tampered
        return orig_request_json(base_url, path, *args, **kwargs)

    monkeypatch.setattr(client_module, "request_json", mock_request_json)

    at = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=25).run()
    assert not at.exception
    assert at.button(key="live_predict_interval")

    at.button(key="live_predict_interval").click().run()
    assert not at.exception
    assert any("PHÁT HIỆN SAI LỆCH DỰ BÁO" in item.value for item in at.error)
    assert not any("XÁC MINH THÀNH CÔNG" in item.value for item in at.success)

    clear_cache()


def test_frontend_legacy_dashboard_renders_six_tabs_without_interval_controls(api_url, monkeypatch):
    """Test that a legacy dashboard snapshot (without experiments) renders 6 tabs and no interval widgets."""
    monkeypatch.setenv("API_BASE_URL", api_url)
    monkeypatch.setenv("API_KEY", "")
    clear_cache()

    orig_request_json = client_module.request_json

    def mock_request_json(base_url, path, *args, **kwargs):
        if path == "/v1/dashboard":
            snap = orig_request_json(base_url, path, *args, **kwargs)
            legacy_snap = dict(snap)
            legacy_snap.pop("experiments", None)
            return legacy_snap
        return orig_request_json(base_url, path, *args, **kwargs)

    monkeypatch.setattr(client_module, "request_json", mock_request_json)

    at = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=25).run()
    assert not at.exception
    assert len(at.tabs) == 6
    # In legacy mode, interval prediction button and uncertainty toggle must NOT be rendered
    assert not any(getattr(b, "key", None) == "live_predict_interval" for b in at.button)
    assert not any(getattr(t, "key", None) == "show_p10_p90" for t in at.toggle)

    clear_cache()
