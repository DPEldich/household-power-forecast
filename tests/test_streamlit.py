"""Frontend tests use real HTTP and cannot load local ML artifacts."""
import os
from pathlib import Path
import subprocess
import sys
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def clear_cache():
    st.cache_data.clear()
    st.cache_resource.clear()


def test_frontend_calls_real_api(api_url, monkeypatch):
    monkeypatch.setenv("API_BASE_URL", api_url)
    monkeypatch.setenv("API_KEY", "")
    clear_cache()
    at = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=20).run()
    assert not at.exception
    assert len(at.tabs) == 6
    for model in ["Persistence", "XGBoost"]:
        at.selectbox(key="predict_model").select(model).run()
        at.button(key="live_predict").click().run()
        assert not at.exception
        assert any("XÁC MINH THÀNH CÔNG" in item.value for item in at.success)
    at.checkbox[0].check().run()
    assert not at.exception
    assert any("Giá trị thực tế" in item.value for item in at.markdown)
    clear_cache()


def test_backend_unavailable_is_clear_error(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:1")
    clear_cache()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    assert not at.exception
    assert any("Không kết nối được FastAPI" in item.value for item in at.error)
    assert len(at.tabs) == 0
    clear_cache()


def test_fresh_frontend_process_uses_no_ml_libraries(api_url):
    code = f'''
import sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file({str(ROOT / 'frontend/app.py')!r}, default_timeout=20).run()
assert not at.exception
assert len(at.tabs) == 6
for name in ['Persistence', 'XGBoost']:
    at.selectbox(key='predict_model').select(name).run()
    at.button(key='live_predict').click().run()
    assert not at.exception
    assert any('XÁC MINH THÀNH CÔNG' in item.value for item in at.success)
assert 'joblib' not in sys.modules
assert 'xgboost' not in sys.modules
assert 'sklearn' not in sys.modules
print('HTTP_FRONTEND_OK')
'''
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(API_BASE_URL=api_url, API_KEY="", PYTHONUTF8="1")
    process = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                             capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "HTTP_FRONTEND_OK" in process.stdout
