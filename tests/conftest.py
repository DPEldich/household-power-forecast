"""Run real HTTP for frontend integration tests, isolated from user servers."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def api_url(tmp_path_factory):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.pop("HOUSEHOLD_POWER_API_KEY", None)
    env["HOUSEHOLD_POWER_BUNDLE_DIR"] = str(ROOT / "deploy/bundle")
    log_path = tmp_path_factory.mktemp("api") / "uvicorn.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=env, stdout=log, stderr=log,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                if process.poll() is not None:
                    pytest.fail(log_path.read_text(encoding="utf-8"))
                try:
                    if requests.get(url + "/health", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail("FastAPI did not become ready: " + log_path.read_text(encoding="utf-8"))
            yield url
        finally:
            process.terminate()
            process.wait(timeout=10)
