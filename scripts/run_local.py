"""Start both services locally; stop only the child processes owned by this launcher."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=8501)
    args = parser.parse_args()
    for port in [args.api_port, args.frontend_port]:
        if not 1024 <= port <= 65535:
            parser.error("Ports must be between 1024 and 65535")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                parser.error(f"Port {port} is already in use; stop the old service or choose another port")
    if args.api_port == args.frontend_port:
        parser.error("Backend and frontend need different ports")
    api_url = f"http://127.0.0.1:{args.api_port}"
    children = []
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        children.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(args.api_port)],
            cwd=ROOT, creationflags=creation_flags,
        ))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if children[0].poll() is not None:
                raise RuntimeError("FastAPI failed to start. Verify deploy/bundle and backend requirements.")
            try:
                with urlopen(api_url + "/health", timeout=1) as response:
                    if json.load(response)["status"] == "ok":
                        break
            except (URLError, TimeoutError):
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("FastAPI did not become ready within 30 seconds")
        env = os.environ.copy()
        env.update(API_BASE_URL=api_url, API_KEY=env.get("HOUSEHOLD_POWER_API_KEY", ""))
        children.append(subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.headless", "true",
             "--server.address", "127.0.0.1", "--server.port", str(args.frontend_port),
             "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env, creationflags=creation_flags,
        ))
        print(f"Frontend: http://127.0.0.1:{args.frontend_port} | API docs: {api_url}/docs", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        raise RuntimeError("A service exited; stopping its companion service")
    except KeyboardInterrupt:
        pass
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)


if __name__ == "__main__":
    main()
