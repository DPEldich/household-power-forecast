"""Run with: python -m uvicorn backend.main:app --port 8000."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from household_power.api import create_app

app = create_app()
