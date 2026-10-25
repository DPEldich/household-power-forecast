"""Export a small, checksum-verified serving release; never retrain during deployment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from household_power.features import build_features
from household_power.serving import ForecastService


def export_bundle(destination: Path = ROOT / "deploy/bundle") -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    sources = {
        "metadata.json": ROOT / "artifacts/metadata/split_metadata.json",
        "test_predictions.csv": ROOT / "artifacts/predictions/test_predictions.csv",
        "hourly.parquet": ROOT / "data/processed/hourly_power.parquet",
    }
    for path in (ROOT / "artifacts/metrics").glob("*.csv"):
        sources[f"metrics/{path.name}"] = path
    model_names = pd.read_csv(ROOT / "artifacts/metrics/test_metrics.csv")["model_name"].tolist()
    models = {name: f"models/{name.lower().replace(' ', '_')}.joblib" for name in model_names}
    for relative in models.values():
        sources[relative] = ROOT / "artifacts" / relative
    hashes = {}
    for relative, source in sources.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {
        "schema_version": 1, "model_version": f"uci-hourly-{digest}",
        "dataset": "UCI-235", "unit": "kW", "forecast_horizon_hours": 1,
        "timestamp_convention": "naive-local-as-recorded-in-UCI",
        "feature_names": build_features(pd.read_parquet(sources["hourly.parquet"]))[0].columns.tolist(),
        "models": models, "files": hashes,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    service = ForecastService(destination)
    # Verify the release against all stored test predictions before it can be published.
    for name in model_names:
        import numpy as np
        frame = service.predictions
        predicted = service.models[name].predict(service.features.loc[pd.DatetimeIndex(frame.timestamp)])
        np.testing.assert_allclose(predicted, frame[name], atol=1e-6, rtol=1e-6)
    print(f"Exported {manifest['model_version']}: {len(model_names)} verified models, {len(service.predictions)} test hours")
    return manifest


if __name__ == "__main__":
    export_bundle()
