"""Read-only, versioned inference service; training remains an offline operation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import build_features


def records(frame: pd.DataFrame) -> list[dict]:
    """Use real JSON null for missing measurements, never NaN/Infinity."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


class ForecastService:
    def __init__(self, bundle_dir: Path):
        self.root = bundle_dir.resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported serving bundle schema")
        for relative, expected_hash in self.manifest["files"].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Bundle paths must stay within the bundle directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Serving bundle checksum mismatch: {relative}")
        self.version = self.manifest["model_version"]
        self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.tables = {
            name: pd.read_csv(self.root / f"metrics/{filename}")
            for name, filename in {
                "test": "test_metrics.csv", "val": "validation_metrics.csv",
                "cv": "cv_metrics.csv", "ablation": "ablation_metrics.csv",
                "high_load": "high_load_metrics.csv",
            }.items()
        }
        self.predictions = pd.read_csv(self.root / "test_predictions.csv", parse_dates=["timestamp"])
        self.hourly = pd.read_parquet(self.root / "hourly.parquet")
        self.features, _, _ = build_features(self.hourly)
        if self.features.columns.tolist() != self.manifest["feature_names"]:
            raise ValueError("Feature contract does not match the deployed model")
        self.model_names = self.tables["test"]["model_name"].tolist()
        if set(self.model_names) != set(self.manifest["models"]):
            raise ValueError("Model registry differs from evaluation artifacts")
        self.models = {}
        for name, relative in self.manifest["models"].items():
            if relative not in self.manifest["files"]:
                raise ValueError("Every model must be checksum-verified before loading")
            self.models[name] = joblib.load(self.root / relative)
        if self.metadata["validation_selected_winner"] not in self.models:
            raise ValueError("Validation-selected model is missing")

    def dashboard(self) -> dict:
        return {
            "model_version": self.version, "schema_version": 1, "unit": "kW",
            "metadata": self.metadata,
            "metrics": {name: records(table) for name, table in self.tables.items()},
            "predictions": records(self.predictions),
        }

    def history(self, target: pd.Timestamp, hours: int) -> list[dict]:
        if target not in pd.DatetimeIndex(self.predictions["timestamp"]):
            raise KeyError("Target hour is outside the published replay dataset")
        mask = (self.hourly.index >= target - pd.Timedelta(hours=hours)) & (self.hourly.index < target)
        frame = self.hourly.loc[mask, ["raw_active_power"]].rename_axis("timestamp").reset_index()
        return records(frame)

    def predict(self, model: str, target: pd.Timestamp, history: list[dict] | None) -> dict:
        if model not in self.models:
            raise KeyError("Unknown model")
        stored_prediction = None
        if history is None:
            selected = self.predictions[self.predictions["timestamp"] == target]
            if selected.empty:
                raise KeyError("Target hour is outside the published replay dataset")
            sample = self.features.loc[[target]]
            stored_prediction = float(selected.iloc[0][model])
        else:
            timestamps = pd.DatetimeIndex([row["timestamp"] for row in history])
            if timestamps.tz is not None:
                raise ValueError("Use naive local hourly timestamps consistently with the UCI model")
            if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
                raise ValueError("History must have unique, increasing timestamps")
            if len(timestamps) < 168 or timestamps[-1] != target - pd.Timedelta(hours=1):
                raise ValueError("Provide at least 168 hours, ending exactly one hour before the target")
            if not ((timestamps[1:] - timestamps[:-1]) == pd.Timedelta(hours=1)).all():
                raise ValueError("History must contain a continuous hourly grid; use null for missing readings")
            values = pd.Series([row["active_power_kw"] for row in history], index=timestamps, dtype=float)
            if values.notna().sum() == 0:
                raise ValueError("History must contain observed power readings")
            # Append a missing target only to reuse the exact causal training feature builder.
            index = timestamps.append(pd.DatetimeIndex([target]))
            extended = values.reindex(index)
            frame = pd.DataFrame({
                "active_power_ffill3": extended.ffill(limit=3),
                "target": np.nan,
            }, index=index)
            features, _, _ = build_features(frame)
            sample = features.loc[[target], self.features.columns]
            if model in ("Persistence", "Seasonal Naive") and sample[self.models[model].lag_col].isna().any():
                raise ValueError("Baseline lookback is unavailable after bounded forward fill")
        prediction = float(self.models[model].predict(sample)[0])
        if not np.isfinite(prediction):
            raise ValueError("Model produced a non-finite prediction")
        return {
            "model_version": self.version, "model": model,
            "target_timestamp": target.isoformat(), "prediction_kw": prediction,
            "stored_prediction_kw": stored_prediction, "unit": "kW",
            "mode": "historical_replay" if history is None else "provided_history",
        }
