"""Read-only, versioned inference service; training remains an offline operation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .features import build_extended_features, build_features
from .uncertainty import repair_and_clip_quantiles


def records(frame: pd.DataFrame) -> list[dict]:
    """Use real JSON null for missing measurements, never NaN/Infinity."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


class ForecastService:
    def __init__(self, bundle_dir: Path):
        self.root = bundle_dir.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Bundle manifest.json is missing")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported serving bundle schema")

        # 1. Verify all checksummed files first
        for relative, expected_hash in self.manifest["files"].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError(f"Bundle paths must stay within the bundle directory: {relative}")
            if not path.exists():
                raise FileNotFoundError(f"Missing required bundle file: {relative}")
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

        # 2. Check for optional v2 experimental extensions
        if "experiments" in self.manifest:
            exp_conf = self.manifest["experiments"]
            if exp_conf.get("schema_version") != 2:
                raise ValueError("Unsupported experimental schema version in bundle")

            # Validate exact required model roles and metric keys BEFORE deserialization
            required_roles = ["point", "p10", "p50", "p90"]
            models_dict = exp_conf.get("models", {})
            for r in required_roles:
                if r not in models_dict or not models_dict[r]:
                    raise ValueError(f"Missing required experimental model role in manifest: '{r}'")

            required_metrics = ["cv", "val", "test_point", "uncertainty"]
            metrics_dict = exp_conf.get("metrics", {})
            for m in required_metrics:
                if m not in metrics_dict or not metrics_dict[m]:
                    raise ValueError(f"Missing required experimental metric key in manifest: '{m}'")

            if not exp_conf.get("predictions"):
                raise ValueError("Missing experimental predictions file path in manifest")
            if not exp_conf.get("metadata"):
                raise ValueError("Missing experimental metadata file path in manifest")

            # Check that all referenced files are in checksummed manifest["files"] and confined to root
            all_exp_rel_paths = (
                list(models_dict.values())
                + list(metrics_dict.values())
                + [exp_conf["predictions"], exp_conf["metadata"]]
            )
            for rel in all_exp_rel_paths:
                p = (self.root / rel).resolve()
                if not p.is_relative_to(self.root):
                    raise ValueError(f"Experimental path must stay within bundle directory: {rel}")
                if rel not in self.manifest["files"]:
                    raise ValueError(f"Every optional model and artifact must be checksum-verified before loading: {rel}")

            self.exp_metadata = json.loads((self.root / exp_conf["metadata"]).read_text(encoding="utf-8"))
            self.exp_predictions = pd.read_csv(self.root / exp_conf["predictions"], parse_dates=["timestamp"])
            self.exp_tables = {k: pd.read_csv(self.root / rel) for k, rel in metrics_dict.items()}

            # Feature order contract verification
            exp_feat_cols = exp_conf.get("feature_names", [])
            if not exp_feat_cols:
                raise ValueError("Experimental feature_names must be non-empty")
            if len(exp_feat_cols) != len(set(exp_feat_cols)):
                raise ValueError("Experimental feature_names must contain unique names")
            meta_feat_cols = self.exp_metadata.get("selected_feature_names", [])
            if exp_feat_cols != meta_feat_cols:
                raise ValueError(
                    f"Experimental feature order mismatch: manifest {exp_feat_cols} != metadata {meta_feat_cols}"
                )

            # Precompute extended features and check contract
            self.extended_features, _, _ = build_extended_features(self.hourly)
            if not set(exp_feat_cols).issubset(self.extended_features.columns):
                raise ValueError("Experimental feature names mismatch against extended feature generator")

            # Deserialization of verified optional models
            self.exp_models = {k: joblib.load(self.root / rel) for k, rel in models_dict.items()}

            # Verify each model estimator's feature contract against manifest feature order
            for role, model in self.exp_models.items():
                if hasattr(model, "feature_names_in_"):
                    est_feats = list(model.feature_names_in_)
                    if est_feats != exp_feat_cols:
                        raise ValueError(
                            f"Model '{role}' feature_names_in_ mismatch: {est_feats} != {exp_feat_cols}"
                        )
                elif hasattr(model, "n_features_in_"):
                    if model.n_features_in_ != len(exp_feat_cols):
                        raise ValueError(
                            f"Model '{role}' n_features_in_ mismatch: {model.n_features_in_} != {len(exp_feat_cols)}"
                        )
                else:
                    raise ValueError(
                        f"Model '{role}' does not expose feature contract attributes (feature_names_in_ or n_features_in_)"
                    )

            self.has_experiments = True
        else:
            self.has_experiments = False
            self.exp_metadata = None
            self.exp_predictions = None
            self.exp_tables = {}
            self.exp_models = {}
            self.extended_features = None

    def dashboard(self) -> dict:
        data = {
            "model_version": self.version, "schema_version": 1, "unit": "kW",
            "metadata": self.metadata,
            "metrics": {name: records(table) for name, table in self.tables.items()},
            "predictions": records(self.predictions),
        }
        if self.has_experiments:
            data["experiments"] = {
                "schema_version": 2,
                "metadata": self.exp_metadata,
                "metrics": {name: records(table) for name, table in self.exp_tables.items()},
                "predictions": records(self.exp_predictions),
                "nominal_coverage_pct": 80.0,
            }
        return data

    def history(self, target: pd.Timestamp, hours: int) -> list[dict]:
        if target not in pd.DatetimeIndex(self.predictions["timestamp"]):
            raise KeyError("Target hour is outside the published replay dataset")
        mask = (self.hourly.index >= target - pd.Timedelta(hours=hours)) & (self.hourly.index < target)
        frame = self.hourly.loc[mask, ["raw_active_power"]].rename_axis("timestamp").reset_index()
        return records(frame)

    def _prepare_history_frame(self, target: pd.Timestamp, history: list[dict]) -> pd.DataFrame:
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
        index = timestamps.append(pd.DatetimeIndex([target]))
        extended = values.reindex(index)
        return pd.DataFrame({
            "active_power_ffill3": extended.ffill(limit=3),
            "target": np.nan,
        }, index=index)

    def predict(self, model: str, target: pd.Timestamp, history: list[dict] | None) -> dict:
        if model not in self.models:
            raise KeyError(f"Unknown model: {model}")
        stored_prediction = None
        if history is None:
            selected = self.predictions[self.predictions["timestamp"] == target]
            if selected.empty:
                raise KeyError("Target hour is outside the published replay dataset")
            sample = self.features.loc[[target]]
            stored_prediction = float(selected.iloc[0][model])
        else:
            frame = self._prepare_history_frame(target, history)
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

    def predict_interval(self, target: pd.Timestamp, history: list[dict] | None = None) -> dict:
        """Generate point prediction alongside P10, P50, P90 uncertainty interval."""
        if not self.has_experiments:
            raise ValueError("Experimental interval forecasting is not available in the current bundle")

        feat_cols = self.manifest["experiments"]["feature_names"]
        stored_point = None
        stored_p10 = None
        stored_p50 = None
        stored_p90 = None

        if history is None:
            selected = self.exp_predictions[self.exp_predictions["timestamp"] == target]
            if selected.empty:
                raise KeyError("Target hour is outside the published replay dataset")
            sample = self.extended_features.loc[[target], feat_cols]
            row = selected.iloc[0]
            stored_point = float(row["point_pred"])
            stored_p10 = float(row["p10"])
            stored_p50 = float(row["p50"])
            stored_p90 = float(row["p90"])
        else:
            frame = self._prepare_history_frame(target, history)
            ext_feat, _, _ = build_extended_features(frame)
            sample = ext_feat.loc[[target], feat_cols]

        point_pred = float(self.exp_models["point"].predict(sample)[0])
        raw_p10 = float(self.exp_models["p10"].predict(sample)[0])
        raw_p50 = float(self.exp_models["p50"].predict(sample)[0])
        raw_p90 = float(self.exp_models["p90"].predict(sample)[0])

        p10_arr, p50_arr, p90_arr = repair_and_clip_quantiles([raw_p10], [raw_p50], [raw_p90])
        p10 = float(p10_arr[0])
        p50 = float(p50_arr[0])
        p90 = float(p90_arr[0])

        for val in [point_pred, p10, p50, p90]:
            if not np.isfinite(val):
                raise ValueError("Model produced a non-finite prediction")

        point_model_name = self.exp_metadata.get(
            "promoted_point_model",
            self.exp_metadata.get("validation_confirmation", {}).get("promoted_point_model", "baseline")
        )

        return {
            "model_version": self.version,
            "target_timestamp": target.isoformat(),
            "point_model": f"{point_model_name} (ML-03)",
            "point_prediction_kw": point_pred,
            "p10_kw": p10,
            "p50_kw": p50,
            "p90_kw": p90,
            "nominal_coverage_pct": 80.0,
            "stored_point_prediction_kw": stored_point,
            "stored_p10_kw": stored_p10,
            "stored_p50_kw": stored_p50,
            "stored_p90_kw": stored_p90,
            "unit": "kW",
            "mode": "historical_replay" if history is None else "provided_history",
        }
