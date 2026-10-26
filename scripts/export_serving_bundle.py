"""Export a small, checksum-verified serving release; never retrain during deployment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from household_power.features import build_features
from household_power.serving import ForecastService
from household_power.uncertainty import repair_and_clip_quantiles


def export_bundle(
    destination: Path = ROOT / "deploy/bundle",
    include_experiments: bool = True,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)

    # 1. Baseline sources
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

    # 2. Optional ML-03 experimental extension sources
    exp_dir = ROOT / "artifacts/experimental/ml03"
    has_exp = include_experiments and (exp_dir / "metadata/experiment_metadata.json").exists()

    exp_models_map = {}
    exp_metrics_map = {}
    exp_meta = None
    if has_exp:
        exp_meta_path = exp_dir / "metadata/experiment_metadata.json"
        exp_meta = json.loads(exp_meta_path.read_text(encoding="utf-8"))
        sources["experiments/metadata.json"] = exp_meta_path
        sources["experiments/test_predictions.csv"] = exp_dir / "predictions/experimental_test_predictions.csv"

        # Audit files
        if (exp_dir / "metadata/protocol.json").exists():
            sources["experiments/protocol.json"] = exp_dir / "metadata/protocol.json"
        if (exp_dir / "metadata/selection_frozen.json").exists():
            sources["experiments/selection_frozen.json"] = exp_dir / "metadata/selection_frozen.json"
        if (exp_dir / "predictions/experimental_val_predictions.csv").exists():
            sources["experiments/val_predictions.csv"] = exp_dir / "predictions/experimental_val_predictions.csv"

        # Metric files mapping
        metric_files = {
            "cv": "cv_candidate_metrics.csv",
            "val": "validation_candidate_metrics.csv",
            "test_point": "test_point_comparison.csv",
            "uncertainty": "uncertainty_metrics.csv",
            "diagnostics_hour": "diagnostics_by_hour.csv",
            "diagnostics_month": "diagnostics_by_month.csv",
            "diagnostics_change": "diagnostics_by_load_change.csv",
        }
        for key, filename in metric_files.items():
            file_path = exp_dir / "metrics" / filename
            if file_path.exists():
                rel_path = f"experiments/metrics/{filename}"
                sources[rel_path] = file_path
                exp_metrics_map[key] = rel_path

        # Models mapping
        for role, filename in [
            ("point", "selected_point_model.joblib"),
            ("p10", "quantile_p10.joblib"),
            ("p50", "quantile_p50.joblib"),
            ("p90", "quantile_p90.joblib"),
        ]:
            rel_path = f"experiments/models/{filename}"
            sources[rel_path] = exp_dir / f"models/{filename}"
            exp_models_map[role] = rel_path

    # Copy and compute SHA256 hashes
    hashes = {}
    for relative, source in sources.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()

    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    manifest: dict = {
        "schema_version": 1,
        "model_version": f"uci-hourly-{digest}",
        "dataset": "UCI-235",
        "unit": "kW",
        "forecast_horizon_hours": 1,
        "timestamp_convention": "naive-local-as-recorded-in-UCI",
        "feature_names": build_features(pd.read_parquet(sources["hourly.parquet"]))[0].columns.tolist(),
        "models": models,
        "files": hashes,
    }

    if has_exp and exp_meta is not None:
        manifest["experiments"] = {
            "schema_version": 2,
            "feature_names": exp_meta["selected_feature_names"],
            "models": exp_models_map,
            "metadata": "experiments/metadata.json",
            "predictions": "experiments/test_predictions.csv",
            "metrics": exp_metrics_map,
        }

    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")

    # 3. Instantiate ForecastService and verify all baseline and experimental models before release
    service = ForecastService(destination)
    for name in model_names:
        frame = service.predictions
        predicted = service.models[name].predict(service.features.loc[pd.DatetimeIndex(frame.timestamp)])
        np.testing.assert_allclose(predicted, frame[name], atol=1e-6, rtol=1e-6)

    if service.has_experiments:
        exp_frame = service.exp_predictions
        exp_ts = pd.DatetimeIndex(exp_frame["timestamp"])
        exp_feat_cols = service.manifest["experiments"]["feature_names"]
        sample = service.extended_features.loc[exp_ts, exp_feat_cols]

        pred_pt = service.exp_models["point"].predict(sample)
        np.testing.assert_allclose(pred_pt, exp_frame["point_pred"], atol=1e-5, rtol=1e-5)

        r10 = service.exp_models["p10"].predict(sample)
        r50 = service.exp_models["p50"].predict(sample)
        r90 = service.exp_models["p90"].predict(sample)
        p10, p50, p90 = repair_and_clip_quantiles(r10, r50, r90)
        np.testing.assert_allclose(p10, exp_frame["p10"], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(p50, exp_frame["p50"], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(p90, exp_frame["p90"], atol=1e-5, rtol=1e-5)

    print(
        f"Exported {manifest['model_version']}: {len(model_names)} baseline models verified, "
        f"{'experimental v2 models verified, ' if service.has_experiments else ''}"
        f"{len(service.predictions)} test hours"
    )
    return manifest


if __name__ == "__main__":
    export_bundle()
