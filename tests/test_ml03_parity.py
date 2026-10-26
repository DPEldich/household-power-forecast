"""Parity test: Verify all rows in experimental_test_predictions.csv match model .predict() exactly."""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from household_power.features import build_extended_features
from household_power.uncertainty import repair_and_clip_quantiles

ROOT = Path(__file__).resolve().parents[1]


def test_saved_experimental_models_match_predictions_csv_all_rows():
    """Verify point and quantile predictions match saved CSV across all test rows."""
    bundle_dir = ROOT / "deploy/bundle"
    if not (bundle_dir / "experiments/test_predictions.csv").exists():
        pytest.skip("Serving bundle does not contain experimental predictions yet")

    pred_df = pd.read_csv(bundle_dir / "experiments/test_predictions.csv", parse_dates=["timestamp"])
    hourly = pd.read_parquet(bundle_dir / "hourly.parquet")

    ext_features, _, _ = build_extended_features(hourly)

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    exp_conf = manifest["experiments"]
    feat_cols = exp_conf["feature_names"]

    test_ts = pd.DatetimeIndex(pred_df["timestamp"])
    sample = ext_features.loc[test_ts, feat_cols]

    # 1. Point model parity
    point_model = joblib.load(bundle_dir / exp_conf["models"]["point"])
    pred_pt = point_model.predict(sample)
    np.testing.assert_allclose(pred_pt, pred_df["point_pred"], atol=1e-5, rtol=1e-5)

    # 2. Quantile models parity
    m10 = joblib.load(bundle_dir / exp_conf["models"]["p10"])
    m50 = joblib.load(bundle_dir / exp_conf["models"]["p50"])
    m90 = joblib.load(bundle_dir / exp_conf["models"]["p90"])

    r10 = m10.predict(sample)
    r50 = m50.predict(sample)
    r90 = m90.predict(sample)

    p10, p50, p90 = repair_and_clip_quantiles(r10, r50, r90)
    np.testing.assert_allclose(p10, pred_df["p10"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(p50, pred_df["p50"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(p90, pred_df["p90"], atol=1e-5, rtol=1e-5)
