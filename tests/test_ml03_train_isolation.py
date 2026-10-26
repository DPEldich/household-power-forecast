"""Tests verifying threshold calculation, preprocessing isolation, and runner lifecycle strictly on training sets."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from household_power.experiments import (
    ExperimentRunner,
    compute_high_load_threshold,
)
from household_power.features import build_features
from household_power.splits import compute_chronological_splits, get_expanding_cv_folds


def test_runner_train_isolation_and_frozen_selection_lifecycle(tmp_path, monkeypatch):
    """Integration test verifying threshold helper receives train indices only and selection is frozen before test alignment."""
    # 1. Create synthetic 600-hour continuous dataset
    timestamps = pd.date_range("2007-01-01 00:00:00", periods=600, freq="1h")
    t = np.arange(len(timestamps))
    power = 1.0 + 0.5 * np.sin(2 * np.pi * t / 24.0) + 0.1 * np.cos(2 * np.pi * t / 168.0)
    synthetic_hourly = pd.DataFrame(
        {
            "active_power_ffill3": power,
            "target": power,
            "missing_hourly_ratio": 0.0,
        },
        index=timestamps,
    )

    # 2. Setup mock directories
    raw_dir = tmp_path / "data/raw"
    processed_dir = tmp_path / "data/processed"
    baseline_dir = tmp_path / "artifacts"
    exp_dir = tmp_path / "artifacts/experimental/ml03"

    for d in [raw_dir, processed_dir, baseline_dir / "metadata", baseline_dir / "predictions", exp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Compute expected splits for verification
    X_base, y_all, valid_mask = build_features(synthetic_hourly)
    splits_meta, tr_m, va_m, te_m, _ = compute_chronological_splits(synthetic_hourly, valid_mask)
    cv_folds = get_expanding_cv_folds(synthetic_hourly, valid_mask, splits_meta.train_boundary, n_folds=3)

    test_indices = y_all.index[te_m]
    test_actuals = y_all.loc[test_indices].to_numpy()

    # Dummy baseline split_metadata.json
    base_meta = {
        "best_hyperparameters": {
            "XGBoost": {
                "n_estimators": 2,
                "learning_rate": 0.1,
                "max_depth": 2,
                "subsample": 1.0,
            }
        }
    }
    (baseline_dir / "metadata/split_metadata.json").write_text(json.dumps(base_meta), encoding="utf-8")

    # Dummy baseline test_predictions.csv
    base_preds_df = pd.DataFrame(
        {
            "timestamp": test_indices,
            "actual": test_actuals,
            "XGBoost": test_actuals * 0.95,
        }
    )
    base_preds_df.to_csv(baseline_dir / "predictions/test_predictions.csv", index=False)

    # 3. Monkeypatch prepare_and_cache_hourly_data
    monkeypatch.setattr(
        "household_power.experiments.prepare_and_cache_hourly_data",
        lambda **kwargs: (synthetic_hourly, {}),
    )

    # 4. Spy on compute_high_load_threshold
    import household_power.experiments as exp_module
    orig_threshold_helper = exp_module.compute_high_load_threshold
    threshold_call_indices = []

    def spy_threshold(y_train, percentile=90.0):
        idx = y_train.index if isinstance(y_train, pd.Series) else None
        threshold_call_indices.append(idx)
        return orig_threshold_helper(y_train, percentile)

    monkeypatch.setattr(exp_module, "compute_high_load_threshold", spy_threshold)

    # 5. Spy on align_baseline_test_predictions to verify selection_frozen.json exists
    orig_align_helper = exp_module.align_baseline_test_predictions
    frozen_exists_before_align = []

    def spy_align(baseline_csv, test_timestamps, expected_actual):
        frozen_file = exp_dir / "metadata/selection_frozen.json"
        frozen_exists_before_align.append(frozen_file.exists())
        return orig_align_helper(baseline_csv, test_timestamps, expected_actual)

    monkeypatch.setattr(exp_module, "align_baseline_test_predictions", spy_align)

    # 6. Execute runner with tiny XGB config
    runner = ExperimentRunner(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        baseline_artifacts_dir=baseline_dir,
        experimental_dir=exp_dir,
        fixed_xgb_config={"n_estimators": 2, "max_depth": 2, "n_jobs": 1},
    )
    res = runner.run()

    # 7. Assertions
    assert len(threshold_call_indices) == 4
    val_set_idx = set(y_all.index[va_m])
    test_set_idx = set(test_indices)

    for i in range(3):
        fold_train_idx = set(y_all.index[cv_folds[i][0]])
        called_idx = set(threshold_call_indices[i])
        assert called_idx == fold_train_idx
        # Strictly disjoint from val and test
        assert len(called_idx.intersection(val_set_idx)) == 0
        assert len(called_idx.intersection(test_set_idx)) == 0

    # Call 4: full train confirmation
    full_train_idx = set(y_all.index[tr_m])
    called_idx_4 = set(threshold_call_indices[3])
    assert called_idx_4 == full_train_idx
    assert len(called_idx_4.intersection(val_set_idx)) == 0
    assert len(called_idx_4.intersection(test_set_idx)) == 0

    # Assert selection_frozen.json was created before align helper was called
    assert len(frozen_exists_before_align) == 1
    assert frozen_exists_before_align[0] is True

    # Assert protocol.json contains rich CV fold boundaries and policies
    proto = json.loads((exp_dir / "metadata/protocol.json").read_text(encoding="utf-8"))
    assert "cv_folds" in proto
    assert len(proto["cv_folds"]) == 3
    assert "raw_objectives" in proto
    assert "preprocessing_policy" in proto

    # Assert experimental_val_predictions.csv contains enhanced_features_pred
    val_preds = pd.read_csv(exp_dir / "predictions/experimental_val_predictions.csv")
    assert "enhanced_features_pred" in val_preds.columns
    assert "baseline_pred" in val_preds.columns
    assert "point_pred" in val_preds.columns


def test_imputer_fitted_strictly_on_train():
    """Verify that Pipeline SimpleImputer learns medians from train only."""
    runner = ExperimentRunner()
    pipe = runner._get_point_pipeline()

    # Train feature matrix where col 0 median is 10.0
    X_train = pd.DataFrame({
        "f1": [10.0, 10.0, 10.0, 10.0],
        "f2": [1.0, 2.0, 3.0, 4.0],
    })
    y_train = pd.Series([1.0, 2.0, 3.0, 4.0])

    pipe.fit(X_train, y_train)

    imputer = pipe.named_steps["imputer"]
    assert imputer.statistics_[0] == 10.0

    # Validation feature matrix with NaN in f1
    X_val = pd.DataFrame({
        "f1": [np.nan],
        "f2": [2.5],
    })
    # Transform using fitted imputer: must fill with train median 10.0
    imputed_val = imputer.transform(X_val)
    assert imputed_val[0, 0] == 10.0
