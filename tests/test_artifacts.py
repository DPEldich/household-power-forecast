"""Tests for model persistence and artifact roundtrip."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from household_power.models import get_model_pipeline


def test_saved_model_roundtrip(tmp_path: Path):
    """Verify that a fitted model pipeline can be saved to joblib and restored with identical predictions."""
    np.random.seed(42)
    X = pd.DataFrame({
        "lag_1": np.random.uniform(0.5, 3.0, 50),
        "hour": np.random.randint(0, 24, 50),
    })
    y = X["lag_1"] * 0.8 + 0.2

    pipe = get_model_pipeline("Ridge", alpha=1.0)
    pipe.fit(X, y)
    preds_orig = pipe.predict(X)

    # Save to disk
    model_file = tmp_path / "ridge_test.joblib"
    joblib.dump(pipe, model_file)

    # Reload
    loaded_pipe = joblib.load(model_file)
    preds_loaded = loaded_pipe.predict(X)

    np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-6, atol=1e-6)
