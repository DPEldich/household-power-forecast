"""Tests to verify zero data leakage during preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from household_power.models import get_model_pipeline


def test_preprocessing_fitted_on_train_only():
    """Verify that imputer and scaler learn statistics strictly from train data and remain unchanged."""
    np.random.seed(42)
    # Train data: mean ~ 100, std ~ 10, with some NaNs
    X_train = pd.DataFrame({
        "lag_1": np.random.normal(100, 10, 200),
        "lag_2": np.random.normal(100, 10, 200),
    })
    X_train.iloc[0, 0] = np.nan
    y_train = pd.Series(np.random.normal(100, 10, 200))

    # Test data with completely different scale: mean ~ 1000, std ~ 50
    X_test = pd.DataFrame({
        "lag_1": np.random.normal(1000, 50, 100),
        "lag_2": np.random.normal(1000, 50, 100),
    })

    pipe = get_model_pipeline("Ridge", alpha=1.0)
    pipe.fit(X_train, y_train)

    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]

    # Imputer statistics should match training median (~ 100)
    train_median_0 = np.nanmedian(X_train["lag_1"])
    assert np.isclose(imputer.statistics_[0], train_median_0, atol=1e-3)

    # Scaler mean should match training mean (~ 100)
    assert 90 < scaler.mean_[0] < 110

    # Store learned scaler mean
    stored_mean = scaler.mean_.copy()

    # Call predict on test data (which has mean ~ 1000)
    _ = pipe.predict(X_test)

    # The scaler and imputer internal parameters MUST NOT have mutated after transforming test
    np.testing.assert_array_equal(scaler.mean_, stored_mean)
