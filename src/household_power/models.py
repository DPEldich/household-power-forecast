"""Model definitions, baseline implementations, and hyperparameter grids."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logger = logging.getLogger(__name__)


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Persistence baseline: predicts y_s = x_{s-1}."""

    def __init__(self, lag_col: str = "lag_1"):
        self.lag_col = lag_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.lag_col in X.columns:
                return X[self.lag_col].to_numpy()
            raise KeyError(f"Column '{self.lag_col}' not found in feature matrix.")
        raise TypeError("PersistenceBaseline requires a pandas DataFrame input.")


class SeasonalNaiveBaseline(BaseEstimator, RegressorMixin):
    """Seasonal Naive baseline: predicts y_s = x_{s-24}."""

    def __init__(self, lag_col: str = "lag_24"):
        self.lag_col = lag_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.lag_col in X.columns:
                return X[self.lag_col].to_numpy()
            raise KeyError(f"Column '{self.lag_col}' not found in feature matrix.")
        raise TypeError("SeasonalNaiveBaseline requires a pandas DataFrame input.")


def get_model_pipeline(model_name: str, **params) -> Pipeline | BaseEstimator:
    """Create a model pipeline with strict isolation of preprocessing."""
    if model_name == "Persistence":
        return PersistenceBaseline(lag_col="lag_1")
    elif model_name == "Seasonal Naive":
        return SeasonalNaiveBaseline(lag_col="lag_24")
    elif model_name == "Linear Regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression(**params)),
        ])
    elif model_name == "Ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(random_state=42, **params)),
        ])
    elif model_name == "Random Forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", RandomForestRegressor(random_state=42, n_jobs=4, **params)),
        ])
    elif model_name == "Gradient Boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", GradientBoostingRegressor(random_state=42, **params)),
        ])
    elif model_name == "XGBoost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                xgb.XGBRegressor(
                    random_state=42,
                    n_jobs=4,
                    objective="reg:squarederror",
                    eval_metric="mae",
                    **params,
                ),
            ),
        ])
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_hyperparameter_candidates() -> Dict[str, List[Dict[str, Any]]]:
    """Return bounded candidate configurations (up to 3 per ML model)."""
    return {
        "Linear Regression": [
            {"fit_intercept": True},
        ],
        "Ridge": [
            {"alpha": 0.1},
            {"alpha": 1.0},
            {"alpha": 10.0},
        ],
        "Random Forest": [
            {"n_estimators": 80, "max_depth": 8, "min_samples_split": 10},
            {"n_estimators": 100, "max_depth": 12, "min_samples_split": 6},
            {"n_estimators": 120, "max_depth": 16, "min_samples_split": 4},
        ],
        "Gradient Boosting": [
            {"n_estimators": 80, "learning_rate": 0.1, "max_depth": 3},
            {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4},
            {"n_estimators": 120, "learning_rate": 0.1, "max_depth": 4},
        ],
        "XGBoost": [
            {"n_estimators": 80, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.8},
            {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8},
            {"n_estimators": 120, "learning_rate": 0.1, "max_depth": 6, "subsample": 0.9},
        ],
    }
