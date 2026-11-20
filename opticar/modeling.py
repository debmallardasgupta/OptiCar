"""Model definitions, evaluation and model-selection helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from opticar import config as C


def build_preprocessor() -> ColumnTransformer:
    """Impute + encode. Fit on the training split only (inside the Pipeline)."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, C.NUMERIC),
        ("cat", categorical, C.CATEGORICAL),
    ])


def make_estimators() -> dict:
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=25, min_samples_split=10,
            min_samples_leaf=4, random_state=C.RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, random_state=C.RANDOM_STATE,
        ),
    }


def build_pipeline(estimator) -> TransformedTargetRegressor:
    """Raw feature frame in, price in euros out.

    The target is modelled in log space (price is right-skewed) but the
    wrapper converts predictions back, so callers always see euros.
    """
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", build_preprocessor()), ("model", estimator)]),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def evaluate(y_true, y_pred) -> dict:
    """Metrics in euros and in log space (the space models are trained in)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    lt, lp = np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))
    return {
        "RMSE_eur": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_eur": float(mean_absolute_error(y_true, y_pred)),
        "R2_eur": float(r2_score(y_true, y_pred)),
        "RMSE_log": float(np.sqrt(mean_squared_error(lt, lp))),
        "R2_log": float(r2_score(lt, lp)),
        "MedAPE_pct": float(np.median(np.abs(y_true - y_pred) / y_true) * 100),
    }


def interval_ratios(y_true, y_pred, lo: float = 0.1, hi: float = 0.9) -> tuple[float, float]:
    """Empirical 80% band for actual/predicted on held-out data."""
    ratio = np.asarray(y_true, float) / np.clip(np.asarray(y_pred, float), 1, None)
    return float(np.quantile(ratio, lo)), float(np.quantile(ratio, hi))
