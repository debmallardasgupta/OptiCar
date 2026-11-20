import numpy as np
import pandas as pd

from opticar import config as C
from opticar import data as D
from opticar import modeling as M


def _fit(raw_df):
    df = D.prepare(raw_df)
    est = M.make_estimators()["Random Forest"].set_params(n_estimators=20)
    pipe = M.build_pipeline(est).fit(df[C.FEATURES], df[C.TARGET])
    return df, pipe


def test_pipeline_predicts_in_euros_and_beats_baseline(raw_df):
    df, pipe = _fit(raw_df)
    pred = pipe.predict(df[C.FEATURES])
    assert pred.min() > 0
    base = M.make_estimators()["Baseline (mean)"]
    base_pred = M.build_pipeline(base).fit(df[C.FEATURES], df[C.TARGET]).predict(df[C.FEATURES])
    assert M.evaluate(df[C.TARGET], pred)["RMSE_eur"] < M.evaluate(df[C.TARGET], base_pred)["RMSE_eur"]


def test_pipeline_handles_unseen_category_and_missing_values(raw_df):
    df, pipe = _fit(raw_df)
    row = df[C.FEATURES].iloc[[0]].copy()
    row["brand"] = "never_seen_brand"
    row["notRepairedDamage"] = np.nan
    row["powerPS"] = np.nan
    assert np.isfinite(pipe.predict(row)).all()


def test_imputer_is_fit_on_training_data_only(raw_df):
    df = D.prepare(raw_df)
    X_tr, X_te = df[C.FEATURES].iloc[:300], df[C.FEATURES].iloc[300:]
    pipe = M.build_pipeline(M.make_estimators()["Linear Regression"]).fit(X_tr, df[C.TARGET].iloc[:300])
    imputer = pipe.regressor_.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    # medians learnt must come from the train split, not the full frame
    assert np.allclose(imputer.statistics_, X_tr[C.NUMERIC].median().values)


def test_interval_ratios_bracket_one():
    y = np.array([100, 200, 300, 400, 500.0])
    lo, hi = M.interval_ratios(y, y * 1.0)
    assert lo <= 1.0 <= hi
