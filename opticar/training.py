"""End-to-end training: prepare -> split -> CV model selection -> test report."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split

from opticar import __version__
from opticar import config as C
from opticar import data as D
from opticar import modeling as M

TUNE_SPACES = {
    "Random Forest": {
        "regressor__model__n_estimators": [60, 100, 150],
        "regressor__model__max_depth": [15, 25, None],
        "regressor__model__min_samples_leaf": [2, 4, 8],
        "regressor__model__max_features": [0.3, 0.5, 1.0],
    },
    "Gradient Boosting": {
        "regressor__model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "regressor__model__max_iter": [200, 400, 600],
        "regressor__model__max_leaf_nodes": [15, 31, 63],
        "regressor__model__l2_regularization": [0.0, 0.1, 1.0],
    },
}


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, flush=True)


def train(data_path=C.RAW_DATA, model_path=C.MODEL_PATH, reports_dir=C.REPORTS_DIR,
          cv: int = 3, tune: bool = False, tune_iter: int = 8,
          plots: bool = True, quiet: bool = False) -> dict:
    t0 = time.time()
    df = D.prepare(D.load_raw(data_path))
    _log(f"Prepared {len(df):,} listings ({len(C.FEATURES)} features)", quiet)

    X, y = df[C.FEATURES], df[C.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE
    )

    # 1) Model selection by cross-validation on the TRAIN split only.
    estimators = M.make_estimators()
    cv_rmse: dict[str, float] = {}
    for name, est in estimators.items():
        scores = cross_val_score(
            M.build_pipeline(est), X_tr, y_tr, cv=cv, n_jobs=1,
            scoring="neg_root_mean_squared_error",
        )
        cv_rmse[name] = float(-scores.mean())
        _log(f"  CV RMSE (EUR) {name:<20} {cv_rmse[name]:>10,.0f}", quiet)

    best_name = min((n for n in cv_rmse if n != "Baseline (mean)"), key=cv_rmse.get)
    _log(f"Best by CV: {best_name}", quiet)

    # 2) Fit every candidate on the train split (+ optional tuning of the winner).
    fitted = {n: M.build_pipeline(e).fit(X_tr, y_tr) for n, e in estimators.items()}
    final_name, tuned_params = best_name, None
    if tune and best_name in TUNE_SPACES:
        _log(f"Tuning {best_name} ({tune_iter} candidates)...", quiet)
        search = RandomizedSearchCV(
            M.build_pipeline(estimators[best_name]), TUNE_SPACES[best_name],
            n_iter=tune_iter, cv=cv, random_state=C.RANDOM_STATE,
            scoring="neg_root_mean_squared_error", n_jobs=1,
        )
        search.fit(X_tr, y_tr)
        tuned_params = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
        final_name = f"{best_name} (tuned)"
        cv_rmse[final_name] = float(-search.best_score_)
        fitted[final_name] = search.best_estimator_
        _log(f"  tuned CV RMSE {-search.best_score_:,.0f}  {tuned_params}", quiet)
    final = fitted[final_name]

    # 3) One-shot evaluation on the untouched test split.
    test_metrics = {n: M.evaluate(y_te, m.predict(X_te)) for n, m in fitted.items()}
    pred = final.predict(X_te)
    q_lo, q_hi = M.interval_ratios(y_te, pred)

    # 4) Permutation importance on raw columns (groups one-hot columns).
    sample = X_te.sample(min(3000, len(X_te)), random_state=C.RANDOM_STATE)
    imp = permutation_importance(
        final, sample, y_te.loc[sample.index], n_repeats=3,
        random_state=C.RANDOM_STATE, scoring="r2", n_jobs=1,
    )
    importance = (
        pd.DataFrame({"feature": C.FEATURES, "importance": imp.importances_mean})
        .sort_values("importance", ascending=False).reset_index(drop=True)
    )

    summary = {
        "version": __version__,
        "sklearn": sklearn.__version__,
        "n_rows": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "cv_folds": cv, "best_model": final_name, "tuned_params": tuned_params,
        "cv_rmse_eur": cv_rmse, "test_metrics": test_metrics,
        "interval_ratio_p10_p90": [q_lo, q_hi],
        "seconds": round(time.time() - t0, 1),
    }

    # 5) Persist: one artifact = pipeline + metadata needed by the CLI.
    categories = {c: sorted(df[c].dropna().unique().tolist()) for c in C.CATEGORICAL if c != "model"}
    models_by_brand = df.groupby("brand")["model"].agg(lambda s: sorted(s.dropna().unique())).to_dict()
    numeric_ranges = {c: [float(df[c].min()), float(df[c].max())] for c in C.NUMERIC}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": final, "summary": summary, "categories": categories,
        "models_by_brand": models_by_brand, "numeric_ranges": numeric_ranges,
        "interval": (q_lo, q_hi),
    }, model_path, compress=3)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    if plots:
        _make_plots(reports_dir, importance, y_te, pred)
    summary["model_path"] = str(model_path)
    return summary


def _make_plots(reports_dir: Path, importance: pd.DataFrame, y_te, pred) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # plots are optional
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    top = importance.head(8)[::-1]
    ax.barh(top["feature"], top["importance"], color="#2b6cb0")
    ax.set_xlabel("Drop in R² when feature is shuffled")
    ax.set_title("Permutation importance (test sample)")
    fig.tight_layout(); fig.savefig(reports_dir / "feature_importance.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te, pred, s=3, alpha=0.25, color="#2b6cb0")
    lim = [100, 150_000]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual price (EUR)"); ax.set_ylabel("Predicted price (EUR)")
    ax.set_title("Predicted vs actual (test set)")
    fig.tight_layout(); fig.savefig(reports_dir / "predicted_vs_actual.png", dpi=150); plt.close(fig)
