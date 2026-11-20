"""Load a trained artifact and predict prices for new listings."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from opticar import config as C


class ModelNotFoundError(FileNotFoundError):
    pass


def load_artifact(path=C.MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            f"No trained model at {path}. Run `opticar train` first."
        )
    return joblib.load(path)


def validate(row: dict, artifact: dict) -> list[str]:
    """Return human-readable warnings for inputs the model has not seen."""
    warnings = []
    for col, allowed in artifact["categories"].items():
        val = row.get(col)
        if val is not None and val not in allowed:
            warnings.append(f"{col}={val!r} not seen in training (options: {', '.join(map(str, allowed))})")
    brand, model = row.get("brand"), row.get("model")
    if brand and brand in artifact["models_by_brand"] and model and model not in artifact["models_by_brand"][brand]:
        warnings.append(f"model={model!r} not seen for brand {brand!r}")
    for col, (lo, hi) in artifact["numeric_ranges"].items():
        val = row.get(col)
        if val is not None and not (lo <= val <= hi):
            warnings.append(f"{col}={val} is outside the training range [{lo:g}, {hi:g}]; estimate may be unreliable")
    return warnings


def predict_frame(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Predict for a frame with the feature columns; adds price + 80% band."""
    required_numeric = [c for c in C.NUMERIC if c not in df.columns]
    if required_numeric:
        raise ValueError(f"Missing required column(s): {', '.join(required_numeric)}")
    # Optional (categorical) fields may be absent or None: normalise them to
    # NaN, which is what the pipeline's imputer was fit to treat as missing.
    X = df.reindex(columns=C.FEATURES)
    X = X.astype(object).where(X.notna(), np.nan)
    for col in C.NUMERIC:
        X[col] = pd.to_numeric(X[col])
    pred = artifact["pipeline"].predict(X)
    lo, hi = artifact["interval"]
    out = df.copy()
    out["predicted_price_eur"] = pred.round(0)
    out["low_eur"] = (pred * lo).round(0)
    out["high_eur"] = (pred * hi).round(0)
    return out
