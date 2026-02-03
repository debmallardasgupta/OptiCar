import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(y_true, y_pred) -> dict:
    """Evaluate regression model"""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }
