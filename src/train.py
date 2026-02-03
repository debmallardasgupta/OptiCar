import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_models(data_path: str):
    df = pd.read_csv(data_path)

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop('price', axis=1)
    y = np.log(df_encoded['price'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=80,
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    metrics = {
        "lr_rmse": np.sqrt(mean_squared_error(y_test, lr_pred)),
        "lr_r2": r2_score(y_test, lr_pred),
        "rf_rmse": np.sqrt(mean_squared_error(y_test, rf_pred)),
        "rf_r2": r2_score(y_test, rf_pred)
    }

    # Save models
    joblib.dump(lr, "../models/linear_regression.pkl")
    joblib.dump(rf, "../models/random_forest.pkl")

    return metrics


if __name__ == "__main__":
    results = train_models("../data/processed/cars_featured.csv")
    print(results)