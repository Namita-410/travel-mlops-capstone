"""
Trains the same candidate regression models as models/train_regression.py,
but logs every run (params, metrics, and the model artifact) to MLflow so
runs can be compared and the best one promoted/registered.

Usage:
    mlflow ui   # in one terminal, to view the tracking UI at localhost:5000
    python mlflow/train_with_mlflow.py
"""

import os
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from train_regression import load_and_engineer_features  # reuse identical feature logic

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "flights.csv")

EXPERIMENT_NAME = "flight-price-regression"
RANDOM_STATE = 42


def build_pipeline(categorical_cols, numeric_cols, model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def main():
    default_uri = "sqlite:///" + os.path.join(BASE_DIR, "mlflow", "mlflow.db")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", default_uri))
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_and_engineer_features(DATA_PATH)
    categorical_cols = ["from", "to", "flightType", "agency"]
    numeric_cols = ["time", "distance", "month", "day_of_week", "is_weekend",
                     "quarter", "distance_per_hour"]

    X = df[categorical_cols + numeric_cols]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = {
        "LinearRegression": (LinearRegression(), {}),
        "RandomForest": (
            RandomForestRegressor(n_estimators=200, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": 200, "max_depth": 14},
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = (
            XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                         subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.08},
        )

    best_run_rmse = float("inf")
    best_model_name = None

    for name, (model, params) in candidates.items():
        with mlflow.start_run(run_name=name):
            pipe = build_pipeline(categorical_cols, numeric_cols, model)
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = r2_score(y_test, preds)

            mlflow.log_param("model_type", name)
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)

            mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="pickle")

            print(f"[{name}] MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.4f}")

            if rmse < best_run_rmse:
                best_run_rmse = rmse
                best_model_name = name

    print(f"\nBest model by RMSE: {best_model_name} ({best_run_rmse:.3f})")
    print("Run `mlflow ui --backend-store-uri", mlflow.get_tracking_uri(), "` to inspect all runs.")


if __name__ == "__main__":
    main()
