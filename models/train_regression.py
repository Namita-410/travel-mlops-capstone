"""
Flight Price Regression Model
------------------------------
Trains and evaluates models to predict `price` from flights.csv.
Saves the best model + preprocessing pipeline as a single joblib artifact
so the Flask API can load one object at inference time.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

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

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "flights.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "flight_price_model.joblib")
METRICS_OUT = os.path.join(os.path.dirname(__file__), "regression_metrics.json")

RANDOM_STATE = 42


def load_and_engineer_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Parse date -> handles both MM/DD/YYYY and MM-DD-YYYY seen in the raw file
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"] = df["date"].dt.quarter

    # Route as its own feature -- captures origin-destination pricing patterns
    # that 'from' and 'to' alone (as separate categoricals) partially lose.
    df["route"] = df["from"] + " -> " + df["to"]

    # price-per-distance-km is a strong latent driver worth exposing directly
    # to linear models, though tree models can infer it themselves.
    df["distance_per_hour"] = df["distance"] / df["time"].replace(0, np.nan)
    df["distance_per_hour"] = df["distance_per_hour"].fillna(df["distance_per_hour"].median())

    return df


def build_pipeline(categorical_cols, numeric_cols, model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def main():
    df = load_and_engineer_features(DATA_PATH)

    target = "price"
    categorical_cols = ["from", "to", "flightType", "agency"]
    numeric_cols = ["time", "distance", "month", "day_of_week", "is_weekend",
                     "quarter", "distance_per_hour"]

    X = df[categorical_cols + numeric_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
            n_jobs=-1
        )

    results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipe = build_pipeline(categorical_cols, numeric_cols, model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = evaluate(y_test, preds)
        fitted_pipelines[name] = pipe
        print(f"{name}: {results[name]}")

    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} -> {results[best_name]}")

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "model_name": best_name,
        },
        MODEL_OUT,
    )

    with open(METRICS_OUT, "w") as f:
        json.dump({"best_model": best_name, "all_results": results}, f, indent=2)

    print(f"\nSaved model artifact -> {MODEL_OUT}")
    print(f"Saved metrics -> {METRICS_OUT}")


if __name__ == "__main__":
    main()
