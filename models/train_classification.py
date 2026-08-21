"""
Gender Classification Model
-----------------------------
Predicts users.gender (male / female / none) from each user's aggregated
travel behavior in flights.csv and hotels.csv, plus their age and company.

Note: gender itself carries no travel-pattern signal in most real systems;
this is included because it is one of the project's required deliverables.
The evaluation section below reports results honestly, including if the
model performs close to the majority-class baseline.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "gender_classifier.joblib")
METRICS_OUT = os.path.join(os.path.dirname(__file__), "classification_metrics.json")

RANDOM_STATE = 42


def build_user_features():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    flights = pd.read_csv(os.path.join(DATA_DIR, "flights.csv"))
    hotels = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"))

    flight_agg = flights.groupby("userCode").agg(
        total_flights=("travelCode", "count"),
        avg_flight_price=("price", "mean"),
        avg_flight_distance=("distance", "mean"),
        avg_flight_time=("time", "mean"),
        pct_first_class=("flightType", lambda s: (s == "firstClass").mean()),
        pct_economic=("flightType", lambda s: (s == "economic").mean()),
        n_unique_routes=("from", "nunique"),
        n_unique_agencies=("agency", "nunique"),
    ).reset_index()

    hotel_agg = hotels.groupby("userCode").agg(
        total_hotel_stays=("travelCode", "count"),
        avg_hotel_price=("price", "mean"),
        avg_stay_days=("days", "mean"),
        avg_total_spend=("total", "mean"),
        n_unique_places=("place", "nunique"),
    ).reset_index()

    df = users.merge(flight_agg, left_on="code", right_on="userCode", how="left")
    df = df.merge(hotel_agg, left_on="code", right_on="userCode", how="left", suffixes=("", "_hotel"))
    df = df.fillna(0)
    return df


def main():
    df = build_user_features()

    feature_cols_numeric = [
        "age", "total_flights", "avg_flight_price", "avg_flight_distance",
        "avg_flight_time", "pct_first_class", "pct_economic", "n_unique_routes",
        "n_unique_agencies", "total_hotel_stays", "avg_hotel_price",
        "avg_stay_days", "avg_total_spend", "n_unique_places",
    ]
    feature_cols_categorical = ["company"]

    X = df[feature_cols_categorical + feature_cols_numeric]
    y = df["gender"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_categorical),
        ("num", StandardScaler(), feature_cols_numeric),
    ])

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = {}
    fitted = {}
    baseline_acc = y_test.value_counts(normalize=True).max()

    for name, clf in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)

        results[name] = {
            "accuracy": round(acc, 4),
            "macro_precision": round(precision, 4),
            "macro_recall": round(recall, 4),
            "macro_f1": round(f1, 4),
        }
        fitted[name] = pipe
        print(f"{name}: {results[name]}")
        print(classification_report(y_test, preds, zero_division=0))

    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_pipeline = fitted[best_name]

    print(f"\nMajority-class baseline accuracy: {baseline_acc:.4f}")
    print(f"Best model: {best_name} -> {results[best_name]}")
    if results[best_name]["accuracy"] <= baseline_acc + 0.03:
        print("NOTE: model performs close to the majority-class baseline — "
              "gender shows little to no predictive signal in this travel data. "
              "Report this honestly rather than overstating model quality.")

    joblib.dump({
        "pipeline": best_pipeline,
        "categorical_cols": feature_cols_categorical,
        "numeric_cols": feature_cols_numeric,
        "model_name": best_name,
        "classes": sorted(y.unique().tolist()),
    }, MODEL_OUT)

    with open(METRICS_OUT, "w") as f:
        json.dump({
            "best_model": best_name,
            "all_results": results,
            "majority_class_baseline_accuracy": round(baseline_acc, 4),
        }, f, indent=2)

    print(f"\nSaved model -> {MODEL_OUT}")
    print(f"Saved metrics -> {METRICS_OUT}")


if __name__ == "__main__":
    main()
