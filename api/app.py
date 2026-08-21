"""
Flask REST API - Flight Price Prediction
------------------------------------------
Serves the trained regression pipeline (models/flight_price_model.joblib).

Endpoints:
  GET  /health            -> liveness check
  GET  /model-info         -> which model is loaded + its offline metrics
  POST /predict            -> predict a single flight price
  POST /predict-batch      -> predict prices for a list of flights
"""

import os
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "flight_price_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "regression_metrics.json")

app = Flask(__name__)

_artifact = joblib.load(MODEL_PATH)
PIPELINE = _artifact["pipeline"]
CATEGORICAL_COLS = _artifact["categorical_cols"]
NUMERIC_COLS = _artifact["numeric_cols"]
MODEL_NAME = _artifact["model_name"]

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

REQUIRED_RAW_FIELDS = ["from", "to", "flightType", "agency", "time", "distance", "date"]


def engineer_features(record: dict) -> pd.DataFrame:
    """Rebuild the same engineered features used at training time from raw
    request fields, so the API contract stays human-friendly (date, from,
    to...) instead of forcing callers to precompute month/day_of_week/etc."""
    df = pd.DataFrame([record])
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["distance_per_hour"] = df["distance"] / df["time"]
    return df[CATEGORICAL_COLS + NUMERIC_COLS]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({"model_name": MODEL_NAME, "metrics": METRICS}), 200


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    missing = [f for f in REQUIRED_RAW_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        features = engineer_features(payload)
        prediction = float(PIPELINE.predict(features)[0])
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 400

    return jsonify({
        "predicted_price": round(prediction, 2),
        "model_used": MODEL_NAME,
    }), 200


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, list):
        return jsonify({"error": "Request body must be a JSON array of flight records"}), 400

    predictions = []
    for i, record in enumerate(payload):
        missing = [f for f in REQUIRED_RAW_FIELDS if f not in record]
        if missing:
            predictions.append({"index": i, "error": f"Missing fields: {missing}"})
            continue
        try:
            features = engineer_features(record)
            pred = float(PIPELINE.predict(features)[0])
            predictions.append({"index": i, "predicted_price": round(pred, 2)})
        except Exception as exc:
            predictions.append({"index": i, "error": str(exc)})

    return jsonify({"model_used": MODEL_NAME, "predictions": predictions}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
