"""
Airflow DAG: flight_price_pipeline
------------------------------------
Automates the flight-price regression workflow:
  1. validate_data       -> sanity-check flights.csv (schema, nulls, row count)
  2. train_model          -> run models/train_regression.py
  3. evaluate_model       -> check the new model beats a minimum quality bar
  4. notify                -> log completion (stand-in for a Slack/email hook)

Schedule: daily, to reflect new flight bookings landing each day in a
production system (adjust to your real data-refresh cadence).
"""

from datetime import datetime, timedelta
import json
import os
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator

REPO_ROOT = "/opt/airflow/repo"  # mount point inside the Airflow container
DATA_PATH = os.path.join(REPO_ROOT, "data", "flights.csv")
METRICS_PATH = os.path.join(REPO_ROOT, "models", "regression_metrics.json")

MIN_R2_THRESHOLD = 0.90  # gate: reject a retrained model that regresses badly

default_args = {
    "owner": "namita",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def validate_data(**context):
    import pandas as pd

    df = pd.read_csv(DATA_PATH)
    required_cols = {"travelCode", "userCode", "from", "to", "flightType",
                      "price", "time", "distance", "agency", "date"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"flights.csv is missing columns: {missing_cols}")
    if df.isnull().any().any():
        raise ValueError("flights.csv contains null values")
    if len(df) < 1000:
        raise ValueError(f"flights.csv row count too low: {len(df)}")
    print(f"Validated flights.csv: {len(df)} rows, no missing columns/nulls.")


def train_model(**context):
    result = subprocess.run(
        ["python3", os.path.join(REPO_ROOT, "models", "train_regression.py")],
        capture_output=True, text=True, check=False
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Training script failed")


def evaluate_model(**context):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    best = metrics["best_model"]
    r2 = metrics["all_results"][best]["R2"]
    print(f"Best model: {best}, R2={r2}")
    if r2 < MIN_R2_THRESHOLD:
        raise ValueError(f"Model R2 {r2} below minimum threshold {MIN_R2_THRESHOLD}; blocking promotion.")


def notify(**context):
    print("Pipeline complete: flight price model validated, trained, and evaluated successfully.")


with DAG(
    dag_id="flight_price_pipeline",
    description="Validate data, retrain, and evaluate the flight price regression model",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "regression", "flights"],
) as dag:

    t1 = PythonOperator(task_id="validate_data", python_callable=validate_data)
    t2 = PythonOperator(task_id="train_model", python_callable=train_model)
    t3 = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    t4 = PythonOperator(task_id="notify", python_callable=notify)

    t1 >> t2 >> t3 >> t4
