# Travel & Tourism MLOps Capstone

End-to-end ML system on the users / flights / hotels travel dataset: flight
price regression, gender classification, and hotel recommendation — served,
containerized, orchestrated, tracked, and deployed.

## Repo structure

```
data/            users.csv, flights.csv, hotels.csv
models/          training scripts + saved model artifacts (regression,
                 classification, recommender) + metrics JSON
api/             Flask REST API serving the regression model
docker/          Dockerfile + slim requirements for the API image
k8s/             Kubernetes Deployment, Service, and HPA manifests
airflow/dags/    DAG orchestrating data validation -> train -> evaluate
jenkins/         Jenkinsfile for the CI/CD pipeline
mlflow/          MLflow-tracked training script (logs params/metrics/models)
streamlit_app/   Streamlit dashboard: price predictor + recommendations + EDA
notebooks/       Colab notebook covering all three models with analysis
docs/            Stage-by-stage documentation with screenshots
```

## Datasets

- **users.csv** (1,340 rows): code, company, name, gender, age
- **flights.csv** (271,888 rows): travelCode, userCode, from, to, flightType,
  price, time, distance, agency, date
- **hotels.csv** (40,552 rows): travelCode, userCode, name, place, days,
  price, total, date

`travelCode` + `userCode` link a flight/hotel pair to the same trip and user.
Note: each `place` maps to exactly one hotel in this dataset, so the
recommendation model recommends *places* — see `models/train_recommender.py`
for why that's the correct framing here.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Train all three models
python3 models/train_regression.py
python3 models/train_classification.py
python3 models/train_recommender.py

# 2. Serve the regression model
python3 api/app.py            # http://localhost:5000

# 3. Track experiments with MLflow
python3 mlflow/train_with_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

# 4. Run the Streamlit app
streamlit run streamlit_app/app.py

# 5. Build & run the API container
docker build -f docker/Dockerfile -t flight-price-api .
docker run -p 5000:5000 flight-price-api

# 6. Deploy to Kubernetes
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/hpa.yml
```

## Model results (see models/*.json for full metrics)

| Model | Metric | Result |
|---|---|---|
| Flight price regression (XGBoost) | R² / MAE | 0.9999 / $2.22 |
| Gender classification (RandomForest) | Accuracy | 0.35 (baseline 0.336 — see note below) |
| Hotel/place recommender | Hybrid collaborative + price-affinity | top-3 per user |

**Honest note on classification:** gender shows almost no predictive signal
in travel behavior in this dataset — the model performs close to the
majority-class baseline. That's reported as-is rather than inflated.

## API endpoints

- `GET /health` — liveness check
- `GET /model-info` — active model + offline metrics
- `POST /predict` — single flight price prediction
- `POST /predict-batch` — batch predictions

## CI/CD

`jenkins/Jenkinsfile` runs: install deps → train & validate model → API
smoke test → build Docker image → push to registry → deploy to Kubernetes.

## Orchestration

`airflow/dags/flight_price_pipeline_dag.py` runs daily: validate data →
retrain → evaluate against a minimum R² gate → notify.
