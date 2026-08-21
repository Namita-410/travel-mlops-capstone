# Flight Price Prediction API - container image
FROM python:3.11-slim

WORKDIR /app

# Install only what the API needs (keeps the image lean; Airflow/Streamlit
# have their own images below rather than bloating this one).
COPY docker/api-requirements.txt .
RUN pip install --no-cache-dir -r api-requirements.txt

COPY api/ ./api/
COPY models/ ./models/

EXPOSE 5000

ENV PORT=5000

# gunicorn instead of the Flask dev server for a production-appropriate image
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--chdir", "api", "app:app"]
