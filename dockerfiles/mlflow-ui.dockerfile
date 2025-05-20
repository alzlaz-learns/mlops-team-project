# Dockerfile.mlflow-ui
FROM python:3.11-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "ui", "--backend-store-uri", "/mlflow/mlruns", "--host", "0.0.0.0"]
