FROM python:3.11-slim-bullseye AS predict

WORKDIR /workspace

COPY requirements.txt pyproject.toml ./
COPY diabetes_predictor/ diabetes_predictor/
COPY .dvc .dvc
COPY data/raw/*.dvc data/raw/

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install . --no-deps --no-cache-dir && \
    pip install --no-cache-dir "dvc[gdrive]" && \
    rm -rf /root/.cache && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["bash", "-c", "cd /workspace && dvc pull --force && exec python -u -m diabetes_predictor.predict_model"]

