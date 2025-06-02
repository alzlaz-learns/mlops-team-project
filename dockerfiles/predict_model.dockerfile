# predict_model
# Stage 1: Builder
FROM python:3.11-slim-bullseye AS predict

WORKDIR /workspace

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application files and dependencies
COPY requirements.txt pyproject.toml ./
COPY diabetes_predictor/ diabetes_predictor/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install . --no-deps --no-cache-dir

# Stage 2: Runtime
FROM python:3.11-slim-bullseye

WORKDIR /workspace

# Copy installed Python packages
COPY --from=predict /usr/local /usr/local

# Copy code and data needed at runtime
COPY diabetes_predictor/ diabetes_predictor/
COPY data/ data/

# Run prediction
ENTRYPOINT ["python", "-u", "-m", "diabetes_predictor.predict_model"]
