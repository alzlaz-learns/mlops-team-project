# Stage 1: Builder
FROM python:3.11-slim-bullseye AS trainer

WORKDIR /workspace

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependency files and source code
COPY requirements.txt pyproject.toml ./
COPY diabetes_predictor/ diabetes_predictor/
COPY monitoring/ monitoring/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install . --no-deps --no-cache-dir

# Stage 2: Runtime
FROM python:3.11-slim-bullseye

WORKDIR /workspace

# Copy installed packages from builder
COPY --from=trainer /usr/local /usr/local

# Copy application code and data
COPY diabetes_predictor/ diabetes_predictor/
COPY data/ data/
COPY monitoring/ monitoring/

# Set entrypoint
ENTRYPOINT ["python", "-u", "-m", "diabetes_predictor.train_model"]
