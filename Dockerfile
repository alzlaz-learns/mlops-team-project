# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ ./src
COPY main.py .  
COPY diabetes_predictor/ ./diabetes_predictor

# Expose port for FastAPI
EXPOSE 8080

# Start FastAPI with uvicorn
CMD ["uvicorn", "src.predictor:app", "--host", "0.0.0.0", "--port", "8080"]
