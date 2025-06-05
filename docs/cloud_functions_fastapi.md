# FastAPI Deployment via Google Cloud Functions

## Overview
This FastAPI app exposes a machine learning model as a REST API endpoint using Google Cloud Functions. The function includes a health check and a prediction route.

## Files
- `src/predictor.py`: FastAPI logic and model loading
- `main.py`: Entry point for GCP Cloud Functions
- `requirements.txt`: Project dependencies including FastAPI and GCP-specific tools

## Deployment Command (Gen 1)
```bash
gcloud functions deploy predictor-api \
  --entry-point app \
  --runtime python310 \
  --trigger-http \
  --allow-unauthenticated \
  --source=. \
  --region=us-central1 \
  --no-gen2
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
