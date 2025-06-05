# src/predictor.py
import os

import joblib
from fastapi import FastAPI, Request

# Load model from the correct path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../diabetes_predictor/models/model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "FastAPI running on Cloud Functions"}

@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()
    features = payload.get("features")
    if features is None:
        return {"error": "Missing 'features' in request"}
    prediction = model.predict([features])
    return {"prediction": prediction.tolist()}
