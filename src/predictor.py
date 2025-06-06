# src/predictor.py
import os
from typing import Any, Dict, List

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../diabetes_predictor/models/model.joblib"))
model = joblib.load(MODEL_PATH)

app = FastAPI()

@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "FastAPI running on Cloud Functions"}

class FeaturesInput(BaseModel):
    features: List[float]

@app.post("/predict")
async def predict(data: FeaturesInput) -> Dict[str, Any]:
    prediction = model.predict([data.features])
    return {"prediction": prediction.tolist()}
