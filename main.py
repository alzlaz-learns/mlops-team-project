from typing import Tuple

from fastapi.testclient import TestClient
from flask import Request

from src.predictor import app

client = TestClient(app)

def gcf_entry_point(request: Request) -> Tuple[bytes, int, dict[str, str]]:
    if request.method == "POST":
        json_data = request.get_json()
        response = client.post("/predict", json=json_data)
    else:
        response = client.get("/")
    return (response.content, response.status_code, {"Content-Type": "application/json"})
