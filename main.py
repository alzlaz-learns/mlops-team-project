from fastapi.testclient import TestClient
from src.predictor import app

client = TestClient(app)

def gcf_entry_point(request):
    if request.method == "POST":
        json_data = request.get_json()
        response = client.post("/predict", json=json_data)
    else:
        response = client.get("/")
    return (response.content, response.status_code, {"Content-Type": "application/json"})
