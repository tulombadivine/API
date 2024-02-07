from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_list_client():
    response = client.get("/list_client")
    assert response.status_code == 200
    assert isinstance(response.json(), list) 

def test_client_adress():
    response = client.post("/client_adress", json={"client_id": 100042})
    assert response.status_code == 200
    data = response.json()
    assert "client_id" in data
    assert "first_name" in data
    assert "last_name" in data
    assert "adress" in data

def test_predict_for_client():
    response = client.post("/predict_for_client", json={"client_id": 100042})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_image_not_found():
    response = client.get("/get_image/non_existing_image")
    assert response.status_code == 404

def test_prediction_for_all():
    response = client.get("/prediction_for_all")
    assert response.status_code == 200
    assert "data_viz" in response.json()
    assert "list_var" in response.json()