from fastapi.testclient import TestClient
from main import app, history

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_predict_positive():
    response = client.post("/predict/",
                           json={"text": "I like machine learning!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'


def test_predict_negative():
    response = client.post("/predict/",
                           json={"text": "I hate machine learning!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'


def test_get_statistic_healthy():
    response = client.get("/stats")
    assert response.status_code == 200


def test_get_statistic_returns_correct_request_count_and_body():
    history.data.clear()

    client.post("/predict/", json={"text": "test11"})
    client.post("/predict/", json={"text": "test22"})

    response = client.get("/stats")
    print(response)
    assert response.json() == {"2": ["test11", "test22"]}
