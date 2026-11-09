from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "thunai",
    }


def test_nonexistent_endpoint() -> None:
    response = client.get("/nonexistent")
    assert response.status_code == 404
