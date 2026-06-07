from fastapi.testclient import TestClient

from app.api import app
from app.database import init_db


def test_api_core_endpoints():
    init_db()
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "total_events" in metrics.json()


def test_agent_query_endpoint():
    init_db()
    client = TestClient(app)
    response = client.post("/agent/query", json={"question": "Résume les anomalies récentes."})
    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert "sources" in payload

