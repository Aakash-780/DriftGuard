import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from driftguard.app.main import create_app  # noqa: E402


@pytest.fixture
def client(pipeline):
    return TestClient(create_app(pipeline=pipeline))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_version"] == 1


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["version"] == 1 and len(body["feature_names"]) == 30


def test_predict(client, splits):
    rows = splits.X_test.head(3).to_dict("records")
    r = client.post("/predict", json={"records": rows})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3 and set(body["labels"]) <= {"malignant", "benign"}


def test_predict_rejects_bad_input(client, splits):
    row = splits.X_test.head(1).to_dict("records")[0]
    row.pop("mean radius")
    assert client.post("/predict", json={"records": [row]}).status_code == 422
    assert client.post("/predict", json={"records": []}).status_code == 422
    assert client.post("/predict", json={"records": [{"mean radius": "abc"}]}).status_code == 422


def test_monitor_run_generated_and_duplicate(client):
    body = {"scenario": "sudden", "batch_index": 6, "seed": 1}
    first = client.post("/monitor/run", json=body)
    assert first.status_code == 200 and first.json()["status"] == "ok"
    assert client.post("/monitor/run", json=body).json()["status"] == "duplicate_skipped"


def test_monitor_run_requires_exactly_one_source(client):
    assert client.post("/monitor/run", json={}).status_code == 422
    assert client.post("/monitor/run", json={"scenario": "sudden", "records": [{"a": 1.0}]}).status_code == 422


def test_monitor_run_invalid_records(client):
    assert client.post("/monitor/run", json={"records": [{"mean radius": 1.0}] * 30}).status_code == 422


def test_simulate_history_metrics_retrain(client):
    r = client.post("/simulate/drift", json={"scenario": "sudden", "n_batches": 10, "seed": 0})
    assert r.status_code == 200 and len(r.json()["batches"]) == 10
    assert client.post("/simulate/drift", json={"scenario": "sudden", "n_batches": 100}).status_code == 422
    hist = client.get("/history", params={"limit": 5}).json()["events"]
    assert 1 <= len(hist) <= 5
    assert client.get("/history", params={"event_type": "bogus"}).status_code == 422
    m = client.get("/metrics").json()
    assert m["counts"]["monitoring_runs"] == 10
    assert client.post("/retrain", json={}).json()["status"] in ("not_triggered", "promoted", "rejected")


def test_forced_retrain_conflict_without_data(client):
    assert client.post("/retrain", json={"force": True}).status_code == 409
