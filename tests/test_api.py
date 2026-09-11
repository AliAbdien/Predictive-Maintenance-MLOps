"""API unit tests. Deliberately don't require a trained MLflow model to be
registered - the model-loading path is exercised separately by actually
running `python -m src.train` + hitting a live server (see README ->
Verifying locally). Here we monkeypatch the loaded-model slot directly so
these tests run fast and in isolation, e.g. in CI with no MLflow server up.
"""
import numpy as np
from fastapi.testclient import TestClient

from api.main import _state, app

client = TestClient(app)


class _FakeModel:
    """predict_proba stands in for the real sklearn pipeline: returns a high
    failure probability whenever tool wear is large, low otherwise - just
    enough behavior to exercise the request/response contract."""

    def predict_proba(self, X):
        tool_wear = X["Tool wear [min]"].iloc[0]
        p_fail = 0.95 if tool_wear > 200 else 0.01
        return np.array([[1 - p_fail, p_fail]])


def setup_function():
    _state["model"] = _FakeModel()
    _state["model_version"] = "test"
    _state["error"] = None


def teardown_function():
    _state["model"] = None
    _state["model_version"] = None
    _state["error"] = None


VALID_PAYLOAD = {
    "type": "M",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
}


def test_health_ok_when_model_loaded():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model_loaded"] is True


def test_health_degraded_when_model_missing():
    _state["model"] = None
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_predict_low_risk():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_predicted_failure"] is False
    assert body["failure_probability"] < 0.5


def test_predict_high_risk_with_high_tool_wear():
    payload = {**VALID_PAYLOAD, "tool_wear_min": 250}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_predicted_failure"] is True
    assert body["failure_probability"] > 0.5


def test_predict_rejects_invalid_type():
    payload = {**VALID_PAYLOAD, "type": "X"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_returns_503_when_model_not_loaded():
    _state["model"] = None
    _state["error"] = "no champion alias registered"
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 503
