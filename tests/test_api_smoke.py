"""End-to-end API smoke test.

Stubs the heavy Classifier so we can exercise routing, validation, the
abstain path, the auth gate, and the analytics endpoint without loading
torch / transformers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi.testclient")

from govmodel import api as api_module  # noqa: E402
from govmodel.inference import Prediction  # noqa: E402


class _StubClassifier:
    def __init__(self, top_label: str = "bezwaar", top_score: float = 0.91, abstain: bool = False):
        self.top_label = top_label
        self.top_score = top_score
        self.abstain = abstain

    def classify(self, text: str, threshold: float = 0.5, request_id=None) -> Prediction:
        return Prediction(
            request_id=request_id or "stub0001",
            text_chars=len(text),
            text_hash="stubhash",
            top_label=self.top_label,
            top_score=self.top_score,
            threshold=threshold,
            above_threshold=[(self.top_label, self.top_score)],
            all_scores={self.top_label: self.top_score, "klacht": 0.05},
            abstain=self.abstain,
            abstain_reason="stub" if self.abstain else None,
            latency_ms=12.0,
            model_id="stub",
        )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("GOVMODEL_PREDICTION_LOG", str(log))
    # Re-init the prediction logger so it picks up the env var.
    api_module._prediction_logger.path = log
    api_module._classifier = _StubClassifier()
    from fastapi.testclient import TestClient
    return TestClient(api_module.app), log


def test_health_endpoint(client):
    c, _ = client
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "model_id" in body


def test_classify_happy_path(client):
    c, log = client
    r = c.post("/api/classify",
               json={"text": "Hierbij teken ik bezwaar aan tegen uw besluit.", "threshold": 0.5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["top_label"] == "bezwaar"
    assert body["abstain"] is False
    assert log.exists()
    assert "bezwaar" in log.read_text(encoding="utf-8")


def test_classify_rejects_empty_text(client):
    c, _ = client
    r = c.post("/api/classify", json={"text": "", "threshold": 0.5})
    assert r.status_code == 422  # pydantic min_length


def test_classify_abstain_path(client, monkeypatch):
    api_module._classifier = _StubClassifier(top_score=0.10, abstain=True)
    c, _ = client
    r = c.post("/api/classify", json={"text": "ambigue tekst", "threshold": 0.5})
    assert r.status_code == 200
    assert r.json()["abstain"] is True


def test_auth_token_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("GOVMODEL_API_TOKEN", "secret123")
    log = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("GOVMODEL_PREDICTION_LOG", str(log))
    api_module._prediction_logger.path = log
    api_module._classifier = _StubClassifier()
    from fastapi.testclient import TestClient
    c = TestClient(api_module.app)

    r = c.post("/api/classify", json={"text": "tekst", "threshold": 0.5})
    assert r.status_code == 401

    r = c.post("/api/classify", json={"text": "tekst", "threshold": 0.5},
               headers={"Authorization": "Bearer secret123"})
    assert r.status_code == 200


def test_feedback_records_user_label(client):
    c, log = client
    c.post("/api/classify", json={"text": "iets met bezwaar", "threshold": 0.5})
    r = c.post("/api/classify/stub0001/feedback", json={"user_label": "klacht"})
    assert r.status_code == 200
    text = log.read_text(encoding="utf-8")
    assert '"user_label": "klacht"' in text


def test_analytics_summary_reflects_predictions(client):
    c, _ = client
    c.post("/api/classify", json={"text": "een", "threshold": 0.5})
    c.post("/api/classify", json={"text": "twee", "threshold": 0.5})
    r = c.get("/api/analytics/summary?window_hours=24")
    assert r.status_code == 200
    body = r.json()
    assert body["total_predictions"] >= 2
    assert "bezwaar" in body["label_counts"]


def test_dashboard_html_served(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "govmodel" in r.text.lower()
