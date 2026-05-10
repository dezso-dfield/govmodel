"""Lightweight tests for inference helpers — no torch/HF needed."""
from __future__ import annotations

from pathlib import Path

from govmodel.inference import Prediction, PredictionLogger


def _make_pred(label: str = "bezwaar", score: float = 0.9, abstain: bool = False) -> Prediction:
    return Prediction(
        request_id="abc12345",
        text_chars=42,
        text_hash="deadbeef",
        top_label=label,
        top_score=score,
        threshold=0.5,
        above_threshold=[(label, score)],
        all_scores={label: score},
        abstain=abstain,
        abstain_reason=None if not abstain else "low conf",
        latency_ms=12.3,
        model_id="test-model",
    )


def test_prediction_serialises_to_jsonl_round_trip():
    import json
    pred = _make_pred()
    line = pred.to_jsonl()
    parsed = json.loads(line)
    assert parsed["top_label"] == "bezwaar"
    assert parsed["latency_ms"] == 12.3


def test_prediction_to_public_dict_uses_lists_for_tuples():
    pred = _make_pred()
    d = pred.to_public_dict()
    assert d["above_threshold"][0] == ["bezwaar", 0.9]


def test_prediction_logger_omits_raw_text_by_default(tmp_path: Path):
    logger = PredictionLogger(tmp_path / "predictions.jsonl", log_raw_text=False)
    logger.log(_make_pred(), raw_text="een burgerbrief met PII er in")
    line = logger.path.read_text(encoding="utf-8").strip()
    assert "PII" not in line
    assert "burgerbrief" not in line
    assert "deadbeef" in line   # hash made it through


def test_prediction_logger_includes_raw_text_when_opted_in(tmp_path: Path):
    logger = PredictionLogger(tmp_path / "predictions.jsonl", log_raw_text=True)
    logger.log(_make_pred(), raw_text="opt-in tekst")
    assert "opt-in tekst" in logger.path.read_text(encoding="utf-8")


def test_prediction_logger_records_user_feedback(tmp_path: Path):
    import json
    logger = PredictionLogger(tmp_path / "predictions.jsonl", log_raw_text=False)
    logger.log(_make_pred(label="klacht"), user_label="bezwaar")
    record = json.loads(logger.path.read_text(encoding="utf-8").strip())
    assert record["user_label"] == "bezwaar"
    assert record["top_label"] == "klacht"
