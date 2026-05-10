"""Tests for hard-negative mining."""
from __future__ import annotations

import json
from pathlib import Path

from govmodel.mining import MiningConfig, mine, stats, to_label_queue


def _write_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _pred(rid: str, top_label: str, top_score: float,
          above: list[list] | None = None, **extra) -> dict:
    return {
        "ts": "2026-05-10T12:00:00Z",
        "request_id": rid,
        "top_label": top_label,
        "top_score": top_score,
        "above_threshold": above or [[top_label, top_score]],
        **extra,
    }


def test_user_disagreement_is_top_priority(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    _write_log(log, [
        _pred("a", "klacht", 0.85),
        # User-feedback row (the FastAPI feedback endpoint writes this)
        {"ts": "2026-05-10T12:01:00Z", "request_id": "a",
         "user_label": "bezwaar", "feedback_only": True},
    ])
    cands = mine(log)
    assert len(cands) == 1
    assert cands[0].reason == "user_disagreement"
    assert cands[0].score > 0.6
    assert cands[0].record["user_label"] == "bezwaar"


def test_low_confidence_band(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    _write_log(log, [
        _pred("low",  "klacht", 0.35),  # inside default window
        _pred("high", "klacht", 0.80),  # too confident — skip
        _pred("absent", "klacht", 0.10),  # below window
    ])
    cands = mine(log)
    reasons = [c.reason for c in cands]
    assert "low_confidence" in reasons
    assert all(c.reason != "low_confidence" or 0.20 <= c.record["top_score"] <= 0.50 for c in cands)


def test_conflicted_multilabel_detected(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    _write_log(log, [
        _pred("conf", "klacht", 0.62,
              above=[["klacht", 0.62], ["bezwaar", 0.58], ["aanvraag", 0.10]]),
        _pred("clean", "klacht", 0.92,
              above=[["klacht", 0.92], ["bezwaar", 0.05]]),
    ])
    cands = mine(log)
    reasons = [c.reason for c in cands]
    assert "conflicted_multilabel" in reasons
    # The conflicted one should outrank the clean one (which shouldn't even be present)
    conflicted = [c for c in cands if c.reason == "conflicted_multilabel"][0]
    assert conflicted.record["request_id"] == "conf"


def test_user_only_filters_other_reasons(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    _write_log(log, [
        _pred("a", "klacht", 0.35),  # low conf
        _pred("b", "klacht", 0.62,
              above=[["klacht", 0.62], ["bezwaar", 0.58]]),  # conflicted
        _pred("c", "klacht", 0.80),
        {"request_id": "c", "user_label": "bezwaar", "feedback_only": True},
    ])
    cands = mine(log, MiningConfig(user_disagreement_only=True))
    assert len(cands) == 1
    assert cands[0].reason == "user_disagreement"


def test_limit_caps_output(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    _write_log(log, [_pred(f"r{i}", "klacht", 0.35) for i in range(50)])
    cands = mine(log, limit=10)
    assert len(cands) == 10


def test_stats_groups_by_reason(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    _write_log(log, [
        _pred("a", "klacht", 0.35),
        _pred("b", "klacht", 0.36),
        _pred("c", "klacht", 0.62,
              above=[["klacht", 0.62], ["bezwaar", 0.55]]),
    ])
    cands = mine(log)
    s = stats(cands)
    assert s["n_total"] == 3
    assert s["by_reason"]["low_confidence"] == 2
    assert s["by_reason"]["conflicted_multilabel"] == 1


def test_to_label_queue_writes_jsonl(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    out = tmp_path / "queue.jsonl"
    _write_log(log, [_pred("a", "klacht", 0.35)])
    cands = mine(log)
    n = to_label_queue(cands, out)
    assert n == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert rows[0]["_mine_reason"] == "low_confidence"
    assert "_mine_score" in rows[0]


def test_corrupt_lines_are_skipped(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_pred("a", "klacht", 0.35)) + "\n")
        f.write("not-valid-json\n")
        f.write(json.dumps(_pred("b", "klacht", 0.40)) + "\n")
    cands = mine(log)
    assert len(cands) == 2
