"""Tests for the analytics aggregator.

Pure stdlib — no model load, no FastAPI, just JSONL → dict.
"""
from __future__ import annotations

import json
from pathlib import Path

from govmodel.analytics import summary


def _write_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_summary_empty_log(tmp_path: Path):
    s = summary(tmp_path / "missing.jsonl")
    assert s["total_predictions"] == 0
    assert s["abstain_rate"] == 0.0
    assert s["recent"] == []
    assert s["label_counts"] == {}


def test_summary_aggregates_label_counts_and_buckets(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    _write_log(log, [
        {"ts": "2026-05-10T10:00:00Z", "request_id": "a", "top_label": "bezwaar",
         "top_score": 0.91, "abstain": False, "above_threshold": [["bezwaar", 0.91]],
         "latency_ms": 30.0},
        {"ts": "2026-05-10T10:30:00Z", "request_id": "b", "top_label": "klacht",
         "top_score": 0.62, "abstain": False, "above_threshold": [["klacht", 0.62]],
         "latency_ms": 40.0},
        {"ts": "2026-05-10T11:00:00Z", "request_id": "c", "top_label": "bezwaar",
         "top_score": 0.18, "abstain": True, "above_threshold": [],
         "latency_ms": 25.0},
    ])
    s = summary(log, window_hours=1_000_000)  # everything in window
    assert s["total_predictions"] == 3
    assert s["label_counts"]["bezwaar"] == 2
    assert s["label_counts"]["klacht"] == 1
    assert s["abstain_count"] == 1
    assert 0.30 < s["abstain_rate"] < 0.34
    assert s["score_buckets"]["<0.2"] == 1
    assert s["score_buckets"]["0.5-0.8"] == 1
    assert s["score_buckets"][">=0.8"] == 1
    assert s["latency_ms"]["n"] == 3


def test_summary_tracks_feedback_confusions(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    _write_log(log, [
        {"ts": "2026-05-10T10:00:00Z", "request_id": "x", "top_label": "klacht",
         "top_score": 0.6, "abstain": False, "above_threshold": [["klacht", 0.6]],
         "latency_ms": 20.0, "user_label": "bezwaar"},
        {"ts": "2026-05-10T10:01:00Z", "request_id": "y", "top_label": "bezwaar",
         "top_score": 0.8, "abstain": False, "above_threshold": [["bezwaar", 0.8]],
         "latency_ms": 20.0, "user_label": "bezwaar"},
    ])
    s = summary(log)
    assert s["feedback_correct"] == 1
    assert s["feedback_wrong"] == 1
    assert s["feedback_accuracy"] == 0.5
    assert s["top_confusions"][0] == {"predicted": "klacht", "user_label": "bezwaar", "n": 1}


def test_summary_skips_corrupt_lines(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        f.write('{"ts":"2026-05-10T10:00:00Z","top_label":"bezwaar","top_score":0.9,"abstain":false,"latency_ms":10}\n')
        f.write('not-valid-json-broken-line\n')
        f.write('{"ts":"2026-05-10T11:00:00Z","top_label":"klacht","top_score":0.7,"abstain":false,"latency_ms":12}\n')
    s = summary(log)
    assert s["total_predictions"] == 2  # corrupt line skipped, not crashed on
