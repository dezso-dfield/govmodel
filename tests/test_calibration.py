"""Tests for calibration utilities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from govmodel.calibration import (
    ConformalSets,
    conformal_thresholds,
    ece_from_log,
    expected_calibration_error,
    find_threshold_f1,
    reliability_curve,
)


def test_ece_zero_for_calibrated_signal():
    rng = np.random.default_rng(7)
    probs = rng.random(2000)
    labels = (rng.random(2000) < probs).astype(np.int32)
    assert expected_calibration_error(probs, labels) < 0.05


def test_ece_high_for_overconfident():
    probs = np.full(1000, 0.99)
    labels = np.array([0] * 500 + [1] * 500)
    assert expected_calibration_error(probs, labels) > 0.4


def test_reliability_curve_has_n_bins_entries():
    p = np.linspace(0.0, 1.0, 500)
    y = (p > 0.5).astype(np.int32)
    curve = reliability_curve(p, y, n_bins=5)
    assert len(curve) == 5


def test_conformal_threshold_for_each_class():
    rng = np.random.default_rng(0)
    probs = rng.random((400, 3))
    labels = (rng.random((400, 3)) < 0.3).astype(np.int32)
    cs = conformal_thresholds(probs, labels, alpha=0.10)
    assert isinstance(cs, ConformalSets)
    assert cs.thresholds.shape == (3,)


def test_find_threshold_f1_recovers_optimal():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (p > 0.55).astype(np.int32)
    t, f1 = find_threshold_f1(p, y)
    assert f1 > 0.95
    assert 0.5 < t < 0.6


def test_ece_from_log_empty_when_no_feedback(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": "2026-05-10T12:00:00Z", "request_id": "a",
            "top_label": "klacht", "top_score": 0.8,
        }) + "\n")
    out = ece_from_log(log)
    assert out["n_with_feedback"] == 0
    assert out["ece"] == 0.0


def test_ece_from_log_with_feedback(tmp_path: Path):
    log = tmp_path / "predictions.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        # (top_label, top_score, user_label) — for ECE we care about correctness
        ("a", "klacht", 0.9, "klacht"),
        ("b", "klacht", 0.8, "klacht"),
        ("c", "klacht", 0.7, "bezwaar"),
        ("d", "klacht", 0.4, "bezwaar"),
        ("e", "klacht", 0.6, "klacht"),
    ]
    with log.open("w", encoding="utf-8") as f:
        for rid, top, score, _user in rows:
            f.write(json.dumps({
                "ts": "2026-05-10T12:00:00Z", "request_id": rid,
                "top_label": top, "top_score": score,
            }) + "\n")
        for rid, _top, _score, user in rows:
            f.write(json.dumps({"request_id": rid, "user_label": user,
                                "feedback_only": True}) + "\n")
    out = ece_from_log(log)
    assert out["n_with_feedback"] == 5
    assert "ece" in out
    assert "agreement_rate" in out


def test_temperature_scale_reduces_overconfidence():
    pytest.importorskip("torch")
    from govmodel.calibration import temperature_scale

    rng = np.random.default_rng(0)
    n, c = 800, 3
    true_p = rng.random((n, c)) * 0.6 + 0.2
    labels = (rng.random((n, c)) < true_p).astype(np.int32)
    base_logits = np.log(true_p / (1 - true_p + 1e-9))
    overconf = base_logits / 0.3
    def sig(x):
        return 1 / (1 + np.exp(-x))
    pre = expected_calibration_error(sig(overconf), labels)
    T = temperature_scale(overconf, labels)
    post = expected_calibration_error(sig(overconf / T), labels)
    assert post <= pre
    assert T > 1.0
