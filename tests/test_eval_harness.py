"""Tests for the sliced evaluation harness + regression gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from govmodel.eval import EvalReport, evaluate_slice, regression_gate


def _make_clean_slice(n: int = 200, c: int = 9, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = (rng.random((n, c)) < 0.25).astype(np.int32)
    probs = labels.astype(np.float32) * 0.75 + rng.random((n, c)) * 0.25
    return probs, labels


def test_evaluate_slice_basic_shape():
    probs, labels = _make_clean_slice()
    tags = [f"t{i}" for i in range(9)]
    rep = evaluate_slice("gold", probs, labels, tags)
    assert rep.n == 200
    assert rep.macro_f1 > 0.5
    assert len(rep.per_tag) == 9
    for t in rep.per_tag:
        assert 0.0 <= t.f1 <= 1.0
        assert 0.0 <= t.precision <= 1.0
        assert 0.0 <= t.recall <= 1.0


def test_evaluate_slice_rejects_shape_mismatch():
    probs = np.zeros((10, 3))
    labels = np.zeros((10, 2))
    with pytest.raises(ValueError):
        evaluate_slice("x", probs, labels, ["a", "b", "c"])


def test_evaluate_slice_rejects_wrong_tag_count():
    probs = np.zeros((10, 3))
    labels = np.zeros((10, 3))
    with pytest.raises(ValueError):
        evaluate_slice("x", probs, labels, ["a", "b"])


def test_report_markdown_and_json_written(tmp_path: Path):
    probs, labels = _make_clean_slice()
    rep = evaluate_slice("handwritten_realistic", probs, labels,
                        [f"t{i}" for i in range(9)])
    report = EvalReport(model_id="test-v0.1", slices=[rep])
    md = report.to_markdown()
    assert "# Evaluation report" in md
    assert "## `handwritten_realistic` — per-tag breakdown" in md
    json_path, md_path = report.write(tmp_path)
    parsed = json.loads(json_path.read_text())
    assert parsed["model_id"] == "test-v0.1"
    assert "macro_f1_by_slice" in parsed
    assert md_path.read_text().startswith("# Evaluation report")


def test_regression_gate_passes_when_stable():
    probs, labels = _make_clean_slice()
    rep = evaluate_slice("handwritten_realistic", probs, labels,
                        [f"t{i}" for i in range(9)])
    report = EvalReport(model_id="m", slices=[rep])
    baseline = {"handwritten_realistic": rep.macro_f1}
    verdict = regression_gate(report, baseline, max_drop=0.05)
    assert verdict["ok"]


def test_regression_gate_fails_on_drop():
    probs, labels = _make_clean_slice()
    rep = evaluate_slice("handwritten_realistic", probs, labels,
                        [f"t{i}" for i in range(9)])
    report = EvalReport(model_id="m", slices=[rep])
    baseline = {"handwritten_realistic": rep.macro_f1 + 0.20}
    verdict = regression_gate(report, baseline, max_drop=0.05)
    assert not verdict["ok"]
    assert verdict["failed_slices"]


def test_regression_gate_flags_missing_required():
    probs, labels = _make_clean_slice()
    rep = evaluate_slice("robustness_typos_3pct", probs, labels,
                        [f"t{i}" for i in range(9)])
    report = EvalReport(model_id="m", slices=[rep])
    verdict = regression_gate(report, None,
                              require_slices=("handwritten_realistic",))
    assert not verdict["ok"]
    assert verdict["missing_required_slices"] == ["handwritten_realistic"]
