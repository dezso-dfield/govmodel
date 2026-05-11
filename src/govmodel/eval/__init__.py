"""Slice-aware evaluation harness for govmodel.

The legacy `scripts/benchmark.py` runs a single eval against one or more
JSONL files and dumps a JSON. This harness wraps that with:

  - per-tag F1 + optimal threshold + Δ-vs-default
  - macro/micro-F1, ECE, reliability curves
  - JSON + Markdown reports
  - regression gate (`pre-commit` / CI exits 1 when macro-F1 drops)

`harness.py` is pure-Python+numpy. Drivers in `scripts/evaluate_slices.py`
do the model loading and dispatch.
"""
from govmodel.eval.harness import (
    EvalReport,
    SliceReport,
    TagMetrics,
    evaluate_slice,
    regression_gate,
)

__all__ = [
    "EvalReport",
    "SliceReport",
    "TagMetrics",
    "evaluate_slice",
    "regression_gate",
]
