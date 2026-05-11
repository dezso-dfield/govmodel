"""Slice-aware evaluation harness.

Reuses `govmodel.calibration.{find_threshold_f1, expected_calibration_error,
reliability_curve}` so threshold tuning + ECE share one implementation
with the rest of the package.

Pure-Python + numpy. Model loading + inference happens in driver scripts;
this module is fed `(probs, labels)` arrays so it stays testable without
a GPU.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from govmodel.calibration import (
    expected_calibration_error,
    find_threshold_f1,
    reliability_curve,
)


def _confusion_counts(pred: np.ndarray, gold: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    tn = int(((pred == 0) & (gold == 0)).sum())
    return tp, fp, fn, tn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


@dataclass
class TagMetrics:
    tag: str
    n_positives: int
    precision: float
    recall: float
    f1: float
    threshold: float
    f1_at_default: float


@dataclass
class SliceReport:
    slice_name: str
    n: int
    macro_f1: float
    micro_f1: float
    ece: float
    reliability: list[dict[str, float]]
    per_tag: list[TagMetrics]
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_slice(
    slice_name: str,
    probs: np.ndarray,
    labels: np.ndarray,
    tags: Sequence[str],
    *,
    default_threshold: float = 0.5,
) -> SliceReport:
    """One labelled slice → one `SliceReport`."""
    if probs.ndim != 2 or labels.shape != probs.shape:
        raise ValueError(f"shape mismatch probs={probs.shape} labels={labels.shape}")
    if len(tags) != probs.shape[1]:
        raise ValueError(f"expected {probs.shape[1]} tags, got {len(tags)}")

    per_tag: list[TagMetrics] = []
    macro_f1 = 0.0
    tp_total = fp_total = fn_total = 0

    for k, tag in enumerate(tags):
        best_t, best_f1 = find_threshold_f1(probs[:, k], labels[:, k])
        pred = (probs[:, k] >= best_t).astype(np.int32)
        gold = labels[:, k].astype(np.int32)
        tp, fp, fn, _ = _confusion_counts(pred, gold)
        p, r, _f1 = _prf(tp, fp, fn)

        pred_def = (probs[:, k] >= default_threshold).astype(np.int32)
        tp_d, fp_d, fn_d, _ = _confusion_counts(pred_def, gold)
        _, _, f1_def = _prf(tp_d, fp_d, fn_d)

        per_tag.append(TagMetrics(
            tag=tag, n_positives=int(gold.sum()),
            precision=p, recall=r, f1=best_f1,
            threshold=best_t, f1_at_default=f1_def,
        ))
        macro_f1 += best_f1
        tp_total += tp
        fp_total += fp
        fn_total += fn

    n_tags = max(1, len(tags))
    macro_f1 /= n_tags
    _, _, micro_f1 = _prf(tp_total, fp_total, fn_total)
    ece = expected_calibration_error(probs, labels)

    return SliceReport(
        slice_name=slice_name,
        n=int(probs.shape[0]),
        macro_f1=round(float(macro_f1), 4),
        micro_f1=round(float(micro_f1), 4),
        ece=round(float(ece), 4),
        reliability=reliability_curve(probs, labels),
        per_tag=per_tag,
        tags=list(tags),
    )


@dataclass
class EvalReport:
    model_id: str
    slices: list[SliceReport]

    @property
    def macro_f1_by_slice(self) -> dict[str, float]:
        return {s.slice_name: s.macro_f1 for s in self.slices}

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "slices": [s.to_dict() for s in self.slices],
            "macro_f1_by_slice": self.macro_f1_by_slice,
        }

    def write(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "report.json"
        md_path = out_dir / "report.md"
        json_path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False),
                             encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Evaluation report — `{self.model_id}`")
        lines.append("")
        lines.append("## Slice summary")
        lines.append("")
        lines.append("| Slice | N | Macro-F1 | Micro-F1 | ECE |")
        lines.append("|---|---:|---:|---:|---:|")
        for s in self.slices:
            lines.append(f"| `{s.slice_name}` | {s.n} | {s.macro_f1:.3f} | "
                         f"{s.micro_f1:.3f} | {s.ece:.3f} |")
        lines.append("")
        for s in self.slices:
            lines.append(f"## `{s.slice_name}` — per-tag breakdown")
            lines.append("")
            lines.append("| Tag | N⁺ | Threshold | F1 | F1@0.5 | Δ | P | R |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for t in s.per_tag:
                delta = t.f1 - t.f1_at_default
                lines.append(
                    f"| `{t.tag}` | {t.n_positives} | {t.threshold:.2f} | "
                    f"{t.f1:.3f} | {t.f1_at_default:.3f} | {delta:+.3f} | "
                    f"{t.precision:.3f} | {t.recall:.3f} |"
                )
            lines.append("")
        return "\n".join(lines)


def regression_gate(
    new: EvalReport,
    baseline: dict[str, float] | None,
    *,
    max_drop: float = 0.05,
    require_slices: Sequence[str] = ("handwritten_realistic",),
) -> dict[str, Any]:
    """Returns an `ok`-bearing verdict dict for the CI driver to act on.

    The default `require_slices=("handwritten_realistic",)` reflects the
    v0.1 reality: synthetic-test F1 over-rewards memorisation, so the
    handwritten slice is what we gate on.
    """
    new_scores = new.macro_f1_by_slice
    failed: list[dict[str, Any]] = []
    if baseline is not None:
        for slice_name, prev in baseline.items():
            if slice_name not in new_scores:
                continue
            drop = prev - new_scores[slice_name]
            if drop > max_drop:
                failed.append({
                    "slice": slice_name,
                    "prev": prev,
                    "new": new_scores[slice_name],
                    "drop": round(drop, 4),
                })
    missing = [s for s in require_slices if s not in new_scores]
    return {
        "ok": not failed and not missing,
        "max_allowed_drop": max_drop,
        "failed_slices": failed,
        "missing_required_slices": missing,
        "macro_f1_by_slice": new_scores,
    }
