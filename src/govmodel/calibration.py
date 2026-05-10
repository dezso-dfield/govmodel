"""Confidence calibration utilities.

Mirrors School's `_calibration.py`: temperature scaling, ECE, reliability
curves, conformal sets, and an F1-optimal threshold search. The two repos
should converge on a shared library once this stabilises — for now they
diverge intentionally so neither imports across project boundaries.

The analytics dashboard endpoint exposes the live ECE on logged
predictions vs. user feedback, so operators can see when the model's
sigmoids stop meaning what they say.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def temperature_scale(
    logits: np.ndarray, labels: np.ndarray, *,
    max_iter: int = 200, lr: float = 0.01, init: float = 1.0,
) -> float:
    """LBFGS-fit scalar T minimising BCE on the held-out set. Apply with
    `calibrated = logits / T`."""
    import torch  # lazy

    logits_t = torch.from_numpy(logits.astype(np.float32))
    labels_t = torch.from_numpy(labels.astype(np.float32))
    log_T = torch.tensor([float(np.log(init))], requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        T = log_T.exp()
        loss = bce(logits_t / T, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.detach().exp().item())


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 15
) -> float:
    p = probs.flatten()
    y = labels.flatten().astype(np.float32)
    if p.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = p.size
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / total) * abs(acc - conf)
    return float(ece)


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
) -> list[dict[str, float]]:
    p = probs.flatten()
    y = labels.flatten().astype(np.float32)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            out.append({"lo": float(lo), "hi": float(hi), "n": 0,
                        "confidence": float((lo + hi) / 2), "accuracy": 0.0})
            continue
        out.append({
            "lo": float(lo),
            "hi": float(hi),
            "n": int(mask.sum()),
            "confidence": float(p[mask].mean()),
            "accuracy": float(y[mask].mean()),
        })
    return out


@dataclass(frozen=True)
class ConformalSets:
    thresholds: np.ndarray
    alpha: float
    n_calib: int

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return probs >= self.thresholds[None, :]


def conformal_thresholds(
    probs: np.ndarray, labels: np.ndarray, *, alpha: float = 0.10
) -> ConformalSets:
    if probs.ndim != 2 or labels.shape != probs.shape:
        raise ValueError(f"shape mismatch probs={probs.shape} labels={labels.shape}")
    n, c = probs.shape
    thresholds = np.zeros(c, dtype=np.float64)
    for k in range(c):
        pos = probs[labels[:, k] == 1, k]
        if pos.size == 0:
            thresholds[k] = 0.5
            continue
        thresholds[k] = float(np.quantile(pos, alpha))
    return ConformalSets(thresholds=thresholds, alpha=alpha, n_calib=n)


def find_threshold_f1(
    probs: np.ndarray, labels: np.ndarray, *, grid: Sequence[float] | None = None
) -> tuple[float, float]:
    if grid is None:
        grid = np.arange(0.05, 0.95 + 1e-9, 0.025).tolist()
    best_t, best_f1 = 0.5, -1.0
    y = labels.astype(np.int32)
    for t in grid:
        pred = (probs >= t).astype(np.int32)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        if tp == 0:
            f1 = 0.0
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


def ece_from_log(
    log_path,
    *,
    threshold_field: str = "top_score",
    pred_field: str = "top_label",
    label_field: str = "user_label",
) -> dict:
    """Compute ECE on (top_score, top_label == user_label) pairs in the
    prediction log. Used by the analytics dashboard to show the live
    calibration delta as feedback comes in.
    """
    import json
    from pathlib import Path

    path = Path(log_path)
    if not path.exists():
        return {"n_with_feedback": 0, "ece": 0.0}

    # First pass: collect (request_id → top_score, top_label) and (request_id → user_label)
    top_by_rid: dict[str, dict] = {}
    user_by_rid: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("request_id")
            if not rid:
                continue
            if r.get(label_field):
                user_by_rid[rid] = r[label_field]
            if pred_field in r and threshold_field in r:
                top_by_rid[rid] = r

    matched_probs: list[float] = []
    matched_labels: list[int] = []
    for rid, user_label in user_by_rid.items():
        if rid not in top_by_rid:
            continue
        pred = top_by_rid[rid][pred_field]
        score = float(top_by_rid[rid][threshold_field])
        matched_probs.append(score)
        matched_labels.append(1 if pred == user_label else 0)

    if not matched_probs:
        return {"n_with_feedback": 0, "ece": 0.0}
    probs_arr = np.array(matched_probs)
    labels_arr = np.array(matched_labels)
    return {
        "n_with_feedback": len(matched_probs),
        "ece": round(expected_calibration_error(probs_arr, labels_arr, n_bins=10), 4),
        "agreement_rate": round(float(labels_arr.mean()), 4),
        "reliability": reliability_curve(probs_arr, labels_arr, n_bins=8),
    }
