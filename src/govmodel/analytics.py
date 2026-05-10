"""Analytics aggregations over the prediction JSONL log.

Pure stdlib — readable on any host with a checkpoint and a log file. Used by
the FastAPI dashboard and easy to reuse from a notebook.

The aggregator is deliberately stateless: it re-reads the JSONL each call,
which is fine for the volumes we're dealing with (a single municipality
processes O(10⁴) letters/year). For higher throughput, swap for DuckDB.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _default_log_path() -> Path:
    return Path(os.environ.get("GOVMODEL_PREDICTION_LOG", "logs/predictions.jsonl"))


def _read_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a corrupted line rather than crash the dashboard


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


def summary(path: Path | None = None, *, window_hours: float = 24.0) -> dict[str, Any]:
    """Top-level dashboard payload.

    `window_hours`: how far back the time-bucketed aggregations look.
    """
    path = Path(path) if path else _default_log_path()
    records = list(_read_records(path))
    total = len(records)

    if total == 0:
        return _empty_summary(path)

    cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
    label_counts: Counter[str] = Counter()
    above_thresh_counts: Counter[str] = Counter()
    abstain_count = 0
    latencies: list[float] = []
    score_buckets = {"<0.2": 0, "0.2-0.5": 0, "0.5-0.8": 0, ">=0.8": 0}
    hourly_buckets: dict[str, int] = defaultdict(int)
    label_hourly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    confusion_pairs: Counter[tuple[str, str]] = Counter()
    recent: list[dict[str, Any]] = []
    in_window = 0
    feedback_correct = 0
    feedback_wrong = 0

    for r in records:
        top_label = r.get("top_label", "?")
        top_score = float(r.get("top_score", 0.0))
        ts = _parse_ts(r.get("ts", ""))

        label_counts[top_label] += 1
        if r.get("abstain"):
            abstain_count += 1
        latencies.append(float(r.get("latency_ms", 0.0)))

        if top_score < 0.2:
            score_buckets["<0.2"] += 1
        elif top_score < 0.5:
            score_buckets["0.2-0.5"] += 1
        elif top_score < 0.8:
            score_buckets["0.5-0.8"] += 1
        else:
            score_buckets[">=0.8"] += 1

        for lbl, _ in r.get("above_threshold", []):
            above_thresh_counts[lbl] += 1

        if ts and ts >= cutoff:
            in_window += 1
            bucket = ts.strftime("%Y-%m-%d %H:00")
            hourly_buckets[bucket] += 1
            label_hourly[bucket][top_label] += 1

        # If a user_label is present (active-learning feedback), track agreement.
        user_label = r.get("user_label")
        if user_label:
            if user_label == top_label:
                feedback_correct += 1
            else:
                feedback_wrong += 1
                confusion_pairs[(top_label, user_label)] += 1

        recent.append({
            "ts": r.get("ts"),
            "request_id": r.get("request_id"),
            "top_label": top_label,
            "top_score": top_score,
            "abstain": r.get("abstain", False),
            "above_threshold": r.get("above_threshold", []),
            "latency_ms": float(r.get("latency_ms", 0.0)),
        })

    recent.sort(key=lambda x: x["ts"] or "", reverse=True)

    latencies.sort()
    n = len(latencies)
    p50 = latencies[n // 2] if n else 0.0
    p95 = latencies[min(n - 1, int(n * 0.95))] if n else 0.0

    return {
        "log_path": str(path),
        "total_predictions": total,
        "in_window": in_window,
        "window_hours": window_hours,
        "abstain_count": abstain_count,
        "abstain_rate": abstain_count / total if total else 0.0,
        "feedback_correct": feedback_correct,
        "feedback_wrong": feedback_wrong,
        "feedback_accuracy": (
            feedback_correct / (feedback_correct + feedback_wrong)
            if (feedback_correct + feedback_wrong) else None
        ),
        "label_counts": dict(label_counts.most_common()),
        "above_threshold_counts": dict(above_thresh_counts.most_common()),
        "score_buckets": score_buckets,
        "latency_ms": {"p50": p50, "p95": p95, "n": n},
        "hourly": [
            {
                "bucket": bucket,
                "count": hourly_buckets[bucket],
                "by_label": dict(label_hourly[bucket]),
            }
            for bucket in sorted(hourly_buckets)
        ],
        "top_confusions": [
            {"predicted": p, "user_label": u, "n": n}
            for (p, u), n in confusion_pairs.most_common(10)
        ],
        "recent": recent[:25],
    }


def _empty_summary(path: Path) -> dict[str, Any]:
    return {
        "log_path": str(path),
        "total_predictions": 0,
        "in_window": 0,
        "window_hours": 24.0,
        "abstain_count": 0,
        "abstain_rate": 0.0,
        "feedback_correct": 0,
        "feedback_wrong": 0,
        "feedback_accuracy": None,
        "label_counts": {},
        "above_threshold_counts": {},
        "score_buckets": {"<0.2": 0, "0.2-0.5": 0, "0.5-0.8": 0, ">=0.8": 0},
        "latency_ms": {"p50": 0.0, "p95": 0.0, "n": 0},
        "hourly": [],
        "top_confusions": [],
        "recent": [],
    }
