"""Hard-negative mining over the prediction JSONL log.

The Gradio demo and the FastAPI service both append every prediction to
`logs/predictions.jsonl`. This module turns that operational log into a
labelling queue: the predictions most likely to teach the next model
something it doesn't already know.

Three signals, each producing a ranked list:

  - **Low confidence** — top score in a no-man's-land between abstain
    and "obvious". These are the cases where a human label flips the
    model from "I don't know" to a useful training example.

  - **Conflicted multilabel** — multiple labels above threshold AND
    close to each other in probability. Likely cases where the synthetic
    set didn't include the specific tag combination.

  - **User-disagreement** — feedback rows where the user-supplied label
    differs from the top model label. Highest-signal examples for the
    next round; they're already labelled by a human.

The output is a JSONL of candidates with a `_mine_reason` and a
`_mine_score` you can feed straight into the labelling tool.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ABSTAIN_THRESHOLD = 0.20


@dataclass(frozen=True)
class MiningConfig:
    abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD
    low_conf_window: tuple[float, float] = (0.20, 0.50)
    conflict_margin: float = 0.15        # two labels within this margin → conflicted
    multilabel_min_above: float = 0.40   # second label must also be above this
    user_disagreement_only: bool = False


@dataclass
class Candidate:
    record: dict
    reason: str
    score: float                          # 0..1, higher = more useful to label


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _resolve_user_labels(records: list[dict]) -> dict[str, str]:
    """The API can emit two row kinds: a full prediction (with `request_id`
    and `top_label`) and a `feedback_only=True` follow-up with a
    `user_label`. We resolve the latest user label per request_id."""
    latest: dict[str, str] = {}
    for r in records:
        rid = r.get("request_id")
        if rid and r.get("user_label"):
            latest[rid] = r["user_label"]
    return latest


_DEFAULT_MINING_CONFIG = MiningConfig()


def mine(
    log_path: Path,
    cfg: MiningConfig | None = None,
    *,
    limit: int | None = None,
) -> list[Candidate]:
    if cfg is None:
        cfg = _DEFAULT_MINING_CONFIG
    records = list(_read_jsonl(log_path))
    if not records:
        return []
    user_labels = _resolve_user_labels(records)

    # Index by request_id so we don't double-count the prediction + its feedback row.
    by_request: dict[str, dict] = {}
    for r in records:
        if r.get("feedback_only"):
            continue
        rid = r.get("request_id")
        if rid:
            by_request[rid] = r

    out: list[Candidate] = []
    for rid, r in by_request.items():
        top_score = float(r.get("top_score", 0.0))
        top_label = r.get("top_label", "")
        above = r.get("above_threshold", []) or []
        user_label = user_labels.get(rid)

        # Reason 1: user disagreement (always emit if present)
        if user_label and user_label != top_label:
            out.append(Candidate(
                record={**r, "user_label": user_label, "_mine_reason": "user_disagreement"},
                reason="user_disagreement",
                # Disagreements are highest priority. Confidence delta amplifies.
                score=min(1.0, 0.6 + top_score * 0.4),
            ))
            continue

        if cfg.user_disagreement_only:
            continue

        # Reason 2: low confidence in the active band
        if cfg.low_conf_window[0] <= top_score <= cfg.low_conf_window[1]:
            # Score: peak inside the window, fading at edges
            mid = sum(cfg.low_conf_window) / 2.0
            half = (cfg.low_conf_window[1] - cfg.low_conf_window[0]) / 2.0
            distance = abs(top_score - mid) / max(half, 1e-6)
            out.append(Candidate(
                record={**r, "_mine_reason": "low_confidence"},
                reason="low_confidence",
                score=0.55 * (1.0 - distance),
            ))
            continue

        # Reason 3: conflicted multilabel (two top labels close together)
        if len(above) >= 2:
            try:
                first = float(above[0][1])
                second = float(above[1][1])
            except (ValueError, TypeError, IndexError):
                continue
            margin = first - second
            if margin <= cfg.conflict_margin and second >= cfg.multilabel_min_above:
                out.append(Candidate(
                    record={**r, "_mine_reason": "conflicted_multilabel"},
                    reason="conflicted_multilabel",
                    score=0.50 + (cfg.conflict_margin - margin) * 0.5,
                ))

    out.sort(key=lambda c: -c.score)
    if limit is not None:
        out = out[:limit]
    return out


def to_label_queue(candidates: Iterable[Candidate], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        record_keys = ("ts", "request_id", "top_label", "top_score", "above_threshold",
                       "user_label", "_mine_reason", "_mine_score")
        for c in candidates:
            row = {k: c.record.get(k) for k in record_keys}
            row["_mine_score"] = round(c.score, 4)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def stats(candidates: list[Candidate]) -> dict[str, object]:
    by_reason: dict[str, int] = defaultdict(int)
    for c in candidates:
        by_reason[c.reason] += 1
    return {
        "n_total": len(candidates),
        "by_reason": dict(by_reason),
        "max_score": max((c.score for c in candidates), default=0.0),
        "min_score": min((c.score for c in candidates), default=0.0),
    }
