"""Dataset hygiene for govmodel: validation, dedup, leakage, splits.

govmodel-specific bits:
  - validates against the `LabeledExample` schema (multilabel `labels` list,
    not School's single `tag` field).
  - default stratification axis is the *primary* label — a stable first
    label per row — so rare classes like `informatieverzoek_3_11` aren't
    starved by random splits.

Everything is stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from govmodel.training.labels import LABEL_NAMES

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


REQUIRED_FIELDS = {"text", "labels"}
OPTIONAL_FIELDS = {"source", "id", "metadata", "split"}


@dataclass(frozen=True)
class ValidationIssue:
    row_index: int
    kind: str
    detail: str


def validate_examples(
    rows: Iterable[dict],
    *,
    allowed_labels: Sequence[str] | None = None,
    strict: bool = False,
    min_text_length: int = 10,
) -> list[ValidationIssue]:
    """Validate JSONL rows against the `LabeledExample`-shaped schema.

    `strict=True` rejects unknown top-level fields. `min_text_length`
    catches degenerate one-word "answers" that pollute the train set.
    """
    issues: list[ValidationIssue] = []
    allowed = set(allowed_labels or LABEL_NAMES)
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(ValidationIssue(i, "bad_type", f"row is {type(row).__name__}"))
            continue
        for f in REQUIRED_FIELDS:
            if f not in row:
                issues.append(ValidationIssue(i, "missing_field", f))
        text = row.get("text")
        if text is not None:
            if not isinstance(text, str):
                issues.append(ValidationIssue(i, "bad_type", f"text is {type(text).__name__}"))
            elif not text.strip():
                issues.append(ValidationIssue(i, "empty_field", "text"))
            elif len(text.strip()) < min_text_length:
                issues.append(ValidationIssue(i, "text_too_short", f"len={len(text.strip())}"))
        labels = row.get("labels")
        if labels is not None:
            if not isinstance(labels, list) or not labels:
                issues.append(ValidationIssue(i, "labels_empty", repr(labels)))
            else:
                for j, lbl in enumerate(labels):
                    if lbl not in allowed:
                        issues.append(ValidationIssue(i, "label_invalid",
                                                      f"labels[{j}]={lbl!r}"))
        if strict:
            extras = set(row.keys()) - REQUIRED_FIELDS - OPTIONAL_FIELDS
            if extras:
                issues.append(ValidationIssue(i, "unknown_field", ",".join(sorted(extras))))
    return issues


def issues_summary(issues: Sequence[ValidationIssue]) -> dict[str, int]:
    return dict(Counter(i.kind for i in issues))


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def label_distribution(rows: Sequence[dict]) -> dict[str, int]:
    """Count each label across all rows. Multilabel rows contribute multiple counts."""
    counts: Counter[str] = Counter()
    for r in rows:
        for lbl in r.get("labels", []) or []:
            counts[str(lbl)] += 1
    return dict(counts.most_common())


def multilabel_distribution(rows: Sequence[dict]) -> dict[str, int]:
    """Count how many rows fire 1 / 2 / 3+ labels."""
    counts: Counter[str] = Counter()
    for r in rows:
        n = len(r.get("labels", []) or [])
        if n == 0:
            key = "0_no_labels"
        elif n == 1:
            key = "1_singletag"
        elif n == 2:
            key = "2_pair"
        else:
            key = "3plus"
        counts[key] += 1
    return dict(counts)


def cross_distribution(
    rows: Sequence[dict], *, source_key: str = "source"
) -> dict[str, dict[str, int]]:
    """Per-source label breakdown."""
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        src = str(r.get(source_key, "unknown"))
        for lbl in r.get("labels", []) or []:
            out[src][str(lbl)] += 1
    return {k: dict(v) for k, v in out.items()}


def balance_report(rows: Sequence[dict]) -> dict[str, Any]:
    labels = label_distribution(rows)
    multi = multilabel_distribution(rows)
    sources = cross_distribution(rows)
    label_imbalance = (
        max(labels.values()) / min(labels.values()) if labels else 0
    )
    return {
        "n": len(rows),
        "label_distribution": labels,
        "multilabel_distribution": multi,
        "source_distribution": sources,
        "label_imbalance_ratio": round(label_imbalance, 2),
        "n_labels_seen": len(labels),
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


_WHITESPACE = re.compile(r"\s+")


def _normalise_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text.lower().strip())


def exact_dedupe(rows: Iterable[dict], *, fields: Sequence[str] = ("text",)) -> list[dict]:
    """Hash-based exact-duplicate removal across the chosen fields."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        key = "|".join(_normalise_text(str(r.get(f, ""))) for f in fields)
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


def _shingles(text: str, k: int = 5) -> set[str]:
    norm = _normalise_text(text)
    if len(norm) < k:
        return {norm}
    return {norm[i:i + k] for i in range(len(norm) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def near_dedupe(
    rows: Sequence[dict],
    *,
    threshold: float = 0.85,
    field: str = "text",
    shingle_k: int = 5,
) -> list[dict]:
    """Greedy near-dedup via character k-shingles + Jaccard. O(N²)."""
    kept: list[tuple[set[str], dict]] = []
    out: list[dict] = []
    for r in rows:
        sh = _shingles(str(r.get(field, "")), k=shingle_k)
        dup = False
        for prev_sh, _ in kept:
            if jaccard(sh, prev_sh) >= threshold:
                dup = True
                break
        if dup:
            continue
        kept.append((sh, r))
        out.append(r)
    return out


def leakage_between(
    train: Sequence[dict],
    test: Sequence[dict],
    *,
    threshold: float = 0.85,
    field: str = "text",
    shingle_k: int = 5,
) -> list[tuple[int, int, float]]:
    """Returns (test_idx, train_idx, jaccard) for any test row near a train row.

    This is the gate that prevents synthetic→test leakage from inflating
    macro-F1 in benchmark.py.
    """
    train_shingles = [_shingles(str(r.get(field, "")), k=shingle_k) for r in train]
    matches: list[tuple[int, int, float]] = []
    for ti, t_row in enumerate(test):
        t_sh = _shingles(str(t_row.get(field, "")), k=shingle_k)
        for tri, tr_sh in enumerate(train_shingles):
            j = jaccard(t_sh, tr_sh)
            if j >= threshold:
                matches.append((ti, tri, j))
                break
    return matches


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _primary_label(row: dict) -> str:
    labels = row.get("labels") or []
    return str(labels[0]) if labels else "__empty__"


def stratified_split(
    rows: Sequence[dict],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    by: str = "primary_label",
) -> dict[str, list[dict]]:
    """Stratified split keyed by the row's primary (first) label.

    Rare classes like `informatieverzoek_3_11` would otherwise miss test
    coverage under random splitting.
    """
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("ratios must sum to 1.0")
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if by == "primary_label":
            k = _primary_label(r)
        elif by == "source":
            k = str(r.get("source", "unknown"))
        else:
            raise ValueError(f"unknown by={by!r}")
        buckets[k].append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for _k, group in buckets.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
