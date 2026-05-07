"""Dataset-voorbereiding voor de Awb-type-classifier.

Leest JSONL met LabeledExample-records, splitst in train/val/test, en
bereidt voor op HuggingFace datasets-formaat.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path

from govmodel.schemas import LabeledExample
from govmodel.training.labels import encode_labels

logger = logging.getLogger(__name__)


def load_examples(path: Path) -> Iterator[LabeledExample]:
    """Yield LabeledExample-records uit een JSONL-bestand."""
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield LabeledExample.model_validate_json(line)
            except Exception as e:  # noqa: BLE001
                logger.warning("Skip regel %d: %s", line_num, e)


def load_all_examples(paths: list[Path]) -> list[LabeledExample]:
    examples: list[LabeledExample] = []
    for p in paths:
        examples.extend(load_examples(p))
    logger.info("Geladen: %d voorbeelden uit %d bestanden", len(examples), len(paths))
    return examples


def split_train_val_test(
    examples: list[LabeledExample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[LabeledExample], list[LabeledExample], list[LabeledExample]]:
    """Random shuffle-split met seed (legacy, niet aanbevolen)."""
    if not 0 < train_ratio + val_ratio < 1:
        raise ValueError("train_ratio + val_ratio moet tussen 0 en 1 liggen")

    rng = random.Random(seed)
    items = list(examples)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def stratified_split(
    examples: list[LabeledExample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[LabeledExample], list[LabeledExample], list[LabeledExample]]:
    """Per-label stratified split.

    Voor multilabel-voorbeelden wordt het *minst frequente* label gebruikt
    als bucket-key, zodat zeldzame labels evenredig over splits verdeeld worden.
    Resultaat: elke split bevat (afgerond) hetzelfde aandeel per label, geen
    label kan toevallig leeg blijven in val of test.
    """
    if not 0 < train_ratio + val_ratio < 1:
        raise ValueError("train_ratio + val_ratio moet tussen 0 en 1 liggen")

    rng = random.Random(seed)

    # Globale label-frequentie voor 'rarest' tiebreak bij multilabel
    label_counts: dict[str, int] = {}
    for ex in examples:
        for lbl in ex.labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # Bucket per primary label (rarest bij multilabel)
    buckets: dict[str, list[LabeledExample]] = {}
    for ex in examples:
        if not ex.labels:
            key = "__none__"
        else:
            key = min(ex.labels, key=lambda lbl: label_counts.get(lbl, 0))
        buckets.setdefault(key, []).append(ex)

    train: list[LabeledExample] = []
    val: list[LabeledExample] = []
    test: list[LabeledExample] = []
    for key in sorted(buckets):
        items = buckets[key]
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def to_hf_dict(examples: list[LabeledExample]) -> dict[str, list]:
    """Converteer naar dict-formaat voor HuggingFace Dataset.from_dict()."""
    return {
        "text": [ex.text for ex in examples],
        "labels": [encode_labels(ex.labels) for ex in examples],
        "id": [ex.id for ex in examples],
        "source": [ex.source for ex in examples],
    }


def save_split(examples: list[LabeledExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")


def label_distribution(examples: list[LabeledExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in examples:
        for label in ex.labels:
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def source_distribution(examples: list[LabeledExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex.source] = counts.get(ex.source, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))
