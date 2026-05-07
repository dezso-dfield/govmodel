from __future__ import annotations

import pytest

from govmodel.training.dataset import (
    label_distribution,
    source_distribution,
    split_train_val_test,
)
from govmodel.training.labels import (
    ID_TO_LABEL,
    LABEL_NAMES,
    LABEL_TO_ID,
    decode_labels,
    encode_labels,
)
from govmodel.schemas import LabeledExample


def test_label_names_are_unique():
    assert len(LABEL_NAMES) == len(set(LABEL_NAMES))


def test_label_to_id_consistent():
    for i, label in enumerate(LABEL_NAMES):
        assert LABEL_TO_ID[label] == i
        assert ID_TO_LABEL[i] == label


def test_encode_single_label():
    vec = encode_labels(["bezwaar"])
    assert vec[LABEL_TO_ID["bezwaar"]] == 1.0
    assert sum(vec) == 1.0


def test_encode_multilabel():
    vec = encode_labels(["bezwaar", "klacht"])
    assert vec[LABEL_TO_ID["bezwaar"]] == 1.0
    assert vec[LABEL_TO_ID["klacht"]] == 1.0
    assert sum(vec) == 2.0


def test_encode_empty():
    vec = encode_labels([])
    assert vec == [0.0] * len(LABEL_NAMES)


def test_encode_unknown_label_ignored():
    vec = encode_labels(["bezwaar", "niet_bestaand"])  # type: ignore[list-item]
    assert vec[LABEL_TO_ID["bezwaar"]] == 1.0
    assert sum(vec) == 1.0


def test_decode_threshold():
    probs = [0.0] * len(LABEL_NAMES)
    probs[LABEL_TO_ID["aanvraag"]] = 0.9
    probs[LABEL_TO_ID["bezwaar"]] = 0.3
    decoded = decode_labels(probs, threshold=0.5)
    assert decoded == ["aanvraag"]


def test_decode_multilabel():
    probs = [0.0] * len(LABEL_NAMES)
    probs[LABEL_TO_ID["aanvraag"]] = 0.9
    probs[LABEL_TO_ID["bezwaar"]] = 0.7
    decoded = decode_labels(probs, threshold=0.5)
    assert set(decoded) == {"aanvraag", "bezwaar"}


def _make_example(idx: int, label: str = "aanvraag", source: str = "synthetic") -> LabeledExample:
    return LabeledExample(
        id=f"id-{idx}",
        text=f"voorbeeldtekst {idx}",
        source=source,  # type: ignore[arg-type]
        labels=[label],  # type: ignore[list-item]
    )


def test_split_train_val_test_sizes():
    examples = [_make_example(i) for i in range(100)]
    train, val, test = split_train_val_test(examples, train_ratio=0.7, val_ratio=0.15, seed=1)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_split_is_reproducible():
    examples = [_make_example(i) for i in range(50)]
    a = split_train_val_test(examples, seed=42)
    b = split_train_val_test(examples, seed=42)
    assert [e.id for e in a[0]] == [e.id for e in b[0]]


def test_split_invalid_ratios():
    with pytest.raises(ValueError):
        split_train_val_test([], train_ratio=0.5, val_ratio=0.6)


def test_label_distribution():
    examples = [
        _make_example(0, label="aanvraag"),
        _make_example(1, label="aanvraag"),
        _make_example(2, label="bezwaar"),
    ]
    dist = label_distribution(examples)
    assert dist == {"aanvraag": 2, "bezwaar": 1}


def test_source_distribution():
    examples = [
        _make_example(0, source="synthetic"),
        _make_example(1, source="synthetic"),
        _make_example(2, source="rechtspraak"),
    ]
    dist = source_distribution(examples)
    assert dist == {"synthetic": 2, "rechtspraak": 1}
