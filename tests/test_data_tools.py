"""Tests for the data hygiene utilities."""
from __future__ import annotations

from pathlib import Path

import pytest

from govmodel.data.tools import (
    balance_report,
    cross_distribution,
    exact_dedupe,
    issues_summary,
    jaccard,
    label_distribution,
    leakage_between,
    multilabel_distribution,
    near_dedupe,
    read_jsonl,
    stratified_split,
    validate_examples,
    write_jsonl,
)


def _row(labels=("bezwaar",), text="Hierbij teken ik bezwaar aan tegen uw besluit van 12 maart.",
         source="synthetic"):
    return {"text": text, "labels": list(labels), "source": source}


def test_validate_missing_text():
    issues = validate_examples([{"labels": ["bezwaar"]}])
    assert any(i.kind == "missing_field" and i.detail == "text" for i in issues)


def test_validate_text_too_short():
    issues = validate_examples([_row(text="kort")])
    assert any(i.kind == "text_too_short" for i in issues)


def test_validate_label_invalid():
    issues = validate_examples([_row(labels=["not_a_real_label"])])
    assert any(i.kind == "label_invalid" for i in issues)


def test_validate_strict_rejects_unknown_field():
    rows = [{**_row(), "evil_extra": True}]
    issues = validate_examples(rows, strict=True)
    assert any(i.kind == "unknown_field" for i in issues)


def test_validate_labels_empty():
    issues = validate_examples([_row(labels=[])])
    assert any(i.kind == "labels_empty" for i in issues)


def test_issues_summary_counts_by_kind():
    issues = validate_examples([_row(text="kort"), _row(labels=["nope"])])
    summary = issues_summary(issues)
    assert summary["text_too_short"] >= 1
    assert summary["label_invalid"] >= 1


def test_label_distribution_counts_multilabel():
    rows = [_row(labels=["bezwaar", "klacht"]), _row(labels=["bezwaar"])]
    assert label_distribution(rows) == {"bezwaar": 2, "klacht": 1}


def test_multilabel_distribution():
    rows = [
        _row(labels=["bezwaar"]),
        _row(labels=["bezwaar", "klacht"]),
        _row(labels=["bezwaar", "klacht", "woo_verzoek"]),
    ]
    d = multilabel_distribution(rows)
    assert d["1_singletag"] == 1
    assert d["2_pair"] == 1
    assert d["3plus"] == 1


def test_cross_distribution():
    rows = [_row(labels=["a"], source="synthetic"), _row(labels=["b"], source="rechtspraak")]
    # Note labels here aren't real Awb labels, that's fine for the count
    d = cross_distribution(rows)
    assert d["synthetic"]["a"] == 1
    assert d["rechtspraak"]["b"] == 1


def test_balance_report_basic_shape():
    rows = [_row(labels=["bezwaar"])] * 10 + [_row(labels=["klacht"])] * 2
    r = balance_report(rows)
    assert r["n"] == 12
    assert r["label_imbalance_ratio"] == 5.0
    assert r["n_labels_seen"] == 2


def test_exact_dedupe_collapses_whitespace_and_case():
    rows = [
        _row(text="Hierbij teken ik bezwaar aan tegen uw besluit"),
        _row(text="hierbij  Teken ik Bezwaar aan tegen uw besluit "),
        _row(text="Een hele andere klacht over geluidsoverlast"),
    ]
    out = exact_dedupe(rows)
    assert len(out) == 2


def test_jaccard_basics():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 1.0


def test_near_dedupe_catches_one_char_difference():
    rows = [
        _row(text="Hierbij teken ik bezwaar aan tegen uw besluit van 12 maart 2024"),
        _row(text="Hierbij teken ik bezwaar aan tegen uw besluit van 12 maart 2025"),  # 1 digit
        _row(text="Geheel onafhankelijk klagen wij over geluidsoverlast op zaterdag"),
    ]
    out = near_dedupe(rows, threshold=0.85)
    assert len(out) == 2


def test_leakage_between_flags_overlap():
    train = [_row(text="Hierbij teken ik bezwaar aan tegen het besluit van 1 mei")]
    test = [_row(text="hierbij teken ik bezwaar aan tegen het besluit van 1 mei.")]
    matches = leakage_between(train, test, threshold=0.85)
    assert matches and matches[0][:2] == (0, 0)


def test_stratified_split_keeps_primary_label_balance():
    rows = (
        [_row(labels=["bezwaar"])] * 30
        + [_row(labels=["klacht"])] * 10
        + [_row(labels=["woo_verzoek"])] * 5
    )
    splits = stratified_split(rows, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_dist = label_distribution(splits["train"])
    test_dist = label_distribution(splits["test"])
    # Each class must appear in every split (no class starvation).
    for lbl in ("bezwaar", "klacht", "woo_verzoek"):
        assert train_dist.get(lbl, 0) > 0
        assert test_dist.get(lbl, 0) > 0


def test_stratified_split_seed_is_stable():
    rows = [_row(labels=["bezwaar"])] * 20 + [_row(labels=["klacht"])] * 5
    a = stratified_split(rows, seed=11)
    b = stratified_split(rows, seed=11)
    assert [r["labels"][0] for r in a["train"]] == [r["labels"][0] for r in b["train"]]


def test_stratified_split_invalid_ratios():
    with pytest.raises(ValueError):
        stratified_split([_row()], train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_read_write_jsonl_round_trip(tmp_path: Path):
    rows = [_row(labels=["bezwaar"]), _row(labels=["klacht"])]
    p = tmp_path / "out.jsonl"
    n = write_jsonl(p, rows)
    assert n == 2
    back = list(read_jsonl(p))
    assert back == rows
