"""Tests for the augmentation library."""
from __future__ import annotations

from govmodel.data.augment import (
    apply_all,
    casefold_perturb,
    dialect_swap,
    inject_typos,
    nt2_simplify,
    punctuation_perturb,
    shout_emphasis,
    standard_augmentations,
)


def test_typos_deterministic_with_seed():
    text = "Hierbij teken ik bezwaar aan tegen het besluit van 12 maart 2024"
    assert inject_typos(text, rate=0.2, seed=42) == inject_typos(text, rate=0.2, seed=42)


def test_typos_rate_zero_is_identity():
    text = "Geen typos toevoegen alstublieft"
    assert inject_typos(text, rate=0.0, seed=1) == text


def test_typos_different_seed_diverges():
    text = "Hierbij teken ik bezwaar aan tegen het besluit"
    a = inject_typos(text, rate=0.2, seed=1)
    b = inject_typos(text, rate=0.2, seed=99)
    assert a != b


def test_casefold_modes():
    assert casefold_perturb("Hallo", kind="lower") == "hallo"
    assert casefold_perturb("Hallo", kind="upper") == "HALLO"
    assert casefold_perturb("hallo wereld", kind="title") == "Hallo Wereld"


def test_casefold_random_deterministic():
    a = casefold_perturb("Hallo wereld", kind="random", seed=1)
    b = casefold_perturb("Hallo wereld", kind="random", seed=1)
    assert a == b


def test_punctuation_strip():
    text = "Hallo, wereld! Hoe gaat het?"
    out = punctuation_perturb(text, mode="strip")
    assert "," not in out and "!" not in out and "?" not in out


def test_punctuation_double():
    assert punctuation_perturb("Hallo, wereld.", mode="double") == "Hallo,, wereld.."


def test_shout_emphasis_deterministic():
    text = "ik ben echt boos over deze afhandeling"
    a = shout_emphasis(text, rate=0.5, seed=7)
    b = shout_emphasis(text, rate=0.5, seed=7)
    assert a == b


def test_shout_emphasis_rate_zero_is_identity():
    text = "hallo wereld"
    assert shout_emphasis(text, rate=0.0, seed=1) == text


def test_nt2_simplify_rewrites_formal_connectives():
    text = "Echter, aangezien derhalve, hoogachtend."
    out = nt2_simplify(text)
    assert "maar" in out.lower()
    assert "omdat" in out.lower()
    assert "dus" in out.lower()
    assert "groet" in out.lower()


def test_nt2_simplify_idempotent():
    text = "Echter, aangezien derhalve."
    assert nt2_simplify(nt2_simplify(text)) == nt2_simplify(text)


def test_dialect_swap_replaces_anglicism():
    out = dialect_swap("Dit is wrong because hij het niet wist")
    assert "because" not in out
    assert "omdat" in out


def test_dialect_swap_preserves_clean_text():
    text = "Een doodgewone Nederlandse zin"
    assert dialect_swap(text) == text


def test_standard_augmentations_includes_gov_specific():
    augs = standard_augmentations()
    names = [a.name for a in augs]
    # Gov-specific addition vs School's pack
    assert "shout_emphasis" in names
    assert "nt2_simplify" in names
    assert len(set(names)) == len(names)  # unique


def test_apply_all_returns_one_row_per_aug():
    augs = standard_augmentations(seed=3)
    out = apply_all("Een test brief aan de gemeente", augs)
    assert len(out) == len(augs)
    assert [n for n, _ in out] == [a.name for a in augs]
