from __future__ import annotations

import pytest

from govmodel.synthetic import (
    LABEL_PROMPTS,
    STYLE_PROFILES,
    build_messages,
    clean_generated_text,
    is_valid_generation,
)


def test_all_labels_have_required_fields():
    for label, data in LABEL_PROMPTS.items():
        assert "definition" in data, f"{label} missing definition"
        assert "examples" in data, f"{label} missing examples"
        assert "constraints" in data, f"{label} missing constraints"
        assert len(data["examples"]) >= 1, f"{label} needs at least 1 example"
        assert len(data["constraints"]) >= 1, f"{label} needs at least 1 constraint"


def test_all_styles_are_non_empty():
    for style, desc in STYLE_PROFILES.items():
        assert len(desc) > 20, f"Style {style} description too short"


def test_build_messages_structure():
    msgs = build_messages("aanvraag", "formal_short", "Voorbeeld seed.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "aanvraag" in msgs[1]["content"]
    assert "Voorbeeld seed." in msgs[1]["content"]
    assert "formal_short" in msgs[0]["content"] or "formele" in msgs[0]["content"].lower()


def test_build_messages_invalid_label():
    with pytest.raises(KeyError):
        build_messages("nonexistent_label", "formal_short", "seed")  # type: ignore[arg-type]


def test_clean_generated_text_strips_quotes():
    assert clean_generated_text('"  Hallo  "') == "Hallo"


def test_clean_generated_text_strips_code_fences():
    assert clean_generated_text("```\nHallo wereld\n```") == "Hallo wereld"
    assert clean_generated_text("```text\nHallo\n```") == "Hallo"


def test_clean_generated_text_removes_meta_prefix():
    assert clean_generated_text("Brief: Geachte gemeente").startswith("Geachte")
    assert clean_generated_text("Bericht: Hallo").startswith("Hallo")


def test_clean_generated_text_normalizes_whitespace():
    text = "Eerste regel\n\n\n\nTweede regel"
    cleaned = clean_generated_text(text)
    assert "\n\n\n" not in cleaned


def test_is_valid_generation_rejects_too_short():
    assert is_valid_generation("kort") is False


def test_is_valid_generation_rejects_too_long():
    assert is_valid_generation("a" * 5000) is False


def test_is_valid_generation_rejects_meta_intro():
    assert is_valid_generation("Als AI kan ik geen brieven schrijven over dit onderwerp.") is False
    assert is_valid_generation("Natuurlijk! Hier is je brief: Geachte gemeente, etc etc etc.") is False


def test_is_valid_generation_accepts_realistic_text():
    text = (
        "Geachte gemeente, hierbij vraag ik een gehandicaptenparkeerkaart aan. "
        "Mijn moeder kan moeilijk lopen na haar operatie. Ik hoor graag van u."
    )
    assert is_valid_generation(text) is True
