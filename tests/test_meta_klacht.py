"""Tests for the meta-klacht prompt templates."""
from __future__ import annotations

from govmodel.data.meta_klacht import (
    MetaKlachtPromptSpec,
    iter_prompts,
    scenarios,
    styles,
    system_prompt,
)


def test_iter_prompts_covers_full_grid():
    n = sum(1 for _ in iter_prompts(examples_per_combo=1))
    assert n == len(scenarios()) * len(styles())


def test_iter_prompts_yields_unique_specs():
    specs = list(iter_prompts(examples_per_combo=1))
    seen = set()
    for s in specs:
        key = (s.scenario_key, s.style)
        assert key not in seen
        seen.add(key)


def test_iter_prompts_examples_per_combo_multiplies():
    n = sum(1 for _ in iter_prompts(examples_per_combo=3))
    assert n == len(scenarios()) * len(styles()) * 3


def test_each_spec_has_klacht_label():
    for s in iter_prompts(examples_per_combo=1):
        assert s.expected_labels == ["klacht"]


def test_prompts_reference_meta_complaint():
    """All prompts must mention meta-klacht / klachtafhandeling so the
    LLM cannot drift into writing a plain klacht."""
    for s in iter_prompts(examples_per_combo=1):
        assert "meta-klacht" in s.prompt.lower() or "klachtafhandel" in s.prompt.lower()


def test_styles_are_distinct():
    assert len(set(styles())) == len(styles())


def test_system_prompt_forbids_pii():
    sys_p = system_prompt()
    assert "BSN" in sys_p
    assert "IBAN" in sys_p


def test_spec_dataclass_is_frozen():
    s = next(iter_prompts())
    assert isinstance(s, MetaKlachtPromptSpec)
    try:
        s.scenario_key = "changed"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("MetaKlachtPromptSpec should be frozen")
