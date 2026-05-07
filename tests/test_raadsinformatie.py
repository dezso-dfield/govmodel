from __future__ import annotations

from govmodel.pullers.raadsinformatie import (
    CITIZEN_LETTER_PATTERNS,
    build_citizen_letter_query,
    extract_text,
    gemeente_from_index,
)


def test_build_query_has_should_clauses():
    q = build_citizen_letter_query()
    bool_q = q["query"]["bool"]
    assert "should" in bool_q
    assert bool_q["minimum_should_match"] == 1
    # At least one match_phrase per pattern in text
    text_phrases = [
        c["match_phrase"]["text"]
        for c in bool_q["should"]
        if "match_phrase" in c and "text" in c.get("match_phrase", {})
    ]
    for pattern in CITIZEN_LETTER_PATTERNS:
        assert pattern in text_phrases


def test_build_query_excludes_bundles():
    q = build_citizen_letter_query()
    must_not = q["query"]["bool"]["must_not"]
    excluded_names = [
        c["match_phrase"]["name"]
        for c in must_not
        if "match_phrase" in c and "name" in c.get("match_phrase", {})
    ]
    assert "agendabundel" in excluded_names
    assert "begroting" in excluded_names


def test_build_query_includes_extra_patterns():
    q = build_citizen_letter_query(extra_patterns=["custom pattern"])
    text_phrases = [
        c["match_phrase"]["text"]
        for c in q["query"]["bool"]["should"]
        if "match_phrase" in c and "text" in c.get("match_phrase", {})
    ]
    assert "custom pattern" in text_phrases


def test_extract_text_from_list():
    doc = {"text": ["Eerste deel.", "Tweede deel."]}
    assert "Eerste deel." in extract_text(doc)
    assert "Tweede deel." in extract_text(doc)


def test_extract_text_from_string():
    doc = {"text": "  Een enkele string.  "}
    assert extract_text(doc) == "Een enkele string."


def test_extract_text_empty():
    assert extract_text({}) == ""
    assert extract_text({"text": None}) == ""
    assert extract_text({"text": []}) == ""


def test_gemeente_from_index_basic():
    assert gemeente_from_index("ori_aalsmeer_20250410235456") == "aalsmeer"


def test_gemeente_from_index_compound():
    assert gemeente_from_index("ori_den_haag_20240115120000") == "den_haag"


def test_gemeente_from_index_invalid():
    assert gemeente_from_index(None) is None
    assert gemeente_from_index("not_an_ori_index") is None
    assert gemeente_from_index("ori_short") is None
