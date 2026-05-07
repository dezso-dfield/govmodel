from __future__ import annotations

from govmodel.pullers.rechtspraak import (
    extract_awb_citations,
    has_relevant_signal,
    parse_uitspraak_xml,
)
from govmodel.schemas import RechtspraakUitspraak


def test_extract_awb_citations_basic():
    text = "Op grond van artikel 6:5 Awb en art. 9:1 lid 1 dient het bezwaarschrift..."
    citations = extract_awb_citations(text)
    assert "6:5" in citations
    assert "9:1" in citations


def test_extract_awb_citations_filters_irrelevant():
    text = "Verwijzing naar artikel 99:99 valt buiten onze relevante set."
    citations = extract_awb_citations(text)
    assert "99:99" not in citations


def test_extract_awb_citations_unique_sorted():
    text = "art. 6:5, art. 6:5, artikel 9:1, art 4:1"
    citations = extract_awb_citations(text)
    assert citations == ["4:1", "6:5", "9:1"]


def test_extract_awb_citations_handles_letter_suffix():
    text = "artikel 6:5a Awb"
    citations = extract_awb_citations(text)
    # 6:5a is geen relevante artikel-key in onze map dus wordt gefilterd
    # Test bevestigt dat extractie niet crasht op letter-suffix
    assert isinstance(citations, list)


def test_has_relevant_signal_with_awb():
    u = RechtspraakUitspraak(
        ecli="ECLI:NL:TEST:2024:1",
        content_url="https://example.org/test",
        awb_articles_cited=["6:5"],
    )
    assert has_relevant_signal(u) is True


def test_has_relevant_signal_with_woo():
    u = RechtspraakUitspraak(
        ecli="ECLI:NL:TEST:2024:2",
        content_url="https://example.org/test",
        woo_cited=True,
    )
    assert has_relevant_signal(u) is True


def test_has_relevant_signal_without_signals():
    u = RechtspraakUitspraak(
        ecli="ECLI:NL:TEST:2024:3",
        content_url="https://example.org/test",
    )
    assert has_relevant_signal(u) is False


def test_parse_uitspraak_xml_minimal():
    """Sanity: parser kraakt niet op een minimale geldige XML zonder content."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <open-rechtspraak xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                      xmlns:dcterms="http://purl.org/dc/terms/"
                      xmlns:psi="http://psi.rechtspraak.nl/"
                      xmlns:rs="http://www.rechtspraak.nl/schema/rechtspraak-1.0">
        <rdf:RDF>
            <rdf:Description>
                <dcterms:identifier>ECLI:NL:TEST:2024:99</dcterms:identifier>
                <dcterms:creator>Centrale Raad van Beroep</dcterms:creator>
                <dcterms:date>2024-03-15</dcterms:date>
                <dcterms:subject>Bestuursrecht; Socialezekerheidsrecht</dcterms:subject>
                <psi:zaaknummer>23/4567</psi:zaaknummer>
            </rdf:Description>
        </rdf:RDF>
        <rs:uitspraak>
            <rs:parablock>
                <rs:para>De gemeente heeft het bezwaar van appellant ongegrond verklaard
                op grond van artikel 6:5 Awb.</rs:para>
                <rs:para>Tevens heeft appellant een klacht ingediend op grond van art. 9:1.</rs:para>
            </rs:parablock>
        </rs:uitspraak>
    </open-rechtspraak>
    """
    result = parse_uitspraak_xml("ECLI:NL:TEST:2024:99", xml)
    assert result.ecli == "ECLI:NL:TEST:2024:99"
    assert result.instance == "Centrale Raad van Beroep"
    assert result.subject is not None and "Bestuursrecht" in result.subject
    assert result.case_number == "23/4567"
    assert result.full_text is not None
    assert "bezwaar" in result.full_text.lower()
    assert "6:5" in result.awb_articles_cited
    assert "9:1" in result.awb_articles_cited


def test_parse_uitspraak_xml_handles_malformed():
    result = parse_uitspraak_xml("ECLI:NL:TEST:2024:bad", "<not xml")
    assert result.ecli == "ECLI:NL:TEST:2024:bad"
    assert result.full_text is None
    assert result.awb_articles_cited == []
