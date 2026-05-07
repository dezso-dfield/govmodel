from __future__ import annotations

from govmodel.extractors.burger_citaten import (
    extract_citations,
    quality_score,
    suggest_labels,
)


def test_extracts_simple_direct_quote():
    text = (
        "De rechtbank stelt vast dat eiseres heeft aangevoerd: "
        '"Ik ben het niet eens met de hoogte van het schadebedrag dat '
        'door de gemeente is bepaald, mijn schade is veel hoger". '
        "Het college voert hiertegen aan dat..."
    )
    cands = extract_citations(text)
    assert len(cands) == 1
    assert "Ik ben het niet eens" in cands[0].quote


def test_skips_quote_without_actor_context():
    text = (
        "De rechter overweegt dat de stelling 'het was een zonnige dag' "
        "niet relevant is voor de juridische beoordeling van de zaak die "
        "thans voorligt en die op grond van de Awb moet worden behandeld."
    )
    cands = extract_citations(text)
    assert cands == []


def test_extracts_with_appellant():
    text = (
        "Appellant heeft bij brief van 5 maart 2024 het volgende geschreven: "
        '"Hierbij teken ik bezwaar aan tegen uw besluit van 12 februari, '
        'omdat ik het niet eens ben met de afwijzing van mijn aanvraag." '
        "De rechtbank oordeelt..."
    )
    cands = extract_citations(text)
    assert len(cands) >= 1
    assert "bezwaar" in cands[0].quote.lower()


def test_suggest_labels_from_awb_articles():
    context = "De rechtbank verwijst naar artikel 6:5 Awb en stelt dat appellant in zijn bezwaarschrift..."
    labels, reason = suggest_labels(context)
    assert "bezwaar" in labels


def test_suggest_labels_woo():
    context = "Verzoeker heeft op grond van de Wet open overheid om openbaarmaking gevraagd."
    labels, reason = suggest_labels(context)
    assert "woo_verzoek" in labels


def test_suggest_labels_zienswijze():
    context = "In haar zienswijze op het ontwerpbesluit heeft eiseres gesteld..."
    labels, reason = suggest_labels(context)
    assert "zienswijze" in labels


def test_quality_score_higher_for_burger_like():
    burger_quote = "Hierbij teken ik bezwaar aan tegen uw besluit. Ik ben het niet eens met de afwijzing van mijn aanvraag voor huishoudelijke hulp. Met vriendelijke groet."
    rechter_like_quote = "Naar het oordeel van de Afdeling overweegt appellant terecht dat de motivering ontoereikend is volgens de geldende jurisprudentie."
    score_burger = quality_score(burger_quote, "")
    score_rechter = quality_score(rechter_like_quote, "")
    assert score_burger > score_rechter


def test_quality_score_bounded():
    score = quality_score("Ik ik ik mijn mijn", "")
    assert 0.0 <= score <= 1.0
    score2 = quality_score("De rechter de rechter de rechter overweegt", "")
    assert 0.0 <= score2 <= 1.0


def test_extract_handles_empty_text():
    assert extract_citations("") == []
    assert extract_citations(None) == []  # type: ignore[arg-type]


def test_extract_skips_too_short_quotes():
    text = "Eiser heeft geschreven: 'Te kort.' Dit is geen valide burgercitaat."
    cands = extract_citations(text)
    assert cands == []


def test_extract_finds_multiple_quotes_in_one_text():
    text = (
        "Appellant heeft op 1 maart aangevoerd: "
        '"Ik ben het niet eens met de afwijzing van mijn WMO-aanvraag '
        'omdat mijn medische situatie wel degelijk de voorziening rechtvaardigt." '
        "Daarnaast heeft eiseres op 15 maart geschreven: "
        '"Ik klaag hierbij over de bejegening door uw medewerker tijdens '
        'het keukentafelgesprek dat onnodig confronterend was."'
    )
    cands = extract_citations(text)
    assert len(cands) == 2
