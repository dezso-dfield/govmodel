"""Puller voor Open Data Rechtspraak (data.rechtspraak.nl).

Gepubliceerd onder CC0. Wij respecteren een beleefde rate limit en geven
attributie volgens de richtlijnen van de Rechtspraak.

API endpoints (geverifieerd 2026-05-07):
- Zoeken: https://data.rechtspraak.nl/uitspraken/zoeken (Atom feed)
- Inhoud: https://data.rechtspraak.nl/uitspraken/content?id=<ECLI>
  (RDF + rechtspraak-1.0 XML)
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from govmodel.schemas import RechtspraakUitspraak

logger = logging.getLogger(__name__)

SEARCH_URL = "https://data.rechtspraak.nl/uitspraken/zoeken"
CONTENT_URL = "https://data.rechtspraak.nl/uitspraken/content"

USER_AGENT = (
    "govmodel-puller/0.1 "
    "(open-source Awb-type-classifier research; "
    "respect for CC0 attribution and polite rate limits)"
)

DEFAULT_RATE_LIMIT_S = 0.6  # ~1.5 req/s
DEFAULT_TIMEOUT_S = 30.0
SEARCH_PAGE_SIZE = 1000  # API maximum

# XML namespaces
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dcterms": "http://purl.org/dc/terms/",
    "psi": "http://psi.rechtspraak.nl/",
    "rs": "http://www.rechtspraak.nl/schema/rechtspraak-1.0",
}

# Awb-artikelen die juridisch relevant zijn voor onze typering.
# Sleutel = artikel-id ("boek:artikel"); waarde = welke labels het indiceert.
RELEVANT_AWB_ARTICLES: dict[str, tuple[str, ...]] = {
    "1:3": ("aanvraag",),  # definitie aanvraag
    "3:11": ("informatieverzoek_3_11",),
    "3:15": ("zienswijze",),
    "4:1": ("aanvraag",),
    "4:8": ("zienswijze",),
    "6:4": ("bezwaar",),
    "6:5": ("bezwaar",),
    "6:6": ("bezwaar",),
    "7:1": ("bezwaar",),
    "9:1": ("klacht",),
    "9:8": ("klacht",),
}

# Regex voor "artikel 6:5", "art. 9:1", "artikel 6:5 lid 1 Awb", etc.
AWB_CITATION_PATTERN = re.compile(
    r"\b(?:artikel|art\.?)\s*(\d+):(\d+[a-z]?)",
    re.IGNORECASE,
)

# WOO-verwijzingen (Wet open overheid art. 4.1)
WOO_PATTERN = re.compile(
    r"\b(?:Wet\s+open\s+overheid|Woo)\b",
    re.IGNORECASE,
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
def _http_get(client: httpx.Client, url: str, params=None) -> httpx.Response:
    response = client.get(url, params=params, timeout=DEFAULT_TIMEOUT_S)
    response.raise_for_status()
    return response


def search_eclis(
    client: httpx.Client,
    date_from: date,
    date_to: date,
    return_only_with_content: bool = True,
    max_pages: int | None = None,
) -> Iterator[str]:
    """Yield ECLIs uit de zoek-API voor een datumrange.

    Args:
        client: geconfigureerde httpx.Client
        date_from / date_to: datumrange (inclusief)
        return_only_with_content: alleen uitspraken waarvan de tekst beschikbaar is
        max_pages: limiet voor testing

    Yields:
        ECLI-strings zoals "ECLI:NL:RVS:2024:1234"
    """
    page = 0
    seen_eclis: set[str] = set()

    while True:
        # Rechtspraak API verwacht TWEE `date`-parameters met dezelfde naam
        # voor een datum-range — niet `date` + `date2`. httpx accepteert
        # list-of-tuples om dezelfde key meerdere keren te sturen.
        params: list[tuple[str, str | int]] = [
            ("type", "Uitspraak"),
            ("date", date_from.isoformat()),
            ("date", date_to.isoformat()),
            ("max", SEARCH_PAGE_SIZE),
            ("from", page * SEARCH_PAGE_SIZE),
        ]
        if return_only_with_content:
            params.append(("return", "DOC"))

        logger.info("Zoeken: pagina %d, %d-%d (params %s)",
                    page, page * SEARCH_PAGE_SIZE, (page + 1) * SEARCH_PAGE_SIZE, params)
        time.sleep(DEFAULT_RATE_LIMIT_S)
        response = _http_get(client, SEARCH_URL, params)

        try:
            feed = ET.fromstring(response.content)
        except ET.ParseError as e:
            logger.error("Kan zoekresponse niet parsen op pagina %d: %s", page, e)
            return

        entries = feed.findall("atom:entry", NS)
        if not entries:
            logger.info("Geen entries meer op pagina %d, stoppen.", page)
            break

        new_count = 0
        for entry in entries:
            id_el = entry.find("atom:id", NS)
            if id_el is None or not id_el.text:
                continue
            ecli = id_el.text.strip()
            if ecli in seen_eclis:
                continue
            seen_eclis.add(ecli)
            new_count += 1
            yield ecli

        if new_count == 0:
            logger.info("Geen nieuwe entries op pagina %d, stoppen.", page)
            break

        if len(entries) < SEARCH_PAGE_SIZE:
            break

        page += 1
        if max_pages is not None and page >= max_pages:
            logger.info("Max pagina-limiet (%d) bereikt.", max_pages)
            break


def fetch_uitspraak(client: httpx.Client, ecli: str) -> RechtspraakUitspraak:
    """Haal volledige inhoud van één uitspraak op en parse naar schema."""
    logger.debug("Ophalen content voor %s", ecli)
    time.sleep(DEFAULT_RATE_LIMIT_S)
    response = _http_get(client, CONTENT_URL, params={"id": ecli})
    return parse_uitspraak_xml(ecli, response.text)


def parse_uitspraak_xml(ecli: str, xml_content: str) -> RechtspraakUitspraak:
    """Parse rechtspraak content-XML naar RechtspraakUitspraak."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.warning("XML-parse-fout voor %s: %s", ecli, e)
        return RechtspraakUitspraak(
            ecli=ecli,
            content_url=f"{CONTENT_URL}?id={ecli}",
        )

    instance = _find_text(root, ".//dcterms:creator")
    decision_date_str = _find_text(root, ".//dcterms:date")
    subject = _find_text(root, ".//dcterms:subject")
    case_number = _find_text(root, ".//psi:zaaknummer")
    summary = _find_text(root, ".//dcterms:abstract") or _find_text(root, ".//rs:inhoudsindicatie")

    full_text = extract_uitspraak_text(root)
    awb_articles = extract_awb_citations(full_text or "")
    woo_cited = bool(WOO_PATTERN.search(full_text or ""))

    parsed_date: date | None = None
    if decision_date_str:
        try:
            parsed_date = date.fromisoformat(decision_date_str.strip()[:10])
        except ValueError:
            logger.debug("Onparsebare datum '%s' voor %s", decision_date_str, ecli)

    return RechtspraakUitspraak(
        ecli=ecli,
        content_url=f"{CONTENT_URL}?id={ecli}",
        instance=instance,
        decision_date=parsed_date,
        subject=subject,
        case_number=case_number,
        summary=summary,
        full_text=full_text,
        awb_articles_cited=awb_articles,
        woo_cited=woo_cited,
    )


def _find_text(root: ET.Element, xpath: str) -> str | None:
    el = root.find(xpath, NS)
    if el is None or not el.text:
        return None
    return el.text.strip() or None


def extract_uitspraak_text(root: ET.Element) -> str | None:
    """Extract de uitspraak-tekst uit het XML.

    De rechtspraak-1.0 schema plaatst de tekst onder <uitspraak> met
    <parablock>/<para>-structuur. We pakken alle tekst recursief op.
    """
    uitspraak_el = root.find(".//rs:uitspraak", NS)
    if uitspraak_el is None:
        # Fallback: misschien zit het direct onder root zonder namespace
        uitspraak_el = root.find(".//uitspraak")
    if uitspraak_el is None:
        return None

    parts: list[str] = []
    for el in uitspraak_el.iter():
        if el.text:
            text = el.text.strip()
            if text:
                parts.append(text)
        if el.tail:
            tail = el.tail.strip()
            if tail:
                parts.append(tail)
    text = " ".join(parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def extract_awb_citations(text: str) -> list[str]:
    """Extract Awb-artikelverwijzingen zoals '6:5', '9:1' uit tekst.

    Returns gesorteerde unieke lijst van artikel-ids die in onze relevante
    set staan. Niet-relevante artikelen worden gefilterd om ruis te beperken.
    """
    matches = AWB_CITATION_PATTERN.findall(text)
    citations = {f"{book}:{article}" for book, article in matches}
    return sorted(c for c in citations if c in RELEVANT_AWB_ARTICLES)


def has_relevant_signal(uitspraak: RechtspraakUitspraak) -> bool:
    """Filter: behoud uitspraken met minstens één relevante Awb- of WOO-verwijzing."""
    return bool(uitspraak.awb_articles_cited) or uitspraak.woo_cited
