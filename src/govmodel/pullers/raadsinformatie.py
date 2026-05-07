"""Puller voor Open Raadsinformatie (api.openraadsinformatie.nl).

ORI is een Open State Foundation initiatief dat raadsinformatie van
Nederlandse gemeenten ontsluit via Elasticsearch. Indexes per gemeente
volgen pattern `ori_<gemeente>_<timestamp>`.

We zoeken naar documenten die kenmerken hebben van burgerbrieven aan de
gemeente — typisch 'ingekomen stukken' op de raadsagenda.

API: Elasticsearch _search proxy.
Endpoint: https://api.openraadsinformatie.nl/v1/elastic
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

ORI_BASE_URL = "https://api.openraadsinformatie.nl/v1/elastic"
DEFAULT_RATE_LIMIT_S = 0.5
DEFAULT_TIMEOUT_S = 60.0
USER_AGENT = "govmodel-puller/0.1 (open-source Awb-type-classifier research)"

# Tekstpatronen die wijzen op een burger-aan-gemeente bericht in ingekomen
# stukken. Gebruikt in een Elasticsearch query_string of multi_match.
CITIZEN_LETTER_PATTERNS = [
    "ingekomen stuk",
    "ingekomen brief",
    "brief van inwoner",
    "brief van bewoner",
    "burgerbrief",
    "klacht aan de raad",
    "burgerinitiatief",
    "ingekomen klacht",
    "schrijven van",
]

# Filenaam-patronen die typisch zijn voor ingekomen stukken
FILENAME_PATTERNS_RELEVANT = [
    "ingekomen",
    "brief",
    "klacht",
    "burgerinitiatief",
    "zienswijze",
]

DEFAULT_NAME_PATTERNS_EXCLUDE = [
    "agendabundel",
    "presentatie",
    "raadsbesluit",
    "begroting",
    "jaarrekening",
]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
def _http_post(client: httpx.Client, url: str, json_body: dict) -> httpx.Response:
    response = client.post(url, json=json_body, timeout=DEFAULT_TIMEOUT_S)
    response.raise_for_status()
    return response


def build_citizen_letter_query(
    extra_patterns: list[str] | None = None,
    min_text_chars: int = 200,
) -> dict[str, Any]:
    """Bouw een Elasticsearch query voor potentiële burgerbrieven."""
    patterns = list(CITIZEN_LETTER_PATTERNS)
    if extra_patterns:
        patterns.extend(extra_patterns)

    should_clauses: list[dict[str, Any]] = []
    for pattern in patterns:
        should_clauses.append({"match_phrase": {"text": pattern}})
        should_clauses.append({"match_phrase": {"name": pattern}})
    for fname in FILENAME_PATTERNS_RELEVANT:
        should_clauses.append({"wildcard": {"file_name": f"*{fname}*"}})

    must_not = [{"match_phrase": {"name": p}} for p in DEFAULT_NAME_PATTERNS_EXCLUDE]

    return {
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
                "must_not": must_not,
            }
        },
        "_source": ["name", "url", "file_name", "content_type", "original_url", "text"],
    }


def search_documents(
    client: httpx.Client,
    query: dict[str, Any],
    index_pattern: str = "_all",
    page_size: int = 50,
    max_pages: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield documenten via ES scroll-achtige paginering met `from`/`size`.

    NB: voor grotere result-sets is een echte scroll/PIT API beter. Voor v0.1
    is from-size met max_pages ruim voldoende.
    """
    base_url = f"{ORI_BASE_URL}/{index_pattern}/_search"
    page = 0
    seen_ids: set[str] = set()

    while True:
        body = dict(query)
        body["from"] = page * page_size
        body["size"] = page_size

        time.sleep(DEFAULT_RATE_LIMIT_S)
        logger.info("ORI search pagina %d, index=%s", page, index_pattern)
        try:
            response = _http_post(client, base_url, body)
        except httpx.HTTPStatusError as e:
            logger.error("ORI request fout: %s", e)
            return

        data = response.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            logger.info("Geen hits meer op pagina %d", page)
            break

        new_count = 0
        for hit in hits:
            hit_id = hit.get("_id")
            if hit_id in seen_ids:
                continue
            seen_ids.add(hit_id)
            new_count += 1
            yield {
                "id": hit_id,
                "index": hit.get("_index"),
                "score": hit.get("_score"),
                **hit.get("_source", {}),
            }

        if new_count == 0:
            logger.info("Geen nieuwe hits op pagina %d", page)
            break
        if len(hits) < page_size:
            break

        page += 1
        if max_pages is not None and page >= max_pages:
            logger.info("Max pagina-limiet (%d) bereikt", max_pages)
            break


def extract_text(doc: dict[str, Any]) -> str:
    """Combineer de text-array tot één string."""
    text_field = doc.get("text") or []
    if isinstance(text_field, str):
        return text_field.strip()
    if isinstance(text_field, list):
        return "\n\n".join(str(t).strip() for t in text_field if t).strip()
    return ""


def gemeente_from_index(index_name: str | None) -> str | None:
    """Extract gemeente uit indexnaam: 'ori_aalsmeer_20250410235456' → 'aalsmeer'."""
    if not index_name or not index_name.startswith("ori_"):
        return None
    parts = index_name.split("_")
    if len(parts) < 3:
        return None
    # Alles tussen 'ori_' en de timestamp-suffix is de gemeentenaam
    return "_".join(parts[1:-1])
