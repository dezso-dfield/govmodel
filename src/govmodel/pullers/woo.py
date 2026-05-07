"""Puller voor WOO-publicaties via data.overheid.nl CKAN catalog.

Het WOO-publicatielandschap is fragmented: niet alle gemeenten publiceren
machine-leesbaar. We gebruiken data.overheid.nl als startpunt om datasets te
ontdekken en focussen op publishers met JSON/CSV/XML resources. PDFs worden
geregistreerd voor latere OCR-verwerking (buiten v0.1 scope).

API: CKAN action API
- Documentatie: https://docs.ckan.org/en/2.10/api/
- Endpoint: https://data.overheid.nl/data/api/3/action/package_search
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

CKAN_BASE_URL = "https://data.overheid.nl/data/api/3/action"
DEFAULT_RATE_LIMIT_S = 0.6
DEFAULT_TIMEOUT_S = 30.0
USER_AGENT = "govmodel-puller/0.1 (open-source Awb-type-classifier research)"

# Formaten waarvan we verwachten machine-leesbaar te zijn voor v0.1.
MACHINE_READABLE_FORMATS = {"json", "csv", "xml", "jsonl", "ndjson", "tsv", "geojson"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
def _http_get(client: httpx.Client, url: str, params: dict | None = None) -> httpx.Response:
    response = client.get(url, params=params, timeout=DEFAULT_TIMEOUT_S)
    response.raise_for_status()
    return response


def search_woo_datasets(
    client: httpx.Client,
    query: str = "woo",
    rows_per_page: int = 100,
    max_pages: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield WOO-gerelateerde datasets uit data.overheid.nl.

    Returns dict per dataset met velden zoals 'title', 'organization',
    'resources', 'metadata_modified', 'license_id'.
    """
    page = 0
    while True:
        time.sleep(DEFAULT_RATE_LIMIT_S)
        params = {"q": query, "rows": rows_per_page, "start": page * rows_per_page}
        logger.info("CKAN package_search pagina %d, q=%s", page, query)
        response = _http_get(client, f"{CKAN_BASE_URL}/package_search", params)
        data = response.json()

        if not data.get("success"):
            logger.error("CKAN gaf success=false: %s", data.get("error"))
            return

        results = data.get("result", {}).get("results", [])
        if not results:
            logger.info("Geen resultaten meer op pagina %d", page)
            break

        for ds in results:
            yield ds

        if len(results) < rows_per_page:
            break
        page += 1
        if max_pages is not None and page >= max_pages:
            break


def filter_machine_readable_resources(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Geef alleen resources met machine-leesbare formaten terug."""
    resources = dataset.get("resources", []) or []
    machine_readable = []
    for res in resources:
        fmt = (res.get("format") or "").lower().strip()
        if fmt in MACHINE_READABLE_FORMATS:
            machine_readable.append(res)
    return machine_readable


def summarize_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Compact summary geschikt voor JSONL-registratie."""
    org = dataset.get("organization") or {}
    machine_resources = filter_machine_readable_resources(dataset)
    all_resources = dataset.get("resources", []) or []

    return {
        "id": dataset.get("id"),
        "name": dataset.get("name"),
        "title": dataset.get("title"),
        "organization": org.get("title") or org.get("name"),
        "license_id": dataset.get("license_id"),
        "license_title": dataset.get("license_title"),
        "metadata_modified": dataset.get("metadata_modified"),
        "notes_excerpt": (dataset.get("notes") or "")[:300],
        "n_resources_total": len(all_resources),
        "n_resources_machine_readable": len(machine_resources),
        "machine_readable_urls": [
            {
                "format": (r.get("format") or "").lower(),
                "url": r.get("url"),
                "name": r.get("name"),
            }
            for r in machine_resources
        ],
        "all_resource_formats": sorted({(r.get("format") or "").lower() for r in all_resources}),
    }
