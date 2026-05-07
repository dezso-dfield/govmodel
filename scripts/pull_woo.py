"""Discover WOO-datasets via data.overheid.nl CKAN catalog.

Het is een registratie-puller, geen content-puller: we leggen vast welke
WOO-datasets er zijn, met welke licentie, en welke machine-leesbare resources
ze hebben. Daadwerkelijke content-fetch komt in een tweede ronde.

Voorbeeld:
    python scripts/pull_woo.py --output data/raw/woo/registry.jsonl --max-pages 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import httpx

from govmodel.pullers.woo import USER_AGENT, search_woo_datasets, summarize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", default="woo")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("pull_woo")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts = {"datasets": 0, "with_machine_readable": 0}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    with httpx.Client(headers=headers) as client, args.output.open("a", encoding="utf-8") as out:
        for dataset in search_woo_datasets(client, query=args.query, max_pages=args.max_pages):
            summary = summarize_dataset(dataset)
            out.write(json.dumps(summary, ensure_ascii=False) + "\n")
            out.flush()
            counts["datasets"] += 1
            if summary["n_resources_machine_readable"] > 0:
                counts["with_machine_readable"] += 1

    logger.info(
        "Klaar: %d datasets in registry, %d met machine-leesbare resources",
        counts["datasets"], counts["with_machine_readable"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
