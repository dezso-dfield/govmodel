"""Pull potentiële burgerbrieven uit Open Raadsinformatie.

Zoekt via Elasticsearch op patronen die typisch zijn voor 'ingekomen stukken'
en burgerbrieven aan de raad. Output: JSONL met document-metadata + tekst.

Voorbeeld:
    python scripts/pull_raadsinformatie.py \\
        --output data/raw/raadsinformatie/sample.jsonl \\
        --max-pages 5 \\
        --min-text-chars 300
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import httpx

from govmodel.pullers.raadsinformatie import (
    USER_AGENT,
    build_citizen_letter_query,
    extract_text,
    gemeente_from_index,
    search_documents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-pattern", default="_all",
                        help="ES index pattern, bv. 'ori_aalsmeer_*' of '_all'")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--min-text-chars", type=int, default=200,
                        help="Minimale tekstlengte voor opname")
    parser.add_argument("--max-text-chars", type=int, default=20000,
                        help="Maximale tekstlengte (langere docs zijn meestal bundels)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("pull_raadsinformatie")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    query = build_citizen_letter_query(min_text_chars=args.min_text_chars)

    counts = {"hits": 0, "kept": 0, "too_short": 0, "too_long": 0}
    headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}

    with httpx.Client(headers=headers) as client, args.output.open("a", encoding="utf-8") as out:
        for doc in search_documents(
            client, query,
            index_pattern=args.index_pattern,
            page_size=args.page_size,
            max_pages=args.max_pages,
        ):
            counts["hits"] += 1
            text = extract_text(doc)
            if len(text) < args.min_text_chars:
                counts["too_short"] += 1
                continue
            if len(text) > args.max_text_chars:
                counts["too_long"] += 1
                continue

            record = {
                "id": doc.get("id"),
                "gemeente": gemeente_from_index(doc.get("index")),
                "index": doc.get("index"),
                "score": doc.get("score"),
                "name": doc.get("name"),
                "file_name": doc.get("file_name"),
                "content_type": doc.get("content_type"),
                "original_url": doc.get("original_url"),
                "url": doc.get("url"),
                "text": text,
                "text_length": len(text),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            counts["kept"] += 1

            if counts["hits"] % 25 == 0:
                logger.info(
                    "Voortgang: hits=%d behouden=%d te-kort=%d te-lang=%d",
                    counts["hits"], counts["kept"],
                    counts["too_short"], counts["too_long"],
                )

    logger.info(
        "Klaar: hits=%d behouden=%d te-kort=%d te-lang=%d",
        counts["hits"], counts["kept"], counts["too_short"], counts["too_long"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
