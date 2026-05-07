"""Pull bestuursrechtspraak uitspraken uit Open Data Rechtspraak.

Voorbeeld:
    python scripts/pull_rechtspraak.py \\
        --date-from 2024-01-01 \\
        --date-to 2024-01-31 \\
        --output data/raw/rechtspraak/2024-01.jsonl \\
        --max-pages 2

Resumable: ECLIs die al in het output-bestand staan worden overgeslagen.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import httpx

from govmodel.pullers.rechtspraak import (
    USER_AGENT,
    fetch_uitspraak,
    has_relevant_signal,
    search_eclis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date-from", type=date.fromisoformat, required=True,
                        help="ISO-datum (YYYY-MM-DD)")
    parser.add_argument("--date-to", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL-pad")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Limiet op zoekpagina's (handig voor testen)")
    parser.add_argument("--keep-all", action="store_true",
                        help="Niet filteren op relevante Awb/WOO-citaten")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def load_existing_eclis(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    eclis: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ecli" in obj:
                eclis.add(obj["ecli"])
    return eclis


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("pull_rechtspraak")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_eclis(args.output)
    if existing:
        logger.info("Resuming: %d ECLIs reeds opgehaald", len(existing))

    headers = {"User-Agent": USER_AGENT, "Accept": "application/xml"}
    counts = {"found": 0, "skipped": 0, "fetched": 0, "kept": 0, "errors": 0}

    with httpx.Client(headers=headers) as client, args.output.open("a", encoding="utf-8") as out:
        for ecli in search_eclis(
            client, args.date_from, args.date_to, max_pages=args.max_pages
        ):
            counts["found"] += 1
            if ecli in existing:
                counts["skipped"] += 1
                continue

            try:
                uitspraak = fetch_uitspraak(client, ecli)
            except Exception as e:  # noqa: BLE001
                logger.error("Fout bij ophalen %s: %s", ecli, e)
                counts["errors"] += 1
                continue

            counts["fetched"] += 1

            if args.keep_all or has_relevant_signal(uitspraak):
                out.write(uitspraak.model_dump_json() + "\n")
                out.flush()
                counts["kept"] += 1

            if counts["found"] % 50 == 0:
                logger.info(
                    "Voortgang: gevonden=%d opgehaald=%d behouden=%d overgeslagen=%d fouten=%d",
                    counts["found"], counts["fetched"], counts["kept"],
                    counts["skipped"], counts["errors"],
                )

    logger.info(
        "Klaar: gevonden=%d opgehaald=%d behouden=%d overgeslagen=%d fouten=%d",
        counts["found"], counts["fetched"], counts["kept"],
        counts["skipped"], counts["errors"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
