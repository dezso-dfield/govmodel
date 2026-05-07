"""Pseudonimiseer PII in een JSONL-bestand.

Leest een JSONL met records die een tekstveld bevatten (default: 'text' of
'full_text'), redacteert PII, en schrijft naar een nieuw JSONL.

Voorbeeld:
    python scripts/pseudonymize.py \\
        --input data/raw/raadsinformatie/sample.jsonl \\
        --output data/processed/raadsinformatie/sample.jsonl \\
        --text-field text
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from govmodel.pii import pseudonymize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-field", default="text",
                        help="Veld in JSONL dat de te redacteren tekst bevat")
    parser.add_argument("--audit-output", type=Path, default=None,
                        help="Optioneel: schrijf redactie-audit naar dit bestand")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("pseudonymize")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)

    counts = {"records": 0, "redacted_records": 0, "total_redactions": 0}
    by_pattern: dict[str, int] = {}

    audit_handle = args.audit_output.open("w", encoding="utf-8") if args.audit_output else None
    try:
        with args.input.open("r", encoding="utf-8") as inp, \
             args.output.open("w", encoding="utf-8") as out:
            for line in inp:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                counts["records"] += 1

                text = record.get(args.text_field)
                if not isinstance(text, str) or not text:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                redacted, redactions = pseudonymize(text)
                if redactions:
                    counts["redacted_records"] += 1
                    counts["total_redactions"] += len(redactions)
                    for r in redactions:
                        by_pattern[r.pattern_name] = by_pattern.get(r.pattern_name, 0) + 1
                    if audit_handle:
                        audit_handle.write(json.dumps({
                            "record_id": record.get("id") or record.get("ecli"),
                            "redactions": [
                                {
                                    "pattern": r.pattern_name,
                                    "original": r.original,
                                    "start": r.start,
                                    "end": r.end,
                                }
                                for r in redactions
                            ],
                        }, ensure_ascii=False) + "\n")

                record[args.text_field] = redacted
                record["pseudonymized"] = True
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if audit_handle:
            audit_handle.close()

    logger.info(
        "Klaar: %d records, %d met redacties, %d redacties totaal",
        counts["records"], counts["redacted_records"], counts["total_redactions"],
    )
    if by_pattern:
        for name, count in sorted(by_pattern.items()):
            logger.info("  %s: %d", name, count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
