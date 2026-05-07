"""Extract candidate burger-citaten uit Rechtspraak-pulled JSONL.

Verwerkt één of meer rechtspraak-puller-output bestanden, zoekt directe
burger-citaten, en schrijft kandidaten naar een JSONL voor handmatige labeling.

Output-schema per regel:
    {
      "id": "...",
      "ecli": "ECLI:...",
      "source_url": "https://data.rechtspraak.nl/...",
      "instance": "Centrale Raad van Beroep",
      "decision_date": "2024-01-15",
      "subject": "Bestuursrecht",
      "quote": "<echte burgertekst>",
      "preceding_context": "<rechter-tekst vlak voor de quote>",
      "suggested_labels": ["bezwaar"],
      "suggestion_reason": "art 6:5; woord 'bezwaarschrift'",
      "quality_score": 0.78,
      "labels": []          ← LEEG, jij vult handmatig
    }

Labelingsprocedure:
1. Sorteer candidates.jsonl op quality_score (descending)
2. Loop door en kijk naar quote + preceding_context
3. Vul "labels": ["bezwaar"] (etc.) als het een echte burger-tekst is en
   het label klopt
4. Sla op als data/eval/gold_real.jsonl (alleen records met ingevulde labels)
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path

from govmodel.extractors.burger_citaten import extract_citations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="Rechtspraak puller-output JSONL bestanden")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.4,
                        help="Minimale quality score om kandidaat op te nemen")
    parser.add_argument("--min-quote-chars", type=int, default=80)
    parser.add_argument("--max-quote-chars", type=int, default=2000)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("extract_burger_citaten")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts = {"records_read": 0, "candidates_total": 0, "candidates_written": 0}
    written_per_score_band = {"0.4-0.6": 0, "0.6-0.8": 0, "0.8+": 0}
    written_per_label_suggestion: dict[str, int] = {}

    with args.output.open("w", encoding="utf-8") as out:
        for input_path in args.inputs:
            logger.info("Verwerken: %s", input_path)
            with input_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    counts["records_read"] += 1

                    text = rec.get("full_text") or ""
                    if not text:
                        continue

                    candidates = extract_citations(text)
                    counts["candidates_total"] += len(candidates)

                    for c in candidates:
                        if c.quality_score < args.min_score:
                            continue
                        if not (args.min_quote_chars <= len(c.quote) <= args.max_quote_chars):
                            continue

                        out_record = {
                            "id": f"rcs-{uuid.uuid4().hex[:10]}",
                            "ecli": rec.get("ecli"),
                            "source_url": rec.get("content_url"),
                            "instance": rec.get("instance"),
                            "decision_date": rec.get("decision_date"),
                            "subject": rec.get("subject"),
                            "quote": c.quote,
                            "preceding_context": c.preceding_context,
                            "suggested_labels": c.suggested_labels,
                            "suggestion_reason": c.suggestion_reason,
                            "quality_score": round(c.quality_score, 3),
                            # Leeg laten — handmatig invullen na verificatie
                            "labels": [],
                            "label_confidence": None,
                            "labeller_notes": "",
                        }
                        out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                        counts["candidates_written"] += 1

                        # Stats
                        if c.quality_score >= 0.8:
                            written_per_score_band["0.8+"] += 1
                        elif c.quality_score >= 0.6:
                            written_per_score_band["0.6-0.8"] += 1
                        else:
                            written_per_score_band["0.4-0.6"] += 1
                        for label in c.suggested_labels or ["__geen_suggestie__"]:
                            written_per_label_suggestion[label] = written_per_label_suggestion.get(label, 0) + 1

    logger.info("=" * 60)
    logger.info("KLAAR")
    logger.info("Records verwerkt: %d", counts["records_read"])
    logger.info("Citaat-kandidaten gevonden (vóór filter): %d", counts["candidates_total"])
    logger.info("Geschreven naar output: %d", counts["candidates_written"])
    logger.info("")
    logger.info("Per quality-score band:")
    for band, n in written_per_score_band.items():
        logger.info("  %-10s %d", band, n)
    logger.info("")
    logger.info("Suggested-label distributie:")
    for label, n in sorted(written_per_label_suggestion.items(), key=lambda x: -x[1]):
        logger.info("  %-30s %d", label, n)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
