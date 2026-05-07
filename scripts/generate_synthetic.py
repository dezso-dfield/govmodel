"""Genereer synthetic Awb-typering voorbeelden via lokale LLM (LM Studio etc.).

Voorbeeld:
    python scripts/generate_synthetic.py \\
        --output data/synthetic/v0.1.jsonl \\
        --n-per-combination 3 \\
        --labels aanvraag bezwaar klacht \\
        --styles formal_short informal_medium

Standaard worden alle (label × style)-combinaties gegenereerd. Met 9 labels en
8 styles = 72 combinaties × n = totaal aantal voorbeelden per run.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from govmodel.llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMClient
from govmodel.synthetic import (
    LABEL_PROMPTS,
    MULTILABEL_COMBOS,
    STYLE_PROFILES,
    generate_examples,
    generate_multilabel_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-per-combination", type=int, default=3)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--styles", nargs="*", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-multilabel", action="store_true",
                        help="Genereer aanvullend multilabel-combinaties (bv. bezwaar+klacht)")
    parser.add_argument("--multilabel-n", type=int, default=4,
                        help="Aantal voorbeelden per (combo × style) bij multilabel")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("generate_synthetic")

    labels = args.labels or list(LABEL_PROMPTS.keys())
    styles = args.styles or list(STYLE_PROFILES.keys())

    invalid_labels = [lab for lab in labels if lab not in LABEL_PROMPTS]
    if invalid_labels:
        logger.error("Onbekende labels: %s", invalid_labels)
        return 1
    invalid_styles = [s for s in styles if s not in STYLE_PROFILES]
    if invalid_styles:
        logger.error("Onbekende styles: %s", invalid_styles)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    counts = {"generated": 0, "combinations": 0}
    total_combinations = len(labels) * len(styles)
    logger.info(
        "Genereren: %d labels × %d styles × %d voorbeelden = ~%d voorbeelden",
        len(labels), len(styles), args.n_per_combination,
        total_combinations * args.n_per_combination,
    )

    with LLMClient(base_url=args.base_url, model=args.model) as client, \
         args.output.open("a", encoding="utf-8") as out:
        for label in labels:
            for style in styles:
                counts["combinations"] += 1
                logger.info(
                    "[%d/%d] label=%s style=%s",
                    counts["combinations"], total_combinations, label, style,
                )
                for example in generate_examples(
                    client, label, style,
                    n=args.n_per_combination,
                    temperature=args.temperature,
                    rng=rng,
                ):
                    out.write(example.model_dump_json() + "\n")
                    out.flush()
                    counts["generated"] += 1

    if args.include_multilabel:
        logger.info("Genereren multilabel-combinaties: %d combos × ~4 styles × %d",
                    len(MULTILABEL_COMBOS), args.multilabel_n)
        with LLMClient(base_url=args.base_url, model=args.model) as client, \
             args.output.open("a", encoding="utf-8") as out:
            for example in generate_multilabel_examples(
                client, n_per_combo=args.multilabel_n,
                temperature=args.temperature, rng=rng,
            ):
                out.write(example.model_dump_json() + "\n")
                out.flush()
                counts["generated"] += 1

    logger.info("Klaar: %d voorbeelden in %s", counts["generated"], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
