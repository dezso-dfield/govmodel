"""Build a labelling queue from the prediction JSONL log.

Reads the rolling log emitted by the Gradio demo and the FastAPI
service, ranks candidates by how much labelling each one would teach
the next model, and writes a JSONL the labelling tool consumes.

    python scripts/mine_hard_negatives.py \
        --log logs/predictions.jsonl \
        --out data/labeling/queue.jsonl \
        --limit 200

The default selection covers three signals (low-confidence, conflicted
multilabel, user disagreement). Use --user-only if you only want to
re-train on rows that already have human-supplied labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from govmodel.logging_setup import configure_logging
from govmodel.mining import MiningConfig, mine, stats, to_label_queue


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log", type=Path, default=Path("logs/predictions.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--abstain-threshold", type=float, default=0.20)
    parser.add_argument("--low-conf-min", type=float, default=0.20)
    parser.add_argument("--low-conf-max", type=float, default=0.50)
    parser.add_argument("--conflict-margin", type=float, default=0.15)
    parser.add_argument("--multilabel-min-above", type=float, default=0.40)
    parser.add_argument("--user-only", action="store_true")
    args = parser.parse_args()

    configure_logging()

    cfg = MiningConfig(
        abstain_threshold=args.abstain_threshold,
        low_conf_window=(args.low_conf_min, args.low_conf_max),
        conflict_margin=args.conflict_margin,
        multilabel_min_above=args.multilabel_min_above,
        user_disagreement_only=args.user_only,
    )
    cands = mine(args.log, cfg, limit=args.limit)
    n = to_label_queue(cands, args.out)
    summary = stats(cands)
    print(json.dumps({"log": str(args.log), "out": str(args.out),
                      "n_written": n, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
