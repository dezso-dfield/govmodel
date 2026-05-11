"""Pre-train dataset hygiene driver.

Runs the full data-quality pass on one or more JSONL files:

    python scripts/audit_dataset.py \
        --inputs data/synthetic/v0.1.jsonl \
        --eval-against tests/burger_citaten.jsonl \
        --out reports/audit-v0.1.json

Reports schema issues, label distribution, multilabel coverage, exact
duplicates, near-duplicates, and train↔eval leakage. Exits with code
1 if any blocking issue is found (corrupt rows, missing required
fields, train↔eval leakage).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from govmodel.data import (
    balance_report,
    exact_dedupe,
    issues_summary,
    leakage_between,
    multilabel_distribution,
    near_dedupe,
    read_jsonl,
    validate_examples,
)
from govmodel.logging_setup import configure_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, nargs="+", required=True,
                   help="One or more JSONL files to audit (will be concatenated).")
    p.add_argument("--eval-against", type=Path, nargs="*", default=None,
                   help="Optional eval JSONL(s) to check for train↔eval leakage.")
    p.add_argument("--out", type=Path, required=True,
                   help="JSON path where the full audit report lands.")
    p.add_argument("--near-dedupe-threshold", type=float, default=0.85)
    p.add_argument("--leakage-threshold", type=float, default=0.85)
    p.add_argument("--strict", action="store_true",
                   help="Treat unknown top-level fields as issues.")
    p.add_argument("--fail-on-leakage", action="store_true", default=True)
    p.add_argument("--no-fail-on-leakage", dest="fail_on_leakage", action="store_false")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    log = logging.getLogger("audit_dataset")

    rows: list[dict] = []
    for path in args.inputs:
        rows_in = list(read_jsonl(path))
        log.info("loaded", extra={"path": str(path), "n": len(rows_in)})
        rows.extend(rows_in)

    issues = validate_examples(rows, strict=args.strict)
    summary = issues_summary(issues)

    exact_unique = exact_dedupe(rows)
    near_unique = near_dedupe(rows, threshold=args.near_dedupe_threshold)
    balance = balance_report(rows)
    multi = multilabel_distribution(rows)

    leakage: list[dict] = []
    if args.eval_against:
        for eval_path in args.eval_against:
            eval_rows = list(read_jsonl(eval_path))
            matches = leakage_between(
                rows, eval_rows,
                threshold=args.leakage_threshold,
            )
            for ti, tri, j in matches:
                leakage.append({
                    "eval_file": str(eval_path),
                    "eval_index": ti,
                    "train_index": tri,
                    "jaccard": round(j, 4),
                })
            n_overlaps_here = sum(
                1 for m in leakage if m["eval_file"] == str(eval_path)
            )
            log.info("leakage check", extra={
                "eval_file": str(eval_path),
                "n_eval": len(eval_rows),
                "n_overlaps": n_overlaps_here,
            })

    blocking_issue_kinds = {"bad_type", "missing_field", "labels_empty"}
    blocking_issues = [i for i in issues if i.kind in blocking_issue_kinds]

    report = {
        "inputs": [str(p) for p in args.inputs],
        "eval_inputs": [str(p) for p in (args.eval_against or [])],
        "n_total": len(rows),
        "n_exact_unique": len(exact_unique),
        "n_exact_duplicates": len(rows) - len(exact_unique),
        "n_near_unique": len(near_unique),
        "n_near_duplicates": len(rows) - len(near_unique),
        "near_dedupe_threshold": args.near_dedupe_threshold,
        "validation": {
            "n_issues": len(issues),
            "by_kind": summary,
            "n_blocking": len(blocking_issues),
            "sample": [{"row": i.row_index, "kind": i.kind, "detail": i.detail}
                       for i in issues[:20]],
        },
        "balance": balance,
        "multilabel_distribution": multi,
        "leakage": {
            "threshold": args.leakage_threshold,
            "n_overlaps": len(leakage),
            "sample": leakage[:20],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote audit report", extra={"path": str(args.out)})

    fail = bool(blocking_issues) or (args.fail_on_leakage and leakage)
    if fail:
        log.error("audit FAILED",
                  extra={"n_blocking": len(blocking_issues), "n_leakage": len(leakage)})
        return 1
    log.info("audit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
