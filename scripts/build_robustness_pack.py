"""Generate a per-perturbation robustness eval set.

Each input row from a gold/labelled JSONL is replicated once per
standard augmentation:

    {
      "text": "Hierbij teken ik bezwaar... typos appplied",
      "labels": ["bezwaar"],
      "source": "robustness:typos_3pct",
      "_origin": "gold-0042",
      "_perturb": "typos_3pct",
    }

`scripts/evaluate_slices.py` then reports per-perturbation macro-F1 and
the next training cycle knows exactly which surface variant the model
is fragile to.

    python scripts/build_robustness_pack.py \
        --in tests/handwritten_realistic.jsonl \
        --out data/eval/robustness_pack.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from govmodel.data import apply_all, read_jsonl, standard_augmentations, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", type=Path, required=True,
                   help="Input JSONL of labelled examples (e.g. gold or handwritten realistic).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL with one row per (input × augmentation).")
    p.add_argument("--field", default="text",
                   help="Key whose value gets perturbed (default: text).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    augs = standard_augmentations(seed=args.seed)
    rows_in = list(read_jsonl(args.in_path))
    out_rows: list[dict] = []
    for i, row in enumerate(rows_in):
        origin_id = str(row.get("id") or f"row_{i}")
        text = row.get(args.field, "")
        if not isinstance(text, str) or not text.strip():
            continue
        for name, perturbed in apply_all(text, augs):
            new_row = dict(row)
            new_row[args.field] = perturbed
            new_row["_origin"] = origin_id
            new_row["_perturb"] = name
            new_row["source"] = f"robustness:{name}"
            out_rows.append(new_row)
    n = write_jsonl(args.out, out_rows)
    print(json.dumps({
        "input": str(args.in_path),
        "output": str(args.out),
        "n_input": len(rows_in),
        "n_output": n,
        "augmentations": [a.name for a in augs],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
