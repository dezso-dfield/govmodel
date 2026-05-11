"""Sliced eval driver — load checkpoint, run N labelled JSONL slices,
emit per-slice metrics + regression-gate verdict.

Replaces (and eventually supersedes) `scripts/benchmark.py` for the
operational use-case where you want one Macro-F1 number per slice plus
a hard CI gate.

    python scripts/evaluate_slices.py \
        --checkpoint models/awb-classifier-v0.1 \
        --slice handwritten_realistic=data/eval/handwritten_realistic.jsonl \
        --slice handwritten_adversarial=data/eval/handwritten_adversarial.jsonl \
        --slice rechtspraak=data/eval/rechtspraak.jsonl \
        --slice robustness=data/eval/robustness_pack.jsonl \
        --out reports/eval-$(date +%F) \
        --gate-against reports/baseline/report.json \
        --max-drop 0.05
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _parse_slice(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(f"--slice expects name=path, got {arg!r}")
    name, _, path = arg.partition("=")
    return name.strip(), Path(path.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="HuggingFace model id or local checkpoint dir.")
    p.add_argument("--slice", dest="slices", action="append", type=_parse_slice,
                   required=True,
                   help="name=path.jsonl, repeatable.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gate-against", type=Path, default=None,
                   help="Previous report.json — fail when macro-F1 drops > --max-drop.")
    p.add_argument("--max-drop", type=float, default=0.05)
    p.add_argument("--require-slices", nargs="+",
                   default=("handwritten_realistic",))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Lazy heavy imports
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from govmodel.data import read_jsonl
    from govmodel.eval import EvalReport, evaluate_slice, regression_gate
    from govmodel.logging_setup import configure_logging
    from govmodel.schemas import LabeledExample
    from govmodel.training.awb_dataset import AwbDataset, make_collate_fn
    from govmodel.training.labels import LABEL_NAMES

    configure_logging()
    log = logging.getLogger("evaluate_slices")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("loading model", extra={"checkpoint": str(args.checkpoint), "device": str(device)})
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.checkpoint))
    model = model.to(device).eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate = make_collate_fn(pad_id)

    slice_reports = []
    for name, path in args.slices:
        rows = list(read_jsonl(path))
        if not rows:
            log.warning("empty slice", extra={"name": name, "path": str(path)})
            continue
        examples = [LabeledExample.model_validate(r) for r in rows]
        ds = AwbDataset(examples, tokenizer, max_length=512)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"])
                all_logits.append(outputs.logits.float().cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())
        logits = np.concatenate(all_logits, 0)
        labels = np.concatenate(all_labels, 0)
        probs = 1.0 / (1.0 + np.exp(-logits))

        report = evaluate_slice(name, probs, labels, LABEL_NAMES)
        slice_reports.append(report)
        log.info("slice done",
                 extra={"name": name, "n": report.n,
                        "macro_f1": report.macro_f1, "ece": report.ece})

    report = EvalReport(model_id=str(args.checkpoint), slices=slice_reports)
    json_path, md_path = report.write(args.out)
    log.info("report written", extra={"json": str(json_path), "md": str(md_path)})

    baseline = None
    if args.gate_against and args.gate_against.exists():
        baseline = json.loads(args.gate_against.read_text())["macro_f1_by_slice"]
    verdict = regression_gate(report, baseline,
                              max_drop=args.max_drop,
                              require_slices=args.require_slices)
    (args.out / "gate.json").write_text(json.dumps(verdict, indent=2))
    if not verdict["ok"]:
        log.error("regression gate FAILED", extra=verdict)
        return 1
    log.info("regression gate passed", extra=verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
