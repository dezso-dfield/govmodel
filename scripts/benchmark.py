"""Benchmark een getraind Awb-classifier model op een JSONL eval-set.

Gebruik:
    python scripts/benchmark.py \\
        --model models/awb-classifier-v0.1/best \\
        --eval data/eval/gold_realistic.jsonl data/eval/adversarial.jsonl \\
        --output models/awb-classifier-v0.1/benchmark.json

Geeft per eval-bestand: macro/micro F1 (default threshold + tuned + argmax),
per-label F1 + precision/recall, prob_gap, en een verwarringsmatrix.

Als `--thresholds <pad>` wordt meegegeven (bv. final_metrics.json van de
training-run), worden die thresholds gebruikt voor de "tuned" metrics.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import sys
from pathlib import Path

faulthandler.enable(file=sys.stderr)

from govmodel.training.dataset import load_examples
from govmodel.training.labels import LABEL_NAMES, encode_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True,
                        help="Pad naar getraind model-directory (bv. .../best/)")
    parser.add_argument("--eval", type=Path, nargs="+", required=True,
                        help="Eén of meer JSONL eval-bestanden")
    parser.add_argument("--thresholds", type=Path, default=None,
                        help="Optioneel: final_metrics.json met getunede thresholds. "
                             "Indien afwezig: alleen default 0.5 + argmax + per-bestand tuning")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON met alle benchmark-metrics")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("benchmark")

    # Lazy imports
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from govmodel.training.train import (
        compute_confusion,
        compute_metrics_multilabel,
        compute_with_thresholds,
        find_best_thresholds_per_class,
    )

    class AwbDataset(TorchDataset):
        def __init__(self, examples, tokenizer, max_length: int):
            self.encodings = tokenizer(
                [ex.text for ex in examples], truncation=True,
                max_length=max_length, padding=False,
            )
            self.labels = [encode_labels(ex.labels) for ex in examples]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": torch.tensor(self.labels[idx], dtype=torch.float),
            }

    def make_collate(pad_token_id: int):
        def collate(batch):
            max_len = max(len(it["input_ids"]) for it in batch)
            input_ids = torch.tensor([
                it["input_ids"] + [pad_token_id] * (max_len - len(it["input_ids"]))
                for it in batch
            ], dtype=torch.long)
            attention_mask = torch.tensor([
                it["attention_mask"] + [0] * (max_len - len(it["attention_mask"]))
                for it in batch
            ], dtype=torch.long)
            labels = torch.stack([it["labels"] for it in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        return collate

    logger.info("Model laden uit: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    logger.info("Model geladen op %s, params: %d M",
                device, sum(p.numel() for p in model.parameters()) // 1_000_000)

    pretrained_thresholds = None
    if args.thresholds:
        meta = json.loads(args.thresholds.read_text(encoding="utf-8"))
        if "best_thresholds" in meta:
            pretrained_thresholds = np.array(meta["best_thresholds"], dtype=float)
            logger.info("Getunede thresholds geladen uit %s", args.thresholds)

    autocast_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    use_bf16 = autocast_dtype == torch.bfloat16

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate = make_collate(pad_id)

    benchmark: dict[str, object] = {
        "model_path": str(args.model),
        "thresholds_source": str(args.thresholds) if args.thresholds else None,
        "evaluations": {},
    }

    for eval_path in args.eval:
        logger.info("=== Eval-bestand: %s ===", eval_path)
        examples = list(load_examples(eval_path))
        if not examples:
            logger.warning("Lege eval-set, overslaan: %s", eval_path)
            continue

        dist: dict[str, int] = {}
        for ex in examples:
            for lbl in ex.labels:
                dist[lbl] = dist.get(lbl, 0) + 1
        logger.info("Aantal voorbeelden: %d, label-distributie: %s", len(examples), dist)

        ds = AwbDataset(examples, tokenizer, args.max_length)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                    dtype=autocast_dtype, enabled=use_bf16):
                    outputs = model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"])
                all_logits.append(outputs.logits.float().cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())
        logits = np.concatenate(all_logits, 0)
        labels = np.concatenate(all_labels, 0)

        default_metrics = compute_metrics_multilabel((logits, labels))
        confusion = compute_confusion(logits, labels, np.full(len(LABEL_NAMES), 0.5))

        eval_result: dict[str, object] = {
            "n_examples": len(examples),
            "label_distribution": dist,
            "default_threshold_05": default_metrics,
            "confusion_argmax": confusion,
        }

        if pretrained_thresholds is not None:
            eval_result["pretrained_thresholds"] = compute_with_thresholds(
                logits, labels, pretrained_thresholds
            )

        # Self-tuned thresholds op deze set (alleen indicatief — eigenlijk over-fit)
        if any(labels[:, i].sum() > 0 for i in range(labels.shape[1])):
            self_thresholds = find_best_thresholds_per_class(logits, labels)
            eval_result["self_tuned_thresholds_indicative"] = compute_with_thresholds(
                logits, labels, self_thresholds
            )
            eval_result["self_tuned_thresholds_indicative"]["thresholds_per_label"] = {
                LABEL_NAMES[i]: float(t) for i, t in enumerate(self_thresholds)
            }

        # Logging
        logger.info(
            "  default-thr macro_f1=%.4f  argmax_macro_f1=%.4f  prob_gap=%.3f",
            default_metrics["macro_f1"], default_metrics["argmax_macro_f1"],
            default_metrics["prob_gap"],
        )
        if pretrained_thresholds is not None:
            logger.info("  pretrained-thr macro_f1=%.4f",
                        eval_result["pretrained_thresholds"]["macro_f1_tuned"])

        benchmark["evaluations"][str(eval_path)] = eval_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Benchmark opgeslagen: %s", args.output)

    # Compact stdout-summary voor snelle scan
    logger.info("")
    logger.info("=" * 70)
    logger.info("%-35s %10s %10s %10s", "EVAL SET", "n", "argmax_F1", "default_F1")
    logger.info("-" * 70)
    for path, res in benchmark["evaluations"].items():
        logger.info("%-35s %10d %10.4f %10.4f",
                    Path(path).name, res["n_examples"],
                    res["default_threshold_05"]["argmax_macro_f1"],
                    res["default_threshold_05"]["macro_f1"])
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
