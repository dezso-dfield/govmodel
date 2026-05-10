"""Train de Awb-type-classifier op één of meer JSONL-databronnen.

Vereisten:
    pip install -e ".[ml]"
    pip install torch --index-url https://download.pytorch.org/whl/cu124  # voor GPU

Voorbeeld:
    python scripts/train_classifier.py \\
        --inputs data/synthetic/v0.1.jsonl \\
        --output-dir models/awb-classifier-v0.1 \\
        --epochs 5 \\
        --batch-size 8

GPU-gebruik: ondersteunt bf16 op Ampere+ (RTX 3050 Laptop heeft compute 8.6).
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import sys
from pathlib import Path

# faulthandler toont native stacktraces bij CUDA/torch segfaults (anders krijg
# je alleen een exit-code zonder traceback)
faulthandler.enable(file=sys.stderr)

from govmodel.logging_setup import configure_logging
from govmodel.seeding import seed_everything
from govmodel.training.dataset import (
    label_distribution,
    load_all_examples,
    save_split,
    source_distribution,
    split_train_val_test,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="Eén of meer JSONL-databronnen voor train/val")
    parser.add_argument("--eval-inputs", type=Path, nargs="*", default=None,
                        help="Optioneel: aparte JSONL-bronnen voor de test/eval set "
                             "(bv. adversarial gold). Als gegeven: --inputs gaat 100%% "
                             "naar train/val (stratified split), eval-inputs zijn de test-set")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="DTAI-KULeuven/robbert-2023-dutch-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Effectieve batch = batch_size × accumulation_steps")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Wisselt geheugen voor compute (handig bij krappe VRAM)")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pos-weight", type=float, default=8.0,
                        help="BCE pos_weight; balanceert sparse multilabel (1/9 prior → ~8.0)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="stratified", choices=["stratified", "random"],
                        help="Stratified houdt elke class proportioneel in train/val/test (default)")
    parser.add_argument("--save-splits", action="store_true",
                        help="Schrijf train/val/test splits naar output_dir/splits/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Alleen dataset laden en splitten, geen training")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(level=args.log_level)
    logger = logging.getLogger("train_classifier")
    seed_everything(args.seed)

    logger.info("Laden uit %d bron(nen): %s", len(args.inputs), args.inputs)
    examples = load_all_examples(args.inputs)
    if not examples:
        logger.error("Geen voorbeelden gevonden")
        return 1

    label_dist = label_distribution(examples)
    source_dist = source_distribution(examples)
    logger.info("Totaal: %d voorbeelden", len(examples))
    logger.info("Labels: %s", label_dist)
    logger.info("Bronnen: %s", source_dist)

    splitter = stratified_split if args.split == "stratified" else split_train_val_test

    if args.eval_inputs:
        # Externe test-set: --inputs gaat naar train+val, --eval-inputs is de test
        eval_examples = load_all_examples(args.eval_inputs)
        logger.info("Externe test-set geladen: %d voorbeelden uit %s",
                    len(eval_examples), args.eval_inputs)
        # Splits inputs in train+val (resterende ratio gaat naar val)
        # Hergebruik splitter met test_ratio=0 door val_ratio aan te passen
        adj_train_ratio = args.train_ratio / (args.train_ratio + args.val_ratio)
        train, val, leftover = splitter(
            examples,
            train_ratio=adj_train_ratio,
            val_ratio=1.0 - adj_train_ratio - 1e-6,
            seed=args.seed,
        )
        # leftover hoort hier 0 of klein te zijn
        if leftover:
            val.extend(leftover)
        test = eval_examples
        logger.info("Split (%s + extern): train=%d val=%d test=%d (extern)",
                    args.split, len(train), len(val), len(test))
    else:
        train, val, test = splitter(
            examples, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed,
        )
        logger.info("Split (%s): train=%d val=%d test=%d",
                    args.split, len(train), len(val), len(test))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_splits:
        splits_dir = args.output_dir / "splits"
        save_split(train, splits_dir / "train.jsonl")
        save_split(val, splits_dir / "val.jsonl")
        save_split(test, splits_dir / "test.jsonl")
        logger.info("Splits opgeslagen in %s", splits_dir)

    def _json_safe(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [_json_safe(x) for x in v]
        return v

    metadata = {
        "n_examples": len(examples),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "label_distribution": label_dist,
        "source_distribution": source_dist,
        "args": {k: _json_safe(v) for k, v in vars(args).items()},
    }
    (args.output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Dataset metadata: %s", args.output_dir / "dataset_metadata.json")

    if args.dry_run:
        logger.info("--dry-run: training overgeslagen")
        return 0

    # Lazy import om torch/transformers alleen te laden wanneer nodig
    try:
        from govmodel.training.train import train_classifier
    except ImportError as e:
        logger.error("ML deps ontbreken: %s. Install met: pip install -e \".[ml]\"", e)
        return 1

    logger.info("Start training: base=%s epochs=%d batch=%d lr=%g pos_weight=%.1f",
                args.base_model, args.epochs, args.batch_size, args.learning_rate, args.pos_weight)
    try:
        result, final = train_classifier(
            train, val,
            output_dir=args.output_dir,
            test_examples=test,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            pos_weight_value=args.pos_weight,
        )
    except BaseException:
        import traceback
        logger.error("TRAINING GECRASHED — full traceback:")
        traceback.print_exc()
        raise

    (args.output_dir / "history.json").write_text(
        json.dumps(result["history"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Artifacts: %s, %s",
                args.output_dir / "history.json", args.output_dir / "final_metrics.json")

    # Tabelletje samenvatten naar stdout voor menselijke leesbaarheid
    val_tuned = final["val"]["tuned_threshold"]["macro_f1_tuned"]
    val_argmax = final["val"]["default_threshold"]["argmax_macro_f1"]
    summary_lines = [
        "",
        "=" * 60,
        f"BEST EPOCH: {final['best_epoch']}",
        f"VAL  macro-F1 (tuned thresholds): {val_tuned:.4f}",
        f"VAL  macro-F1 (argmax single-label): {val_argmax:.4f}",
    ]
    if "test" in final:
        test_tuned = final["test"]["tuned_threshold"]["macro_f1_tuned"]
        test_argmax = final["test"]["default_threshold"]["argmax_macro_f1"]
        summary_lines.append(f"TEST macro-F1 (tuned thresholds): {test_tuned:.4f}")
        summary_lines.append(f"TEST macro-F1 (argmax single-label): {test_argmax:.4f}")
    summary_lines.append("=" * 60)
    for line in summary_lines:
        logger.info(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
