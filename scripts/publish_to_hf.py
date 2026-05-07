"""Upload het getrainde model en optioneel de dataset naar HuggingFace Hub.

Vereist:
    pip install huggingface_hub
    huggingface-cli login   # plak je write-token

Voorbeeld — alleen model uploaden:
    python scripts/publish_to_hf.py \\
        --model-dir models/awb-classifier-v0.2/best \\
        --repo-id NoaberAI/govmodel-awb-classifier-v0.1 \\
        --model-card MODEL_CARD.md

Voorbeeld — model + dataset:
    python scripts/publish_to_hf.py \\
        --model-dir models/awb-classifier-v0.2/best \\
        --repo-id NoaberAI/govmodel-awb-classifier-v0.1 \\
        --model-card MODEL_CARD.md \\
        --dataset-files data/processed/synthetic/v0.2.jsonl data/eval/gold_realistic.jsonl data/eval/adversarial.jsonl data/eval/gold_real.jsonl \\
        --dataset-repo-id NoaberAI/govmodel-awb-data-v0.1 \\
        --datasheet DATASHEET.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model-upload (vereist)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Pad naar getraind model directory (bv. models/awb-classifier-v0.2/best)")
    parser.add_argument("--repo-id", required=True,
                        help="HF model repo (bv. NoaberAI/govmodel-awb-classifier-v0.1)")
    parser.add_argument("--model-card", type=Path, default=Path("MODEL_CARD.md"),
                        help="Markdown-file om als README.md naar het model-repo te uploaden")
    parser.add_argument("--include-final-metrics", type=Path, default=None,
                        help="Pad naar final_metrics.json om mee te uploaden (extra context voor adopters)")
    parser.add_argument("--include-history", type=Path, default=None,
                        help="Pad naar history.json")
    parser.add_argument("--private", action="store_true",
                        help="Maak het model-repo private (default: public)")

    # Dataset-upload (optioneel)
    parser.add_argument("--dataset-files", type=Path, nargs="*", default=None,
                        help="Optioneel: JSONL-bestanden om als dataset-repo te uploaden")
    parser.add_argument("--dataset-repo-id", default=None,
                        help="HF dataset repo (bv. NoaberAI/govmodel-awb-data-v0.1)")
    parser.add_argument("--datasheet", type=Path, default=Path("DATASHEET.md"),
                        help="Markdown-file om als README.md naar het dataset-repo te uploaden")
    parser.add_argument("--dataset-private", action="store_true")

    parser.add_argument("--dry-run", action="store_true",
                        help="Toon wat geüpload zou worden, doe niets daadwerkelijk")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("publish_to_hf")

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("huggingface_hub niet geïnstalleerd. Run: pip install huggingface_hub")
        return 1

    if not args.model_dir.exists():
        logger.error("Model-dir niet gevonden: %s", args.model_dir)
        return 1
    if not args.model_card.exists():
        logger.error("Model card niet gevonden: %s", args.model_card)
        return 1

    api = HfApi()

    # ---- Model upload ----
    logger.info("=== Model upload ===")
    logger.info("Repo: %s (private=%s)", args.repo_id, args.private)
    logger.info("Bron: %s", args.model_dir)

    files_in_model = sorted(p for p in args.model_dir.iterdir() if p.is_file())
    logger.info("Bestanden in model-dir (%d):", len(files_in_model))
    for f in files_in_model:
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info("  %-40s  %6.1f MB", f.name, size_mb)

    if args.dry_run:
        logger.info("[dry-run] zou repo aanmaken en uploaden")
    else:
        logger.info("Repo aanmaken (idempotent)...")
        create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

        logger.info("Model card uploaden als README.md...")
        api.upload_file(
            path_or_fileobj=str(args.model_card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add model card",
        )

        logger.info("Model-bestanden uploaden...")
        api.upload_folder(
            folder_path=str(args.model_dir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload model from {args.model_dir.name}",
        )

        if args.include_final_metrics and args.include_final_metrics.exists():
            logger.info("Upload final_metrics.json voor context...")
            api.upload_file(
                path_or_fileobj=str(args.include_final_metrics),
                path_in_repo="training_metrics.json",
                repo_id=args.repo_id,
                repo_type="model",
                commit_message="Add training metrics",
            )

        if args.include_history and args.include_history.exists():
            api.upload_file(
                path_or_fileobj=str(args.include_history),
                path_in_repo="training_history.json",
                repo_id=args.repo_id,
                repo_type="model",
                commit_message="Add training history",
            )

        logger.info("✅ Model upload klaar: https://huggingface.co/%s", args.repo_id)

    # ---- Dataset upload (optioneel) ----
    if args.dataset_files and args.dataset_repo_id:
        logger.info("")
        logger.info("=== Dataset upload ===")
        logger.info("Repo: %s (private=%s)", args.dataset_repo_id, args.dataset_private)
        for f in args.dataset_files:
            if not f.exists():
                logger.error("Dataset-bestand niet gevonden: %s", f)
                return 1
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info("  %-50s  %6.2f MB", str(f), size_mb)

        if args.dry_run:
            logger.info("[dry-run] zou dataset-repo aanmaken en uploaden")
        else:
            logger.info("Dataset-repo aanmaken (idempotent)...")
            create_repo(args.dataset_repo_id, repo_type="dataset",
                        private=args.dataset_private, exist_ok=True)

            if args.datasheet.exists():
                logger.info("Datasheet uploaden als README.md...")
                api.upload_file(
                    path_or_fileobj=str(args.datasheet),
                    path_in_repo="README.md",
                    repo_id=args.dataset_repo_id,
                    repo_type="dataset",
                    commit_message="Add datasheet",
                )

            for f in args.dataset_files:
                logger.info("Upload %s...", f.name)
                # Plaats per type onder een subdir voor netheid
                if "synthetic" in str(f):
                    target = f"synthetic/{f.name}"
                elif "eval" in str(f):
                    target = f"eval/{f.name}"
                else:
                    target = f.name
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=target,
                    repo_id=args.dataset_repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload {f.name}",
                )

            logger.info("✅ Dataset upload klaar: https://huggingface.co/datasets/%s", args.dataset_repo_id)

    logger.info("")
    logger.info("Klaar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
