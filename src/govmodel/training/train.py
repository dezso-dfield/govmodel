"""Train een multilabel Awb-type-classifier op RobBERT/BERTje.

Vereist optional dependencies uit pyproject.toml [ml]:
    pip install -e ".[ml]"
    pip install torch --index-url https://download.pytorch.org/whl/cu124  # voor GPU

Het model wordt opgeslagen in ``output_dir`` en bevat een config met
``id2label``/``label2id`` zodat het zonder code-wijziging laadbaar is via
HuggingFace ``pipeline("text-classification")``.
"""

from __future__ import annotations

import logging
from pathlib import Path

# Ml-imports staan binnen functies om import-tijd licht te houden voor tests
# en CLI-help die geen torch nodig hebben.

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "DTAI-KULeuven/robbert-2023-dutch-base"


def compute_metrics_multilabel(eval_pred, threshold: float = 0.5):
    """Compute multiple multilabel metrics, plus argmax-based single-label metrics
    voor sanity-check (BCE op sparse multilabel kan threshold-F1=0 geven terwijl
    het model wel degelijk leert)."""
    import numpy as np
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    from govmodel.training.labels import LABEL_NAMES

    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels_int = labels.astype(int)

    # Threshold-based predictions (klassiek multilabel)
    preds_thr = (probs >= threshold).astype(int)

    # Argmax-based predictions (single-label aanname; sanity check)
    preds_argmax = np.zeros_like(preds_thr)
    preds_argmax[np.arange(len(probs)), probs.argmax(axis=1)] = 1

    # Diagnostiek: gemiddelde kans op de TRUE class
    true_mask = labels_int.astype(bool)
    mean_prob_true = float(probs[true_mask].mean()) if true_mask.any() else 0.0
    mean_prob_false = float(probs[~true_mask].mean()) if (~true_mask).any() else 0.0

    metrics: dict[str, float] = {
        "macro_f1": float(f1_score(labels_int, preds_thr, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels_int, preds_thr, average="micro", zero_division=0)),
        "argmax_macro_f1": float(f1_score(labels_int, preds_argmax, average="macro", zero_division=0)),
        "argmax_micro_f1": float(f1_score(labels_int, preds_argmax, average="micro", zero_division=0)),
        "mean_prob_true_class": mean_prob_true,
        "mean_prob_false_class": mean_prob_false,
        "prob_gap": mean_prob_true - mean_prob_false,
    }

    _, _, f, _ = precision_recall_fscore_support(
        labels_int, preds_argmax, average=None, zero_division=0,
        labels=list(range(len(LABEL_NAMES))),
    )
    for i, name in enumerate(LABEL_NAMES):
        metrics[f"f1_{name}_argmax"] = float(f[i])
    return metrics


def train_classifier(
    train_examples,
    val_examples,
    output_dir: Path,
    test_examples=None,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 10,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 3e-5,
    max_length: int = 512,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    use_bf16: bool | None = None,
    gradient_checkpointing: bool = False,
    pos_weight_value: float = 8.0,
):
    """Train het model en retourneer de Trainer.

    Imports torch/transformers laat — alleen aanroepen als ml deps geïnstalleerd zijn.
    """
    import numpy as np
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    # NIET importeren: transformers.Trainer of transformers.TrainingArguments —
    # die triggeren een import-chain naar datasets→aiohttp→ssl die op Windows crasht.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup

    from govmodel.training.labels import LABEL_NAMES, encode_labels

    class AwbDataset(TorchDataset):
        """Pure-torch dataset — vermijdt HF datasets/pyarrow."""

        def __init__(self, examples, tokenizer, max_length: int):
            self.encodings = tokenizer(
                [ex.text for ex in examples],
                truncation=True,
                max_length=max_length,
                padding=False,
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

    def make_collate_fn(pad_token_id: int):
        def collate(batch: list[dict]) -> dict:
            max_len = max(len(item["input_ids"]) for item in batch)
            input_ids = torch.tensor([
                item["input_ids"] + [pad_token_id] * (max_len - len(item["input_ids"]))
                for item in batch
            ], dtype=torch.long)
            attention_mask = torch.tensor([
                item["attention_mask"] + [0] * (max_len - len(item["attention_mask"]))
                for item in batch
            ], dtype=torch.long)
            labels = torch.stack([item["labels"] for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        return collate

    if use_bf16 is None:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/7] CUDA available: %s, bf16: %s", torch.cuda.is_available(), use_bf16)
    if torch.cuda.is_available():
        logger.info("[1/7] GPU: %s, free VRAM: %.0f MiB",
                    torch.cuda.get_device_name(0),
                    (torch.cuda.mem_get_info()[0] / 1024 / 1024))

    logger.info("[2/7] Tokenizer laden: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    logger.info("[3/7] Datasets bouwen + tokenizen (pure torch, geen pyarrow)")
    train_ds = AwbDataset(train_examples, tokenizer, max_length)
    val_ds = AwbDataset(val_examples, tokenizer, max_length)

    logger.info("[4/7] Model laden: %s", base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL_NAMES),
        problem_type="multi_label_classification",
        id2label={i: lbl for i, lbl in enumerate(LABEL_NAMES)},
        label2id={lbl: i for i, lbl in enumerate(LABEL_NAMES)},
    )
    logger.info("[4/7] Model geladen, params: %d M", sum(p.numel() for p in model.parameters()) // 1_000_000)

    logger.info("[5/7] DataLoaders bouwen (pure pytorch)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate = make_collate_fn(pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Class-balanced BCE: positief monster wordt zwaarder gewogen (1/9 prior → 8x).
    # Zonder dit blijft sigmoid hangen rond 0.1 en geeft threshold 0.5 altijd F1=0.
    pos_weight = torch.full((len(LABEL_NAMES),), pos_weight_value,
                            dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def evaluate_loader(loader):
        model.eval()
        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                    dtype=autocast_dtype, enabled=use_bf16):
                    outputs = model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"])
                all_logits.append(outputs.logits.float().cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())
        return np.concatenate(all_logits, 0), np.concatenate(all_labels, 0)

    logger.info("[6/7] Training loop starten — %d epochs × %d steps/epoch (effective batch %d), "
                "pos_weight=%.1f, lr=%g, bf16=%s",
                epochs, len(train_loader), batch_size * gradient_accumulation_steps,
                pos_weight_value, learning_rate, use_bf16)

    best_score = -1.0
    best_epoch = 0
    best_val_logits = None
    best_val_labels = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels_in = batch["labels"]
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=autocast_dtype, enabled=use_bf16):
                outputs = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"])
                loss = loss_fn(outputs.logits.float(), labels_in) / gradient_accumulation_steps
            loss.backward()
            if step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            running_loss += loss.item() * gradient_accumulation_steps
            n_batches += 1

            if step % 10 == 0:
                logger.info("epoch=%d step=%d/%d loss=%.4f lr=%.2e",
                            epoch, step, len(train_loader), running_loss / n_batches,
                            scheduler.get_last_lr()[0])

        avg_train_loss = running_loss / max(n_batches, 1)

        val_logits, val_labels = evaluate_loader(val_loader)
        metrics = compute_metrics_multilabel((val_logits, val_labels))
        metrics["epoch"] = epoch
        metrics["train_loss"] = avg_train_loss
        history.append(metrics)
        logger.info(
            "EPOCH %d eval: thr_macro_f1=%.4f  argmax_macro_f1=%.4f  argmax_micro_f1=%.4f  "
            "prob_gap=%.3f (true=%.3f vs false=%.3f)  train_loss=%.4f",
            epoch, metrics["macro_f1"], metrics["argmax_macro_f1"], metrics["argmax_micro_f1"],
            metrics["prob_gap"], metrics["mean_prob_true_class"],
            metrics["mean_prob_false_class"], avg_train_loss,
        )

        # Selecteer beste model op argmax_macro_f1 (robuuster dan threshold-F1
        # die door pos_weight nu wel weer relevant kan worden)
        score = max(metrics["argmax_macro_f1"], metrics["macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_logits = val_logits
            best_val_labels = val_labels
            best_dir = output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            logger.info("→ Best model opgeslagen (epoch=%d, score=%.4f)", epoch, best_score)

    # ---- threshold tuning per class op val ----
    thresholds = find_best_thresholds_per_class(best_val_logits, best_val_labels)
    val_tuned = compute_with_thresholds(best_val_logits, best_val_labels, thresholds)
    logger.info("Best per-class thresholds: %s", {LABEL_NAMES[i]: round(t, 2) for i, t in enumerate(thresholds)})
    logger.info("Tuned val macro_f1=%.4f (vs default-thr %.4f, argmax %.4f)",
                val_tuned["macro_f1_tuned"],
                history[best_epoch - 1]["macro_f1"],
                history[best_epoch - 1]["argmax_macro_f1"])

    # ---- final report (val + optionele test) ----
    final = {
        "best_epoch": best_epoch,
        "best_thresholds": [float(t) for t in thresholds],
        "thresholds_per_label": {LABEL_NAMES[i]: float(t) for i, t in enumerate(thresholds)},
        "val": {
            "default_threshold": _strip_meta(history[best_epoch - 1]),
            "tuned_threshold": val_tuned,
        },
    }

    if test_examples:
        logger.info("Test-set evaluatie (n=%d) met beste model + getunede thresholds",
                    len(test_examples))
        test_ds = AwbDataset(test_examples, tokenizer, max_length)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
        test_logits, test_labels = evaluate_loader(test_loader)
        test_default = compute_metrics_multilabel((test_logits, test_labels))
        test_tuned = compute_with_thresholds(test_logits, test_labels, thresholds)
        final["test"] = {
            "default_threshold": _strip_meta(test_default),
            "tuned_threshold": test_tuned,
        }
        logger.info(
            "TEST: tuned_macro_f1=%.4f  argmax_macro_f1=%.4f  prob_gap=%.3f",
            test_tuned["macro_f1_tuned"], test_default["argmax_macro_f1"],
            test_default["prob_gap"],
        )
        confusion = compute_confusion(test_logits, test_labels, thresholds)
        final["test_confusion"] = confusion

    logger.info("[7/7] Klaar. Best epoch=%d, val tuned_macro_f1=%.4f",
                best_epoch, val_tuned["macro_f1_tuned"])

    return {"history": history, "final": final}, final


def _strip_meta(d: dict) -> dict:
    """Verwijder epoch/train_loss zodat we metrics kunnen vergelijken tussen splits."""
    return {k: v for k, v in d.items() if k not in {"epoch", "train_loss"}}


def find_best_thresholds_per_class(logits, labels, grid=None):
    """Zoek per class de threshold die F1 op deze set maximaliseert."""
    import numpy as np
    from sklearn.metrics import f1_score

    if grid is None:
        grid = np.arange(0.05, 0.95, 0.02)

    probs = 1.0 / (1.0 + np.exp(-logits))
    labels_int = labels.astype(int)
    best_thr = np.full(probs.shape[1], 0.5, dtype=float)
    for c in range(probs.shape[1]):
        if labels_int[:, c].sum() == 0:
            continue  # geen positives in val — laat default
        best_f1 = -1.0
        for thr in grid:
            pred = (probs[:, c] >= thr).astype(int)
            f = f1_score(labels_int[:, c], pred, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thr[c] = float(thr)
    return best_thr


def compute_with_thresholds(logits, labels, thresholds) -> dict:
    """Compute multilabel metrics met per-class thresholds."""
    import numpy as np
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    from govmodel.training.labels import LABEL_NAMES

    probs = 1.0 / (1.0 + np.exp(-logits))
    labels_int = labels.astype(int)
    preds = (probs >= thresholds).astype(int)

    macro = float(f1_score(labels_int, preds, average="macro", zero_division=0))
    micro = float(f1_score(labels_int, preds, average="micro", zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(
        labels_int, preds, average=None, zero_division=0,
        labels=list(range(len(LABEL_NAMES))),
    )
    out: dict[str, float] = {
        "macro_f1_tuned": macro,
        "micro_f1_tuned": micro,
    }
    for i, name in enumerate(LABEL_NAMES):
        out[f"f1_{name}_tuned"] = float(f[i])
        out[f"precision_{name}_tuned"] = float(p[i])
        out[f"recall_{name}_tuned"] = float(r[i])
    return out


def compute_confusion(logits, labels, thresholds) -> dict:
    """Verwarringsmatrix op argmax-prediction (single-label assumption)."""
    import numpy as np

    from govmodel.training.labels import LABEL_NAMES

    probs = 1.0 / (1.0 + np.exp(-logits))
    pred_idx = probs.argmax(axis=1)
    true_idx = labels.argmax(axis=1)
    n = len(LABEL_NAMES)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(true_idx, pred_idx, strict=True):
        matrix[t, p] += 1
    return {
        "labels": list(LABEL_NAMES),
        "matrix": matrix.tolist(),
        "row_is_true_col_is_pred": True,
    }
