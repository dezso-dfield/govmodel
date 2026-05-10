"""Single source of truth for the Awb classifier's torch Dataset + collate.

Used by both `training/train.py` and `scripts/benchmark.py` (previously
duplicated). Pure-PyTorch — no `datasets` / pyarrow — so this module is
safe to import on Windows and inside the FastAPI service.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.data import Dataset as TorchDataset

from govmodel.training.labels import encode_labels


class AwbDataset(TorchDataset):
    """Multilabel Awb classifier dataset.

    `examples` is any iterable of objects with `.text` (str) and `.labels`
    (Sequence[str]) attributes — typically `LabeledExample`.
    """

    def __init__(self, examples: Iterable, tokenizer, max_length: int) -> None:
        examples = list(examples)
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
    """Pad-and-stack collate. Returns a callable suitable for DataLoader."""

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.tensor(
            [item["input_ids"] + [pad_token_id] * (max_len - len(item["input_ids"]))
             for item in batch],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [item["attention_mask"] + [0] * (max_len - len(item["attention_mask"]))
             for item in batch],
            dtype=torch.long,
        )
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate
