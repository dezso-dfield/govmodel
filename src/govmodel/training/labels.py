"""Label-encoding voor de multilabel Awb-type-classifier."""

from __future__ import annotations

from govmodel.schemas import AwbLabel

# Vaste volgorde — wordt gepersisteerd in modelcard en config.
LABEL_NAMES: list[AwbLabel] = [
    "aanvraag",
    "bezwaar",
    "beroep",
    "klacht",
    "melding",
    "zienswijze",
    "woo_verzoek",
    "informatieverzoek_3_11",
    "vraag_overig",
]

LABEL_TO_ID: dict[AwbLabel, int] = {label: i for i, label in enumerate(LABEL_NAMES)}
ID_TO_LABEL: dict[int, AwbLabel] = {i: label for label, i in LABEL_TO_ID.items()}


def encode_labels(labels: list[AwbLabel]) -> list[float]:
    """Encode lijst labels naar multi-hot float-vector."""
    vec = [0.0] * len(LABEL_NAMES)
    for label in labels:
        idx = LABEL_TO_ID.get(label)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def decode_labels(probs: list[float], threshold: float = 0.5) -> list[AwbLabel]:
    """Decode multi-hot probability-vector naar lijst labels boven threshold."""
    return [LABEL_NAMES[i] for i, p in enumerate(probs) if p >= threshold and i < len(LABEL_NAMES)]
