"""Inference core — shared by Gradio demo, FastAPI service, and the dashboard.

Everything that takes raw text and returns a Prediction lives here. Keep
this module side-effect free apart from model load; logging happens at the
caller.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from govmodel.errors import ModelLoadError

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.environ.get("GOVMODEL_MODEL_ID",
                                  "NoaberAI/govmodel-awb-classifier-v0.1")
DEFAULT_THRESHOLD = float(os.environ.get("GOVMODEL_DEFAULT_THRESHOLD", "0.5"))
ABSTAIN_THRESHOLD = float(os.environ.get("GOVMODEL_ABSTAIN_THRESHOLD", "0.20"))


@dataclass
class Prediction:
    """One classification result."""

    request_id: str
    text_chars: int                          # raw length, NOT the text itself (privacy)
    text_hash: str                           # short content hash for de-dup analytics
    top_label: str
    top_score: float
    threshold: float
    above_threshold: list[tuple[str, float]] # sorted desc by score
    all_scores: dict[str, float]
    abstain: bool                            # True iff top_score < ABSTAIN_THRESHOLD
    abstain_reason: str | None
    latency_ms: float
    model_id: str
    ts: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def to_public_dict(self) -> dict[str, Any]:
        """Caller-visible dict — mirrors `asdict` but tuples → lists for JSON."""
        d = asdict(self)
        d["above_threshold"] = [list(t) for t in self.above_threshold]
        return d


class Classifier:
    """Wraps a HuggingFace AutoModelForSequenceClassification."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ModelLoadError(
                "torch/transformers not installed — run `pip install -e .[ml]`"
            ) from e

        self.model_id = model_id
        self._torch = torch
        logger.info("loading model", extra={"model_id": model_id})
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self.model.to(device).eval()

        id2label = self.model.config.id2label
        self.labels_in_order = [id2label[i] for i in range(len(id2label))]
        logger.info("model loaded",
                    extra={"device": device, "n_labels": len(self.labels_in_order)})

    def classify(
        self,
        text: str,
        threshold: float = DEFAULT_THRESHOLD,
        request_id: str | None = None,
    ) -> Prediction:
        from hashlib import sha1

        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        t0 = time.perf_counter()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits[0].float().cpu()
        probs = self._torch.sigmoid(logits).numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        all_scores = {self.labels_in_order[i]: float(probs[i]) for i in range(len(probs))}
        top_idx = int(probs.argmax())
        top_label = self.labels_in_order[top_idx]
        top_score = float(probs[top_idx])

        above = sorted(
            ((lbl, s) for lbl, s in all_scores.items() if s >= threshold),
            key=lambda x: -x[1],
        )

        abstain = top_score < ABSTAIN_THRESHOLD
        abstain_reason = None
        if abstain:
            abstain_reason = (
                f"top score {top_score:.2f} below abstain threshold "
                f"{ABSTAIN_THRESHOLD:.2f} — route to human"
            )

        return Prediction(
            request_id=request_id or str(uuid.uuid4())[:8],
            text_chars=len(text),
            text_hash=sha1(text.encode("utf-8")).hexdigest()[:12],
            top_label=top_label,
            top_score=top_score,
            threshold=threshold,
            above_threshold=above,
            all_scores=all_scores,
            abstain=abstain,
            abstain_reason=abstain_reason,
            latency_ms=latency_ms,
            model_id=self.model_id,
        )


_PREDICTION_LOG_PATH = os.environ.get(
    "GOVMODEL_PREDICTION_LOG", "logs/predictions.jsonl"
)


class PredictionLogger:
    """Append-only JSONL logger for predictions.

    Privacy: by default we log only `text_chars` and a short `text_hash`,
    NOT the raw text. Set GOVMODEL_LOG_RAW_TEXT=1 to also log the text
    (e.g. when running on already-pseudonymised data for an active-learning
    feedback loop).
    """

    def __init__(self, path: str | Path = _PREDICTION_LOG_PATH,
                 *, log_raw_text: bool | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if log_raw_text is None:
            log_raw_text = os.environ.get("GOVMODEL_LOG_RAW_TEXT", "0") == "1"
        self.log_raw_text = log_raw_text

    def log(self, prediction: Prediction, *, raw_text: str | None = None,
            user_label: str | None = None) -> None:
        record: dict[str, Any] = asdict(prediction)
        if self.log_raw_text and raw_text is not None:
            record["text"] = raw_text
        if user_label:
            record["user_label"] = user_label
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
