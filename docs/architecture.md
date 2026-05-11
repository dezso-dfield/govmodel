# Architecture — govmodel

govmodel is an open-source Dutch text classifier that routes inbound
municipal correspondence into 9 `Awb` (Algemene wet bestuursrecht)
categories. The artifact you see in HuggingFace is `RobBERT-2023-base`
fine-tuned with a multilabel sigmoid head; the artifact you see in this
repo is the surrounding **operating system**: training, evaluation,
inference, calibration, analytics, and the human-in-the-loop feedback
mechanism that lets the next retrain do strictly better than the last.

## High-level diagram

```
              ┌─────────────────────────────────────────────────────────────┐
              │                       INGESTION                            │
              │  pullers/* : Rechtspraak · WOO · Open Raadsinformatie       │
              │  synthetic.py : Qwen3-4B generates 9 labels × 11 styles     │
              └─────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────────────────────────┐
              │                       PII REDACTION                         │
              │  pii.py : BSN (11-proef), IBAN (mod-97), email, phone,      │
              │           postcode, address regex                           │
              └─────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
   ┌────────────┐   training/      ┌───────────────────────────────┐  artifacts
   │ JSONL  ── ─┼─► dataset.py ──► │   training/train.py            ├──►  HF Hub
   │ (synthetic │   awb_dataset.py │   - AwbDataset + collate       │     +
   │  + gold)   │   seeding.py     │   - BCEWithLogitsLoss          │     models/
   └────────────┘                   │     pos_weight=8.0             │     awb-v0.1/
                                    │   - bf16 mixed precision       │
                                    │   - per-class threshold tuning │
                                    └──────────────┬─────────────────┘
                                                   │
                                                   ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                          INFERENCE CORE                                 │
   │  src/govmodel/inference.py                                              │
   │  ┌──────────────┐   ┌─────────────────┐   ┌─────────────────────────┐  │
   │  │ Classifier   │──►│ Prediction      │──►│ PredictionLogger        │  │
   │  │   - HF model │   │   - text_hash   │   │   - JSONL append-only    │  │
   │  │   - sigmoid  │   │   - all_scores  │   │   - GOVMODEL_LOG_RAW_TEXT│  │
   │  │   - 512 tok  │   │   - abstain     │   │     opt-in for raw text │  │
   │  └──────────────┘   │   - latency_ms  │   └─────────────────────────┘  │
   │                     └─────────────────┘                                  │
   └──────────────┬──────────────────────┬───────────────────┬───────────────┘
                  │                      │                   │
                  ▼                      ▼                   ▼
   ┌────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
   │   Gradio app.py     │  │   FastAPI            │  │   Dashboard HTML     │
   │   port 7860         │  │   src/govmodel/api.py│  │   served at /        │
   │   - human triage UI │  │   - /api/classify    │  │   - label distrib    │
   │   - examples        │  │   - /api/feedback    │  │   - score buckets    │
   │                     │  │   - /api/predictions │  │   - p50/p95 latency  │
   │                     │  │   - /api/analytics/* │  │   - ECE + reliability│
   │                     │  │   - Bearer auth      │  │   - top confusions   │
   └────────────────────┘  └──────────┬───────────┘  └──────────┬───────────┘
                                       │                          │
                                       └──────────┬───────────────┘
                                                  │
                                                  ▼
                          ┌──────────────────────────────────────┐
                          │   logs/predictions.jsonl              │
                          │   - request_id, top_label, top_score  │
                          │   - above_threshold, abstain          │
                          │   - latency_ms, model_id, ts          │
                          │   - text_hash (always)                │
                          │   - text (opt-in only)                │
                          │   - user_label (when /feedback fires) │
                          └──────────────────┬───────────────────┘
                                              │
                                              ▼
                          ┌──────────────────────────────────────┐
                          │   mining.py + calibration.py          │
                          │   - low-confidence / conflict / user  │
                          │   - ECE + reliability curve           │
                          │   - conformal thresholds              │
                          └──────────────────┬───────────────────┘
                                              │
                                              ▼
                          ┌──────────────────────────────────────┐
                          │   scripts/mine_hard_negatives.py      │
                          │   ──► JSONL labelling queue           │
                          │   ──► next training cycle             │
                          └──────────────────────────────────────┘
```

## Model

| Field | Value |
|---|---|
| Backbone | `DTAI-KULeuven/robbert-2023-dutch-base` (124M params) |
| Head | `nn.Linear(hidden, 9)` (multilabel sigmoid) |
| Loss | `BCEWithLogitsLoss(pos_weight=8.0)` — balances the ≈1/9 label prior |
| Optimiser | AdamW, lr 3e-5, weight decay 0.01, linear warmup |
| Precision | bf16 mixed (auto-detected on Ampere+) |
| Max length | 512 tokens; silently truncates today — long-document handling is on the v0.3 roadmap |
| Threshold | Per-class, fit on validation; saved alongside the checkpoint |

The classifier itself is intentionally small. The leverage lives in
*data quality* (PII-clean synthetic + jurisprudence), *threshold tuning*
(per-class, not flat 0.5), and the surrounding operability layer.

## Why a custom PyTorch loop (not `Trainer`)

`transformers.Trainer` triggers a `datasets → pyarrow → aiohttp → ssl`
import chain that crashes on Windows. Municipal IT shops run Windows; we
write the training loop directly so the same recipe works on every
target environment. See [`src/govmodel/training/train.py`](../src/govmodel/training/train.py).

## Inference path

Both the Gradio demo and the FastAPI service flow through one
[`Classifier`](../src/govmodel/inference.py). A `Prediction` carries:

- `top_label` + `top_score`
- `all_scores` — full sigmoid vector
- `above_threshold` — sorted multilabel hits
- `abstain` + `abstain_reason` (top score below `GOVMODEL_ABSTAIN_THRESHOLD`)
- `latency_ms`, `request_id`, `text_chars`, `text_hash`
- (Optional) `text` only when `GOVMODEL_LOG_RAW_TEXT=1` — privacy-by-default.

## Abstain mechanism

Below `GOVMODEL_ABSTAIN_THRESHOLD` (default 0.20) the model surfaces
*"insufficient confidence — route to human"* instead of guessing. This
exists because municipal users need calibrated uncertainty, not raw
sigmoids: a 0.15 top-score is unreliable enough that an unconfident
suggestion costs more than no suggestion at all.

## Calibration

[`src/govmodel/calibration.py`](../src/govmodel/calibration.py) gives:

- **Temperature scaling** — LBFGS-fit scalar `T` minimising BCE on a
  held-out set. Apply with `calibrated = logits / T`.
- **Expected Calibration Error (ECE)** — bin-wise reliability summary.
- **Reliability curves** — per-bin confidence vs. accuracy, fed into the
  dashboard.
- **Mondrian per-class conformal split** — per-class thresholds yielding
  a target recall on calibration data.
- **`ece_from_log()`** — operator-side: derives live ECE from
  `(top_score, top_label == user_label)` pairs in the prediction log,
  so the dashboard shows calibration quality as feedback arrives.

## Active-learning loop

[`src/govmodel/mining.py`](../src/govmodel/mining.py) reads the
prediction log and ranks candidates by labelling-value:

1. **`user_disagreement`** (highest priority) — feedback rows where the
   user's label differs from the model's top label.
2. **`low_confidence`** — top scores inside an active learning window
   (default 0.20–0.50).
3. **`conflicted_multilabel`** — two top labels within `conflict_margin`
   of each other, both above `multilabel_min_above`.

[`scripts/mine_hard_negatives.py`](../scripts/mine_hard_negatives.py)
emits a ready-to-label JSONL with `_mine_reason` + `_mine_score` per
row. The result feeds the next round of synthetic generation /
hand-labelling.

## Privacy posture

- PII redaction at ingest time ([`pii.py`](../src/govmodel/pii.py)).
- Prediction log stores `text_hash` (SHA-1 first 12) by default; raw
  text only when `GOVMODEL_LOG_RAW_TEXT=1`.
- FastAPI Bearer auth via `GOVMODEL_API_TOKEN`; CORS allowlist via
  `GOVMODEL_CORS_ORIGINS` — wildcard is **not** the default.
- All operator-visible PII paths are scoped to opt-in.

## Deployment surface

| Target | Entry point | Notes |
|---|---|---|
| HuggingFace Spaces | [`app.py`](../app.py) | Gradio auto-detected; pin `requirements.txt`. |
| Self-hosted | `uvicorn govmodel.api:app` | FastAPI + analytics dashboard; runs on CPU at ~30 ms/letter. |
| On-prem zaaksysteem | `/api/classify` via Bearer | Suite4Sociaal / Civision / Decos JOIN integrations. ONNX export is on the v0.3 roadmap. |

## Roadmap items not yet implemented

See [`../plan.md`](../../plan.md) for the full map. Highlights:

- Hierarchical (super-type → fine-type) head to disambiguate `klacht` ↔ `bezwaar`.
- Long-document encoder (Longformer-NL or chunk-and-pool RobBERT).
- Evidence-span extraction head (token-level BIO over guideline highlights).
- Domain-adaptive MLM pretraining on Rechtspraak/Kamerstukken/WOO.
- LLM-RAG fallback for low-confidence inputs.
- Distillation + ONNX-INT8 for on-prem CPU deployment.
- Multilingual extension for NT2/Turkish/Arabic/Frisian.
