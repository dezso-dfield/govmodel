# Repository structure — govmodel

How the directories fit together, what each module is for, and where to
start reading depending on what you need to do.

## Top-level layout

```
govmodel/
├── README.md, MODEL_CARD.md, DATASHEET.md     # public docs
├── CHANGELOG.md
├── LICENSE                                    # EUPL-1.2
├── pyproject.toml                             # package + extras
├── app.py                                     # Gradio entry point (HF Spaces)
├── data-sources.md                            # source-by-source provenance
├── labeling-guidelines.md                     # how each Awb label is decided
├── docs/                                      # ← you are here
│   ├── architecture.md                        # how it all fits together
│   ├── structure.md                           # this file
│   └── dataset.md                             # data + labels
├── src/govmodel/                              # importable package
│   ├── __init__.py
│   ├── schemas.py                             # Pydantic LabeledExample
│   ├── pii.py                                 # BSN/IBAN/email/phone redaction
│   ├── synthetic.py                           # Qwen-driven generation
│   ├── llm_client.py                          # OpenAI-compatible LLM client
│   ├── errors.py                              # typed exceptions
│   ├── seeding.py                             # seed_everything()
│   ├── logging_setup.py                       # JSON-line stdlib logging
│   ├── inference.py                           # shared Classifier + Prediction
│   ├── api.py                                 # FastAPI service + endpoints
│   ├── analytics.py                           # aggregations over the log
│   ├── calibration.py                         # ECE, temperature scaling, conformal
│   ├── mining.py                              # hard-negative miner
│   ├── dashboard/index.html                   # single-file analytics UI
│   ├── pullers/                               # data source clients
│   │   ├── rechtspraak.py
│   │   ├── woo.py
│   │   ├── raadsinformatie.py
│   │   └── burger_citaten.py
│   └── training/
│       ├── labels.py                          # LABEL_NAMES + encode/decode
│       ├── dataset.py                         # JSONL loader + splitters
│       ├── awb_dataset.py                     # PyTorch Dataset + collate
│       └── train.py                           # custom training loop
├── scripts/                                   # CLI entry points
│   ├── train_classifier.py                    # train end-to-end
│   ├── benchmark.py                           # eval against held-out sets
│   ├── mine_hard_negatives.py                 # build labelling queue from log
│   ├── generate_synthetic.py                  # call Qwen via LLMClient
│   ├── pull_rechtspraak.py / pull_woo.py / ...# data ingestion
│   └── pseudonymize_*.py                      # PII pass on raw pulls
├── tests/                                     # 97 passing tests
│   ├── test_pii.py
│   ├── test_synthetic.py
│   ├── test_training_labels.py
│   ├── test_inference_helpers.py
│   ├── test_analytics.py
│   ├── test_api_smoke.py                      # FastAPI routing + auth
│   ├── test_seeding_and_logging.py
│   ├── test_llm_client.py
│   ├── test_mining.py
│   └── test_calibration.py
└── .github/workflows/ci.yml                   # pytest + ruff on push
```

## Reading order by task

### "I want to use the trained model"

1. [`README.md`](../README.md) for the quick-start.
2. [`app.py`](../app.py) — Gradio demo using the shared core.
3. [`src/govmodel/inference.py`](../src/govmodel/inference.py) — the
   `Classifier` you'd embed in your own code.

### "I want to call the model as an HTTP service"

1. [`src/govmodel/api.py`](../src/govmodel/api.py) — every endpoint is documented inline.
2. Run: `uvicorn govmodel.api:app --port 8001`.
3. Hit `GET /` for the dashboard, `GET /api/health`, `POST /api/classify`.

### "I want to retrain"

1. [`docs/dataset.md`](dataset.md) for the data inventory.
2. [`scripts/train_classifier.py`](../scripts/train_classifier.py) — CLI driver.
3. [`src/govmodel/training/train.py`](../src/govmodel/training/train.py) — the actual loop.
4. [`src/govmodel/training/awb_dataset.py`](../src/govmodel/training/awb_dataset.py) — shared between training and benchmark.

### "I want to improve the model with feedback data"

1. Run the FastAPI service with the `/feedback` endpoint exposed.
2. As corrections come in they're appended to `logs/predictions.jsonl`.
3. `python scripts/mine_hard_negatives.py --log logs/predictions.jsonl --out data/labeling/queue.jsonl`
4. Label the queue (any tool that reads JSONL), append to training data, retrain.

### "I want to add a new data source"

1. Read [`data-sources.md`](../data-sources.md) for the existing pattern.
2. Add a puller under [`src/govmodel/pullers/`](../src/govmodel/pullers/).
3. Add a `scripts/pull_<source>.py` driver.
4. Pseudonymise with [`src/govmodel/pii.py`](../src/govmodel/pii.py) before persistence.
5. Update [`docs/dataset.md`](dataset.md).

## Request flow — `POST /api/classify`

```
HTTP POST /api/classify
   │ {"text": "...", "threshold": 0.5}
   ▼
api.py :: classify()
   │  - Optional Bearer-token check (require_token)
   │  - Validate via Pydantic ClassifyRequest
   ▼
inference.Classifier.classify(text, threshold)
   │  - tokenizer(text, truncation=True, max_length=512)
   │  - model.forward() → logits
   │  - torch.sigmoid → probs
   │  - top-1 + multilabel filter + abstain check
   ▼
Prediction (dataclass)
   │  - request_id, text_hash, top_label, top_score, all_scores,
   │    above_threshold, abstain, latency_ms, ts
   ▼
PredictionLogger.log(prediction, raw_text=text)
   │  - JSONL append (text-hash only by default)
   ▼
return JSON {prediction.to_public_dict()}
```

## Request flow — feedback loop

```
HTTP POST /api/classify/{request_id}/feedback
   │ {"user_label": "klacht"}
   ▼
api.py :: feedback()
   │  - Optional Bearer-token check
   ▼
Append feedback row to logs/predictions.jsonl
   │  {"request_id": "...", "user_label": "klacht", "feedback_only": true}
   ▼
mining.py :: _resolve_user_labels()
   │  - Resolves latest user_label per request_id
   ▼
mining.py :: mine()
   │  - Ranks: user_disagreement > low_confidence > conflicted_multilabel
   ▼
to_label_queue(candidates, out_path)
   │  - Writes ranked JSONL with _mine_reason + _mine_score
   ▼
[Human labelling tool reads queue.jsonl]
   ▼
Append corrected labels to data/synthetic/<next_version>.jsonl
   ▼
scripts/train_classifier.py → new checkpoint
```

## Configuration surface

Almost everything is configurable via env vars so deployments don't need
to edit code:

| Variable | Default | Purpose |
|---|---|---|
| `GOVMODEL_MODEL_ID` | `NoaberAI/govmodel-awb-classifier-v0.1` | HF model id or local path. |
| `GOVMODEL_DEFAULT_THRESHOLD` | `0.5` | Multilabel cutoff. |
| `GOVMODEL_ABSTAIN_THRESHOLD` | `0.20` | Below this, route to human. |
| `GOVMODEL_PREDICTION_LOG` | `logs/predictions.jsonl` | JSONL log path. |
| `GOVMODEL_LOG_RAW_TEXT` | `0` | Set `1` to log raw text (privacy: opt-in). |
| `GOVMODEL_API_TOKEN` | (unset) | Optional Bearer auth on POST endpoints. |
| `GOVMODEL_CORS_ORIGINS` | (empty = same-origin) | Comma-separated allowlist. |
| `GOVMODEL_LOG_LEVEL` | `INFO` | Standard `logging` levels. |
| `GOVMODEL_LOG_JSON` | `1` | Set `0` for human-readable lines. |
| `LLM_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible LLM for synthetic gen. |
| `LLM_MODEL` | `qwen/qwen3-4b-2507` | Model name on the local server. |
| `LLM_TIMEOUT_S` | `180` | Per-request timeout. |
| `LLM_API_KEY` | (unset) | Bearer for the LLM server. |

## Tests

CI ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs:

- `pytest` with the four data-puller tests excluded (they need network).
- `ruff check` on the dezso_dev-introduced modules.

Local quick sweep: `pytest -q --ignore=tests/test_burger_citaten.py
--ignore=tests/test_raadsinformatie.py --ignore=tests/test_rechtspraak.py
--ignore=tests/test_woo.py`.

## Branch + release flow

- `main` — public, tracks the v0.1 checkpoint on HF Hub.
- `dezso_dev` — active development; everything described here lives here.
- New checkpoints push to HF Hub before tagging the corresponding commit.
