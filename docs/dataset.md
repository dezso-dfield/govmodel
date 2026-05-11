# Dataset — govmodel

What the v0.1 classifier was trained on, where it came from, how labels
are decided, and what the next-cycle data looks like. For source-level
provenance see [`data-sources.md`](../data-sources.md); for label
definitions see [`labeling-guidelines.md`](../labeling-guidelines.md).

## Label taxonomy (9 classes)

The taxonomy follows the Dutch `Algemene wet bestuursrecht` (Awb). Each
class is a multilabel sigmoid output — one letter can be both `bezwaar`
and `klacht`.

| Label | Description | Trigger |
|---|---|---|
| `aanvraag` | Verzoek om beschikking | Vergunning, voorziening, subsidie. |
| `bezwaar` | Verzet tegen besluit | "Hierbij teken ik bezwaar aan tegen…" |
| `beroep` | Administratief beroep | Bij een ander bestuursorgaan. |
| `klacht` | Klacht over gedrag | Gedrag bestuursorgaan/medewerker. |
| `melding` | Mededeling zonder besluitvraag | Verhuizing, sloop, leerplicht. |
| `zienswijze` | Reactie op ontwerp-besluit | "Hierbij dien ik mijn zienswijze in." |
| `woo_verzoek` | Openbaarmakingsverzoek | Wet open overheid. |
| `informatieverzoek_3_11` | Inzage stukken | Lopende uitgebreide procedure. |
| `vraag_overig` | Reguliere correspondentie | Geen Awb-procedure herkend. |

Full per-label decision rules are in [`labeling-guidelines.md`](../labeling-guidelines.md).

## Data sources

### Synthetic (the bulk of v0.1)

| Field | Value |
|---|---|
| Generator | Qwen3-4B-Instruct via [`scripts/generate_synthetic.py`](../scripts/generate_synthetic.py) |
| Combinations | 9 labels × 11 writing styles + 64 explicit multilabel mixes |
| Count | ≈ 856 examples in v0.1 |
| Writing styles | Formal · informal · NT2-B1 · legal-dense · chaotic · short · long · emotional · neutral · clipped · over-polite |
| Multilabel mixes | 4 explicit combos trained today: `bezwaar+klacht`, `aanvraag+klacht`, `bezwaar+woo`, `klacht+vraag_overig` |
| Quality control | [`src/govmodel/pii.py`](../src/govmodel/pii.py) redacts BSN / IBAN / phone / email / postcode after generation (Qwen sometimes invents PII) |

**Known gap.** Only 4 multilabel combos are trained explicitly. Other
combinations (e.g. `melding+zienswijze`) are untested and may produce
spurious outputs at inference.

### Open Data Rechtspraak

| Field | Value |
|---|---|
| Source | <https://www.rechtspraak.nl/Uitspraken/paginas/open-data.aspx> |
| Licence | CC0 |
| Puller | [`src/govmodel/pullers/rechtspraak.py`](../src/govmodel/pullers/rechtspraak.py) |
| Use | Real-world eval set + future training signal. |

### WOO catalog (data.overheid.nl)

| Field | Value |
|---|---|
| Source | <https://data.overheid.nl/> |
| Licence | CC0 / public-domain |
| Puller | [`src/govmodel/pullers/woo.py`](../src/govmodel/pullers/woo.py) |
| Use | `woo_verzoek` real-world examples. |

### Open Raadsinformatie

| Field | Value |
|---|---|
| Source | <https://openraadsinformatie.nl/> |
| Puller | [`src/govmodel/pullers/raadsinformatie.py`](../src/govmodel/pullers/raadsinformatie.py) |
| Use | Council-meeting context; not yet in v0.1 training. |

### Burger-citaten

Hand-written realistic letters for adversarial eval (not training). See
[`tests/test_burger_citaten.py`](../tests/test_burger_citaten.py) for
the schema sanity check.

## Splits

| Split | Source | Size (v0.1) | Notes |
|---|---|---:|---|
| Train | synthetic | ~600 | Stratified by label. |
| Val | synthetic | ~90 | Used for early stopping + per-class threshold tuning. |
| Test (synth) | synthetic | ~216 | Same distribution as train — over-rewards memorisation. |
| Eval: handwritten realistic | hand-crafted | 30 | macro-F1 ≈ 0.90. |
| Eval: handwritten adversarial | hand-crafted | 48 | macro-F1 ≈ 0.71. |
| Eval: real Rechtspraak | jurisprudence | 17 | macro-F1 ≈ 0.65; `bezwaar` 0.77, `woo_verzoek` 0.75, `klacht` 0.44. |

Known threshold-tuning concern: per-class thresholds are currently fit
on the same `val` set used for early stopping
([`training/train.py:277`](../src/govmodel/training/train.py)). A
separate threshold split is on the v0.2 roadmap.

## PII handling

[`src/govmodel/pii.py`](../src/govmodel/pii.py) covers:

| Pattern | Validator |
|---|---|
| BSN | 11-proef arithmetic check |
| IBAN | mod-97 check |
| Email | RFC-lite regex |
| Phone (NL) | National & +31 prefix |
| Postcode (NL) | 4 digits + 2 letters |

**Not covered today (regex-only, no NER).** Names, organisation names,
locations. The "no PII in published dataset" claim is therefore
aspirational — see [`MODEL_CARD.md`](../MODEL_CARD.md) for the
documented limitation. spaCy-nl NER integration is on the v0.2 roadmap.

## Synthetic generation in detail

```
scripts/generate_synthetic.py
  ├── LLMClient.from_env()                # localhost:1234 by default
  ├── for label in LABEL_NAMES:
  │      for style in WRITING_STYLES:
  │           for _ in range(EXAMPLES_PER_COMBO):
  │                prompt = render_prompt(label, style)
  │                text = client.chat(...)
  │                row  = {"text": text, "labels": [label], "source": "synthetic_v0.1"}
  │                yield row
  └── pii.redact(row.text) before persistence
```

Configurable via env: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT_S`,
`PARALLEL`, `EXAMPLES_PER_COMBO`, `TEMPERATURE`, `MAX_TOKENS`.

## Schema (`LabeledExample`)

```python
# src/govmodel/schemas.py
class LabeledExample(BaseModel):
    text: str
    labels: list[str]
    source: str
    id: str | None = None
    metadata: dict[str, Any] = {}
```

JSONL is the canonical interchange format:

```jsonl
{"text": "Hierbij teken ik bezwaar...", "labels": ["bezwaar"], "source": "synthetic_v0.1", "id": "synth-bezwaar-formal-0042"}
{"text": "Beste gemeente, op grond van...", "labels": ["woo_verzoek"], "source": "synthetic_v0.1", "id": "synth-woo-formal-0007"}
```

## Active-learning-derived data (next cycle)

The new feedback loop turns operational predictions into training
candidates:

```
logs/predictions.jsonl  ──►  scripts/mine_hard_negatives.py  ──►  queue.jsonl
                                                                       │
                                                                       ▼
                                                       human label → data/feedback/<date>.jsonl
                                                                       │
                                                                       ▼
                                          scripts/train_classifier.py --inputs data/synthetic/v0.1.jsonl data/feedback/*.jsonl
```

Mining surfaces three categories:

- **User disagreement** — feedback rows where the human disagrees with
  the top label. Highest priority.
- **Low confidence** — top score in the 0.20–0.50 band (configurable).
- **Conflicted multilabel** — two top labels within ~0.15 of each other.

See [`docs/architecture.md#active-learning-loop`](architecture.md#active-learning-loop)
for the full mechanism.

## Eval slices the next retrain inherits

| Slice | Path | Used for |
|---|---|---|
| Synthetic test | `data/synthetic/v0.1.test.jsonl` (carved at split time) | Sanity / over-fitting check |
| Handwritten realistic | `tests/test_burger_citaten.py` source | Realistic-tone generalisation |
| Handwritten adversarial | `tests/test_burger_citaten.py` (hard subset) | Edge cases |
| Real Rechtspraak | Pulled via puller | Domain transfer |
| Conformal calibration | Held-out subset of train | `calibration.conformal_thresholds()` |

## Data tooling (`src/govmodel/data/`, `src/govmodel/eval/`)

The `dezso_dev` branch adds a hygiene + augmentation + evaluation stack
the next retrain inherits:

| Tool | Purpose |
|---|---|
| `data.validate_examples` | Schema check against `LabeledExample`. Catches degenerate short texts, empty labels, invalid label names, unknown extra fields. |
| `data.balance_report` | Per-label / per-source / multilabel-arity distribution + imbalance ratio. |
| `data.exact_dedupe` | Whitespace + case-insensitive duplicate collapse. |
| `data.near_dedupe` | Greedy 5-char shingle + Jaccard near-dedup (threshold configurable). |
| `data.leakage_between` | Synthetic↔eval-set near-duplicate detector — the gate that stops `benchmark.py` from being inflated by memorised rows. |
| `data.stratified_split` | Primary-label stratified train/val/test split so rare classes like `informatieverzoek_3_11` get test coverage. |
| `data.standard_augmentations` | 9 deterministic Dutch-flavoured perturbations: typos at 3% / 8%, casing, ALL-CAPS emphasis spans (gov-specific), punctuation strip/double, NT2/B1 simplification, dialect / anglicism / Turkish-Arabic-loan swaps. |
| `data.formal_to_informal` | Optional LLM-driven rewrite (lazy-loads `LLMClient`). Used for training augmentation, never for robustness eval. |
| `data.meta_klacht.iter_prompts` | 8 scenarios × 5 styles prompt templates for the v0.2 meta-klacht synthetic batch (R&D item G2 from plan.md). |
| `eval.evaluate_slice` | Per-tag F1 + optimal threshold + Δ-vs-default + macro/micro F1 + ECE + reliability curve. |
| `eval.regression_gate` | CI gate: macro-F1 must not drop > `max_drop` on any shared slice. |

Driver scripts:

```
python scripts/audit_dataset.py \
    --inputs data/synthetic/v0.1.jsonl \
    --eval-against data/eval/handwritten_realistic.jsonl \
                   data/eval/rechtspraak.jsonl \
    --out reports/audit-v0.1.json
# exits 1 if leakage or blocking issues found

python scripts/build_robustness_pack.py \
    --in data/eval/handwritten_realistic.jsonl \
    --out data/eval/robustness_pack.jsonl

python scripts/evaluate_slices.py \
    --checkpoint models/awb-classifier-v0.1 \
    --slice handwritten_realistic=data/eval/handwritten_realistic.jsonl \
    --slice handwritten_adversarial=data/eval/handwritten_adversarial.jsonl \
    --slice rechtspraak=data/eval/rechtspraak.jsonl \
    --slice robustness=data/eval/robustness_pack.jsonl \
    --out reports/eval-2026-05-11 \
    --gate-against reports/baseline/report.json \
    --max-drop 0.05
```

## Roadmap

- Meta-klacht hand-crafted set (50–100 examples) for `klacht` F1.
- Hard-negative mining from the feedback loop (mechanism ready, awaits production traffic).
- Real Rechtspraak corpus extension from 1 month → 6 months.
- spaCy-nl NER for name / org / location redaction.
- Multilabel coverage beyond the 4 explicit mixes.
- NT2 / Turkish / Arabic / Frisian extension for equity.
