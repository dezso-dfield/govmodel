# Model Card — govmodel/awb-classifier-v0.1

## Model details

- **Naam**: govmodel-awb-classifier-v0.1
- **Type**: Multilabel sequence classifier (Transformer encoder + sigmoid head)
- **Base model**: [DTAI-KULeuven/robbert-2023-dutch-base](https://huggingface.co/DTAI-KULeuven/robbert-2023-dutch-base)
- **Parameters**: 124M
- **Talen**: Nederlands (primair); Engels en NT2-Nederlands matig ondersteund
- **Licentie**: EUPL-1.2
- **Versie**: 0.1 (release-baseline)
- **Publicatie**: Noaber AI — [info@noaberai.nl](mailto:info@noaberai.nl)
- **Trainingsdatum**: mei 2026

## Beoogd gebruik

### Primary use case

Classificatie van inkomende digitale gemeentepost (e-mail, webformulier, gescande brief) volgens **Algemene wet bestuursrecht-typering**, om routering en termijnbewaking bij Nederlandse gemeenten te ondersteunen.

Output: één of meer labels uit:
`aanvraag` · `bezwaar` · `beroep` · `klacht` · `melding` · `zienswijze` · `woo_verzoek` · `informatieverzoek_3_11` · `vraag_overig`

### Out-of-scope

- **Automatische besluitvorming** — strikt verboden voor dit gebruik. AI Act vereist mens-in-the-loop voor besluiten over individuen.
- **Inhoudelijke beoordeling van aanvragen** — model bepaalt het *type* document, niet of een aanvraag toegewezen moet worden.
- **Risico- of fraudedetectie** — niet getraind voor, en politiek/juridisch ongewenst (toeslagenaffaire-context).
- **Niet-Nederlandse teksten** — primair NL; Engels werkt soms maar onbetrouwbaar.

### Beoogde gebruikers

- KCC- en post-medewerkers bij gemeenten (advies-component)
- Behandelaars in zaaksystemen (initiële routering)
- Researchers en leveranciers die op deze baseline doorbouwen

## Trainingsdata

### Bronnen

- **~860 synthetic burger-aan-gemeente brieven**, gegenereerd door Qwen3-4B-Instruct lokaal, verspreid over:
  - 9 labels × 11 schrijfstijlen × ~8 voorbeelden = 792 single-label
  - 4 multilabel-combinaties (bezwaar+klacht, aanvraag+klacht, etc.) × 4 styles × 4 = 64 multilabel
- **Geen casuïstiek uit gemeenten** in v0.1 — alleen publieke bronnen + LLM-augmentatie

### Stijl-coverage

`formal_short`, `formal_long`, `informal_short`, `informal_medium`, `angry_medium`, `polite_questioning`, `nt2_simple`, `elderly_handwritten_style`, `legal_dense`, `rambling_long`, `chaotic_typing`

### Pseudonimisering

Alle voorbeelden door regex-laag met:
- BSN met 11-proef validatie
- IBAN met mod-97 validatie
- E-mail / telefoon (alleen werkelijke NL-kengetallen) / postcode

Op de v0.2 dataset werden 182 PII-redacties uitgevoerd in 100 records (Qwen3 verzon emails/postcodes/telefoons in zijn synthetic outputs).

## Trainingsprocedure

| Hyperparameter | Waarde |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-5 |
| Schedule | Linear warmup, 10% warmup ratio |
| Batch size | 8 (effectief 8) |
| Epochs | 10 (best model: epoch 4) |
| Loss | BCEWithLogitsLoss met `pos_weight=8` (1/9-prior balancering) |
| Gradient clipping | norm 1.0 |
| Mixed precision | bf16 (Ampere+) |
| Weight decay | 0.01 |
| Max sequence length | 512 |
| Hardware | NVIDIA RTX 3050 Laptop, 4 GB VRAM |
| Trainingstijd | ~5 minuten |

Best model selectie: `argmax_macro_f1` op stratified val split. Per-class threshold-tuning op val voor multilabel-decoding.

## Evaluatieresultaten

### Synthetic test set (216 voorbeelden, gestratificeerd uit zelfde distributie als training)

| Metric | Waarde |
|---|---|
| argmax macro-F1 | 0.95 |
| argmax micro-F1 | 0.95 |
| prob_gap (true vs false) | 0.96 |

### Handgeschreven realistische set (n=30, "alledaagse stroom")

argmax macro-F1: **0.903**

Per label:

| Label | F1 | n |
|---|---|---|
| aanvraag | 0.89 | 4 |
| bezwaar | 1.00 | 4 |
| beroep | 1.00 | 1 |
| klacht | 0.89 | 4 |
| melding | 1.00 | 4 |
| zienswijze | 1.00 | 3 |
| woo_verzoek | 0.80 | 3 |
| informatieverzoek_3_11 | 0.80 | 2 |
| vraag_overig | 0.75 | 5 |

### Adversarial set (n=48, hard cases: NT2, juridisch, multilabel, kort/emotioneel)

argmax macro-F1: **0.715**

Per label:

| Label | F1 | n |
|---|---|---|
| aanvraag | 0.43 | 5 |
| bezwaar | 0.70 | 12 |
| beroep | 0.80 | 2 |
| klacht | 0.50 | 10 |
| melding | 0.75 | 4 |
| zienswijze | 1.00 | 4 |
| woo_verzoek | 0.86 | 3 |
| informatieverzoek_3_11 | 1.00 | 1 |
| vraag_overig | 0.40 | 14 |

### Echte burgerteksten uit Nederlandse jurisprudentie (n=17, Open Data Rechtspraak CC0)

| Label | F1 | n | Toelichting |
|---|---|---|---|
| **bezwaar** | **0.77** | 6 | Bezwaarschriften over heffingen, WOZ, schadefonds — robuust |
| **woo_verzoek** | **0.75** | 5 | Uitvoerige Woo/Wob-verzoeken — robuust |
| klacht | 0.44 | 6 | Reguliere klachten oké; juridisch-dichte en meta-klachten zwakker |
| 6 overige labels | n.v.t. | 0 | Niet aanwezig in jurisprudentie-set; alleen synthetic-getoetst |

## Beperkingen en bekende issues

### Synthetic→real domain gap

Trainingsdata is door een LLM (Qwen3-4B) gegenereerd. Echte burgertaal verschilt op subtiele manieren:
- **Echte klachten zijn vaker meta-klachten** ("klacht over hoe mijn bezwaar is afgehandeld") — synthetic varianten zijn directer
- **Echte taal heeft meer juridisch jargon** ("Awb 9:7", "klokkenluiderregeling", "zorgvuldigheidsbeginsel") — synthetic minder
- **Schade-formuleringen** in echte klachten ("causaal verband", "trauma") lijken structureel op bezwaar-taal

**Praktisch effect**: klacht-detectie zakt van 0.89 (synthetic-realistisch) naar 0.44 (echt). Bezwaar en WOO houden grotendeels stand.

### Onvolledige real-world coverage

6 van 9 labels zijn **niet vertegenwoordigd** in onze real-world eval-set (rechtspraak gaat vooral over geschillen). Voor `aanvraag`, `beroep`, `melding`, `zienswijze`, `informatieverzoek_3_11`, `vraag_overig` hebben we **alleen synthetic en handgeschreven evaluatie** — geen echte data van burgers.

### vraag_overig is een restcategorie

Lekt structureel naar `klacht` (vraag wordt klacht door geïrriteerde toon) en `aanvraag` (vraag over een aanvraag wordt aanvraag). Subtype-splitting in v0.2 voorzien.

### Beperkte taalcoverage

- Engelse tekst: matig (1 voorbeeld in adversarial-set, model herkende het wel als aanvraag, maar onbetrouwbaar)
- NT2-Nederlands: gemengd; eenvoudige NT2 werkt, sterk gebroken NT2 minder
- Meertalige post (Turks, Arabisch, Pools, etc. die in NL gemeenten voorkomen): niet getraind

### Multilabel coverage is asymmetrisch

Trainingsdata bevat 4 specifieke multilabel-combinaties (bezwaar+klacht, aanvraag+klacht, aanvraag+vraag, zienswijze+vraag). Andere combinaties (bv. melding+zienswijze) zijn niet expliciet getraind en kunnen onzekere outputs geven.

### Threshold-afhankelijkheid

Default threshold 0.5 op sigmoid werkt redelijk, maar **per-class tuning** (zie `final_metrics.json` in elk model) verbetert macro-F1 met +0.05 tot +0.20. Aanbeveling: gebruik altijd de getunede thresholds.

## Bias-overwegingen

### Wat we hebben gemeten

- Synthetic data is via één LLM (Qwen3-4B) gegenereerd → mogelijke clichés, beperkt stijlbereik
- Adversarial-set is door één auteur geschreven → zelfde-blik-bias
- Real-world eval is uit Nederlandse jurisprudentie van **één maand** (jan 2024) → temporele bias

### Wat we NIET hebben gemeten

- Performance per regio / dialect
- Performance per sociaal-demografische groep
- Performance op meer-dan-NT2-niveau-A2-Nederlands
- Performance op brieven van/over kwetsbare groepen (jeugdzorg, schuldhulpverlening, GGZ-context)

Adopters wordt aangeraden bias-evaluatie op hun eigen data uit te voeren voor productie-inzet.

## AI Act-positie

**Risiconiveau**: *Limited risk* (EU AI Act 2024/1689).

Motivering:
- Het model **classificeert document-type**, neemt geen besluit over een individu
- Geen Annex III-categorie (geen toegang tot essentiële diensten, geen kredietbeoordeling, geen wetshandhaving)
- Output is een routerings-advies; mens beslist over inhoudelijke afhandeling

Vereisten waar dit model aan voldoet:
- ✅ Transparantie over AI-gebruik (deze model card)
- ✅ Eerlijke beperkingen-documentatie
- ✅ Trainingsdata-herkomst gedocumenteerd ([DATASHEET.md](DATASHEET.md))

**Niet automatisch geschikt voor high-risk-gebruik** zoals besluitvorming in sociaal domein zonder substantieel aanvullend werk (mens-in-the-loop, audit-logging, bezwaar-route).

## Gebruik

```python
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="NoaberAI/govmodel-awb-classifier-v0.1",
    top_k=None,
)

# Multilabel: alle scores boven threshold
result = clf("Hierbij teken ik bezwaar aan tegen uw besluit van 12 maart...")
# → [{'label': 'bezwaar', 'score': 0.91}, ...]

# Productie: gebruik per-class thresholds uit final_metrics.json
```

## Citatie

```
@misc{govmodel2026,
  title  = {govmodel: Open-source Awb-typering classifier voor Nederlandse gemeentepost},
  author = {Noaber AI},
  year   = {2026},
  url    = {https://huggingface.co/NoaberAI/govmodel-awb-classifier-v0.1},
  note   = {EUPL-1.2}
}
```

## Contact

[info@noaberai.nl](mailto:info@noaberai.nl)
