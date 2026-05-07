# Changelog

Alle noemenswaardige wijzigingen aan dit project worden hier bijgehouden.

Format gebaseerd op [Keep a Changelog](https://keepachangelog.com/), versie-schema [Semantic Versioning](https://semver.org/).

## [v0.1.0] — 2026-05-07

Eerste publieke release.

### Added
- RobBERT-2023-base classifier voor Awb-typering van inkomende digitale gemeentepost
- 9 labels: `aanvraag`, `bezwaar`, `beroep`, `klacht`, `melding`, `zienswijze`, `woo_verzoek`, `informatieverzoek_3_11`, `vraag_overig`
- Multilabel-ondersteuning incl. expliciete combinaties (bezwaar+klacht etc.)
- ~860 synthetic trainingsvoorbeelden gegenereerd via Qwen3-4B over 11 schrijfstijlen
- Pseudonimiseringspipeline (BSN 11-proef, IBAN mod-97, e-mail/telefoon/postcode)
- Pure-PyTorch trainingsloop met `pos_weight=8` BCE, per-class threshold-tuning, gradient clipping, bf16
- Pullers voor Open Data Rechtspraak, data.overheid.nl WOO-catalog, Open Raadsinformatie
- Burger-citaten extractor uit jurisprudentie
- Drie evaluatie-sets:
  - `gold_realistic.jsonl` (n=30) — handgeschreven alledaagse voorbeelden
  - `adversarial.jsonl` (n=48) — handgeschreven hard cases
  - `gold_real.jsonl` (n=17) — echte burger-citaten uit jurisprudentie (CC0 Open Data Rechtspraak)
- Benchmark- en compare-scripts
- README, MODEL_CARD, DATASHEET (Gebru-style)
- EUPL-1.2 licentie

### Performance
- Real-world burgerteksten (jurisprudentie n=17): bezwaar 0.77, woo_verzoek 0.75, klacht 0.44
- Handgeschreven realistic (n=30): macro-F1 0.90
- Adversarial (n=48): macro-F1 0.71

### Known issues / v0.2 prioriteiten
- Klacht-detectie zwak op juridisch-dichte en meta-klachten
- 6 van 9 labels nog niet vertegenwoordigd in real-world eval-set
- `vraag_overig` is restcategorie en lekt naar andere labels
- NER-pseudonimisering (namen/adressen/organisaties) is nog regex-only — spaCy-NER toevoegen in v0.2
