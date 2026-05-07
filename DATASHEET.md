# Datasheet — govmodel-awb-classifier-v0.1 datasets

Volgens Gebru et al. ([Datasheets for Datasets](https://arxiv.org/abs/1803.09010)). Beschrijft de drie datasets die in v0.1 zijn gebruikt: synthetic training/val, handgeschreven evaluation, en jurisprudentie-extractie.

---

## 1. Motivatie

### Waarom is deze dataset gemaakt?

Om een Nederlandse Awb-typering classifier te trainen en evalueren zonder afhankelijkheid van casuïstiek uit gemeenten (privacy-gevoelig, AVG-blokker voor v0.1). De combinatie van LLM-augmentatie + publieke bronnen geeft een verdedigbare baseline zonder design-partner-overeenkomst.

### Wie heeft de dataset gemaakt?

Noaber AI (auteur: Joran van Beek), 2026.

### Wie heeft gefinancierd?

Geen externe financiering. Eigen tijd en lokale GPU.

---

## 2. Samenstelling

### Wat zit in elke dataset?

#### `data/processed/synthetic/v0.2.jsonl` — 856 voorbeelden

LLM-gegenereerde burger-aan-gemeente brieven.

| Veld | Type | Beschrijving |
|---|---|---|
| `id` | string (uuid) | Uniek per voorbeeld |
| `text` | string | De gegenereerde brieftekst (na pseudonymisering) |
| `source` | enum | Altijd `"synthetic"` |
| `labels` | list[label] | Awb-type label(s) — single of multi |
| `notes` | string | `style=<naam>; model=<llm>` |
| `language` | string | `"nl"` |
| `pseudonymized` | bool | `true` |

Distributie:
- 9 labels: aanvraag, bezwaar, beroep, klacht, melding, zienswijze, woo_verzoek, informatieverzoek_3_11, vraag_overig
- 11 styles: formal_short/long, informal_short/medium, angry_medium, polite_questioning, nt2_simple, elderly_handwritten_style, legal_dense, rambling_long, chaotic_typing
- 64 multilabel voorbeelden (bezwaar+klacht, aanvraag+klacht, aanvraag+vraag_overig, zienswijze+vraag_overig)

#### `data/eval/gold_realistic.jsonl` — 30 voorbeelden

Handgeschreven realistische voorbeelden door één auteur. Bedoeld als "alledaagse KCC-stroom" — wat een gemeentemedewerker dagelijks ziet.

| Label | n |
|---|---|
| aanvraag | 4 |
| bezwaar | 4 |
| beroep | 1 |
| klacht | 4 |
| melding | 4 |
| zienswijze | 3 |
| woo_verzoek | 3 |
| informatieverzoek_3_11 | 2 |
| vraag_overig | 5 |

Alle single-label, hoge confidence.

#### `data/eval/adversarial.jsonl` — 48 voorbeelden

Handgeschreven hard cases: multilabel-combinaties, NT2-Nederlands, juridisch-dichte advocaat-stijl, hoofdletter-emotie, ééncijfer-berichten ("bezwaar"), Engels, intrekking, hardheidsclausule-beroep, anonieme klacht, etc.

Distribution:
- 6 multilabel
- 14 vraag_overig
- 12 bezwaar, 10 klacht, 5 aanvraag, 4 zienswijze, 4 melding, 3 woo, 2 beroep, 1 informatieverzoek_3_11

Confidence-niveaus variëren (`hoog`/`middel`/`laag`); `borderline_flags` gemarkeerd waar relevant.

#### `data/eval/gold_real.jsonl` — 17 voorbeelden

**Echte burger-citaten** geëxtraheerd uit Nederlandse jurisprudentie (Open Data Rechtspraak, januari 2024). Alle uit publieke uitspraken waar de rechter expliciet quoteert wat een burger heeft geschreven.

| Label | n |
|---|---|
| bezwaar | 6 |
| klacht | 6 |
| woo_verzoek | 5 |

Andere 6 labels: niet vertegenwoordigd (jurisprudentie gaat vooral over geschillen).

Veld `notes` bevat de bron-ECLI per voorbeeld voor traceability.

### Hoeveel voorbeelden in totaal?

| Dataset | Voorbeelden |
|---|---|
| Synthetic training/val | 856 |
| Handgeschreven realistic | 30 |
| Handgeschreven adversarial | 48 |
| Echte burger-citaten | 17 |
| **Totaal** | **951** |

### Bevatten data persoonsgegevens?

**Synthetic**: nee, gepseudonymiseerd via regex (BSN 11-proef, IBAN mod-97, e-mail/telefoon/postcode). 100 van 856 records hadden Qwen3-verzonnen PII die geredacteerd is.

**Handgeschreven**: gebruikt placeholders zoals `[naam]`, `[adres]` — geen echte personen.

**Echte burger-citaten**: rechtspraak-data is **al gepseudonimiseerd** door de Rechtspraak (`[appellant]`, `[plaats]`, `[Z]`, etc.). Aanvullende regex-laag toegepast voor zekerheid.

### Zijn er gevoelige onderwerpen?

Sommige adversarial- en gold_real-voorbeelden raken gevoelige domeinen: schuldhulpverlening, dementie/wettelijke vertegenwoordiging, jeugdzorg, GGZ-context, ontslag-procedures. Allen *over* die onderwerpen, niet *met* personen-data.

---

## 3. Verzamelproces

### Hoe zijn de synthetic data verzameld?

Via lokale LLM-server (LM Studio met Qwen3-4B-Instruct Q4_K_M):
1. Per (label × style)-combinatie: 8 generatie-aanroepen via OpenAI-compatible chat completion API
2. Per multilabel-combinatie: 4 generatie-aanroepen × 4 styles
3. Temperatuur: 0.85
4. Output-validatie: lengte 30-4000 chars, geen meta-prefixen, geen "Als AI..."-openingen
5. Deduplicatie: niet expliciet (relevant: temperatuur 0.85 + verschillende styles = lage duplicate-rate)

Code: [`src/govmodel/synthetic.py`](src/govmodel/synthetic.py), [`scripts/generate_synthetic.py`](scripts/generate_synthetic.py).

### Hoe zijn de handgeschreven evaluatiedata verzameld?

Door één auteur (Joran van Beek), best-effort gelabeld volgens [`labeling-guidelines.md`](labeling-guidelines.md). Beperking: één labeller, geen Cohen's κ tegen tweede labeller. Voor v0.2 voorzien: tweede onafhankelijke labeller (jurist).

### Hoe zijn de echte burger-citaten verzameld?

1. **Open Data Rechtspraak puller** ([`scripts/pull_rechtspraak.py`](scripts/pull_rechtspraak.py)) downloadde alle bestuursrechtspraak van januari 2024 (391 records met relevante Awb/Woo-citaten)
2. **Burger-citaten-extractor** ([`scripts/extract_burger_citaten.py`](scripts/extract_burger_citaten.py)) zocht in elke uitspraak naar quoted text voorafgegaan door `actor + intro-verb`-patroon (bijv. "appellant heeft geschreven: ...")
3. **40 candidates** uit 391 records, gefilterd op quality_score (eerste-persoon, brief-achtige formulering, geen rechter-jargon binnen quote)
4. **Handmatige verificatie**: 17 van 40 zijn daadwerkelijk burgertekst (rest is wettekst, intern overheids-mail, expert-verklaring, of besluitstekst)

### Wanneer is de data verzameld?

- Synthetic + handgeschreven: 7 mei 2026
- Rechtspraak: januari 2024 publicaties, geëxtraheerd 7 mei 2026

### Ethische review?

Niet vereist (geen personen, geen experimentele setup). Wel: een interne reflectie op AI Act-positionering en bias-overwegingen ([MODEL_CARD.md](MODEL_CARD.md)).

---

## 4. Preprocessing / cleaning

### Welke preprocessing is toegepast?

#### Synthetic
1. Output-cleaning: strip markdown code-fences, leading/trailing quotes, meta-prefixes ("Brief:", "Bericht:")
2. Whitespace-normalisatie
3. Validatie: minimum 30 / maximum 4000 chars, geen "Als AI..."/"Natuurlijk!"/"Zeker,"-openingen
4. PII-pseudonymisering via regex (zie [Pseudonimisering](#pseudonimisering) hieronder)

#### Rechtspraak burger-citaten
1. Extractie via regex op quote-tekens en actor+verb-context
2. Quality-score op heuristieken (eerste-persoon-pronouns, brief-formulering, lengte, afwezigheid rechter-jargon)
3. Handmatige selectie en labeling
4. PII-pseudonymisering toegepast als extra veiligheidslaag (Rechtspraak doet dit al)

### Pseudonimisering

Regex-laag in [`src/govmodel/pii.py`](src/govmodel/pii.py):

| Patroon | Validatie |
|---|---|
| BSN | 9 cijfers + 11-proef |
| IBAN-NL | NL\d{2}[A-Z]{4}\d{10} + mod-97 |
| E-mail | RFC 5322 simplified |
| Telefoonnummer | Alleen werkelijke NL-kengetallen (geen 9-cijferige zaaknummers gemarkeerd) |
| Postcode | 4 cijfers + 2 letters, optioneel huisnummer |

Aanvullende laag voor namen / adressen / organisaties via NER: **niet in v0.1**, voorzien voor v0.2 met spaCy of fine-tuned NL PII-NER.

---

## 5. Gebruik

### Voor welke taken kan de dataset gebruikt worden?

- Awb-typering training en evaluatie (primair)
- Onderzoek naar synthetic-to-real domain gaps in NL gov-tekst-classificatie
- Benchmark voor andere NL Awb-classifiers
- Stijlanalyse-onderzoek (synthetic dataset bevat 11 expliciete styles)

### Niet geschikt voor

- Trainen van content-generation modellen (synthetic data is niet bedoeld als demonstratie van Nederlands taalgebruik)
- Identificatie van individuele burgers (alle data is gepseudonymiseerd)
- Beoordeling van inhoudelijke geldigheid van bezwaren/klachten (alleen typering)

---

## 6. Distributie

### Hoe wordt de dataset gedistribueerd?

- Synthetic + handgeschreven evaluatie: gepubliceerd onder dezelfde repository, **CC-BY-4.0** (eigen creatie van Noaber AI)
- Rechtspraak-data: bron is **CC0** (Open Data Rechtspraak); onze geëxtraheerde + gelabelde versie eveneens **CC-BY-4.0** (de labeling/extractie-component is eigen werk)
- Hosting: GitHub repository + HuggingFace Hub dataset-page (`NoaberAI/govmodel-awb-data-v0.1`)

### Wanneer beschikbaar?

Bij v0.1 release (mei 2026).

---

## 7. Onderhoud

### Wie onderhoudt?

Noaber AI ([info@noaberai.nl](mailto:info@noaberai.nl)).

### Hoe wordt feedback verwerkt?

- GitHub Issues voor data-fouten of label-disagreements
- Email voor privacy-meldingen of takedown-verzoeken

### Hoe wordt de dataset bijgewerkt?

- v0.2: meer rechtspraak (Q1-Q2 2024), klacht-augmentatie, gemeente-design-partner-data
- Versies semver: v0.1 → v0.2 → v1.0
- Wijzigingen gelogd in `CHANGELOG.md` van de repo

### Hoe lang ondersteund?

v0.1 blijft beschikbaar als research baseline. Productie-inzet wordt aanbevolen om naar nieuwste versie te migreren bij major releases.

---

## Contact

Voor vragen, takedown-verzoeken, of design-partnership: [info@noaberai.nl](mailto:info@noaberai.nl).
