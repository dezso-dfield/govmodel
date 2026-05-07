# govmodel — Awb-typering classifier voor inkomende gemeentepost

> Eerste open-source NL-classifier voor Awb-typering. Sterke prestatie op echte burgerteksten uit jurisprudentie: **bezwaar (0.77)** en **WOO-verzoeken (0.75)** — samen het grootste deel van procedureel relevante inkomende post. Klacht-detectie werkt voor reguliere varianten; juridisch-complexe en meta-klachten (zoals klachten over de afhandeling van eerdere bezwaren) zijn de prioriteit voor v0.2. Zes overige Awb-typen voorlopig alleen synthetic-getoetst.

**Licentie:** EUPL-1.2 · **Contact:** [info@noaberai.nl](mailto:info@noaberai.nl) · **Status:** v0.1 — research baseline

🎮 **Probeer 'm live (geen install nodig):** [huggingface.co/spaces/NoaberAI/govmodel-demo](https://huggingface.co/spaces/NoaberAI/govmodel-demo)

---

## Wat is dit?

govmodel typeert binnenkomende digitale post bij Nederlandse gemeenten volgens de **Algemene wet bestuursrecht (Awb)**. Voor elk bericht — e-mail, webformulier, gescande brief — bepaalt het model welk Awb-type het is:

| Label | Toelichting | Awb-grondslag |
|---|---|---|
| `aanvraag` | Verzoek om beschikking | art. 1:3 lid 3, hfst. 4 |
| `bezwaar` | Verzet tegen eerder besluit | hfst. 6 + 7 |
| `beroep` | Administratief beroep bij ander bestuursorgaan | afd. 7.2 |
| `klacht` | Klacht over gedrag van bestuursorgaan | hfst. 9 |
| `melding` | Mededeling zonder besluitvraag | divers |
| `zienswijze` | Reactie op ontwerp-besluit / voornemen | art. 3:15, 4:8 |
| `woo_verzoek` | Verzoek openbaarmaking documenten | Woo art. 4.1 |
| `informatieverzoek_3_11` | Inzage stukken in lopende procedure | Awb art. 3:11 |
| `vraag_overig` | Reguliere correspondentie zonder Awb-procedure | — |

Multilabel: een brief kan tegelijk **bezwaar én klacht** zijn (en die combinatie wordt expliciet ondersteund).

## Voor wie

- Gemeentelijke KCC's en post-afdelingen die inkomende digitale stroom moeten classificeren
- Leveranciers van zaaksystemen (Suite4Sociaal, Civision, Decos JOIN, Mozard, Open Zaak) die een Awb-typering-component willen aanbieden
- Onderzoekers in NL gov-tech / Common Ground die op deze baseline willen bouwen

**Niet voor:** automatische besluitvorming. Het model **adviseert**, een mens **beslist**. Conform AI-verordening: deze classifier is *limited risk* (alleen routering, geen besluit over individu).

## Snel aan de slag

**Geen install — direct in je browser:** [Live Gradio demo](https://huggingface.co/spaces/NoaberAI/govmodel-demo) — plak een burgerbrief, krijg meteen het Awb-type.

**In je eigen Python (3 regels):**

```python
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="NoaberAI/govmodel-awb-classifier-v0.1",
    top_k=None,
)

text = "Hierbij teken ik bezwaar aan tegen uw besluit van 12 maart..."
print(clf(text))
# → [{'label': 'bezwaar', 'score': 0.91}, {'label': 'klacht', 'score': 0.04}, ...]
```

**Via HuggingFace Inference API (geen Python nodig):** zie de "Inference API" widget rechts op de [model-pagina](https://huggingface.co/NoaberAI/govmodel-awb-classifier-v0.1).

## Installatie

```bash
pip install transformers torch
# Voor GPU (Windows + CUDA 12.x):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Voor zelf trainen of de pipeline reproduceren:

```bash
git clone <repo-url>
cd govmodel
pip install -e ".[ml]"
```

## Performance

### Op echte burgerteksten (uit Nederlandse jurisprudentie, n=17)

| Label | F1 | n | Toelichting |
|---|---|---|---|
| **bezwaar** | **0.77** | 6 | Sterke detectie van bezwaarschriften over heffingen, WOZ, schadevergoeding |
| **woo_verzoek** | **0.75** | 5 | Robuust op uitvoerige Woo/Wob-verzoeken |
| klacht | 0.44 | 6 | Reguliere klachten gaan goed; juridisch-dichte en meta-klachten zijn een v0.2-prioriteit |
| 6 overige | n.v.t. | 0 | Niet vertegenwoordigd in jurisprudentie-set; alleen synthetic-getoetst |

### Op handgeschreven realistische voorbeelden (n=30)

Macro-F1: **0.90** (argmax). Per label tussen 0.75 en 1.00, behalve `vraag_overig` op 0.75.

### Op adversarial test set (n=48)

Macro-F1: **0.71** (argmax). Bevat NT2-Nederlands, juridisch jargon, multilabel-combinaties, advocaat-stijl, kort/emotioneel.

### Op synthetic test-split (n=216, zelfde distributie als training)

Macro-F1: **0.95** — *let op: deze meting bevestigt alleen dat het model de training-distributie heeft geleerd; voorspelt geen real-world prestatie. Bewust apart vermeld om misinterpretatie te voorkomen.*

Volledige metrics, per-label breakdown en verwarringsmatrices: zie [`MODEL_CARD.md`](MODEL_CARD.md).

## Hoe het is gemaakt

**Trainingsdata** (alleen publieke bronnen + LLM-augmentatie, geen casuïstiek uit gemeenten):
- ~860 synthetic burger-aan-gemeente brieven, gegenereerd door **Qwen3-4B** lokaal, verspreid over 9 labels × 11 schrijfstijlen (formeel/informeel/NT2/advocaat-jargon/chaotisch/etc.) plus 4 expliciete multilabel-combinaties
- Gepseudonimiseerd via regex-laag met BSN 11-proef, IBAN mod-97-validatie, e-mail/telefoon/postcode-redactie
- Geen persoonsgegevens in de gepubliceerde dataset

**Evaluatiedata**:
- 17 echte burger-citaten uit Open Data Rechtspraak (CC0)
- 30 handgeschreven realistische voorbeelden ("alledaagse stroom")
- 48 handgeschreven adversarial voorbeelden (multilabel, NT2, edge-cases)

**Architectuur**:
- Base model: [RobBERT-2023-base](https://huggingface.co/DTAI-KULeuven/robbert-2023-dutch-base) (KU Leuven, 124M parameters, MIT)
- Multilabel sigmoid classifier head, BCE-loss met `pos_weight=8` (1/9-prior balancering)
- Pure PyTorch training (geen HuggingFace Trainer wegens Windows-compatibiliteit)
- Per-class threshold-tuning op val set
- 10 epochs, AdamW + linear warmup, gradient clipping, bf16 mixed precision
- ~5 minuten training op een RTX 3050 Laptop (4 GB VRAM)

Volledige reproductie-instructies: zie [`DATASHEET.md`](DATASHEET.md).

## Gebruik in productie

Dit is een **research baseline**. Voordat je het in productie zet:

1. **Test op je eigen post** — synthetic-trained modellen kunnen domein-gap vertonen op specifieke gemeentelijke schrijfstijlen
2. **Houd mens-in-the-loop** — model adviseert, behandelaar beslist
3. **Log voorspellingen + correcties** — gebruik die als feedback voor v0.2-finetuning
4. **AI Act compliance** — limited risk, maar wel transparantie en uitlegbaarheid borgen
5. **Geen automatische besluitvorming** — strikt verboden in dit gebruik
6. **DPIA** — niet vereist voor het model zelf (alleen publieke trainingsdata), wél voor je integratie als je dossierdata classificeert

## Roadmap

**v0.2 (focus: real-world coverage)**
- Klacht-augmentatie: 50-100 meta-klacht en juridisch-klacht voorbeelden
- 6 maanden Rechtspraak ipv 1 maand → meer real-world variatie
- Hard-negative training op v0.1-fouten
- Real-world validatie design-partner-gemeente (alle 9 labels)

**v0.3 (focus: production-readiness)**
- Subtype-splitting van `vraag_overig` (informatie / bedank / intrekking)
- Calibrated confidence + abstain-mechanisme
- ONNX-export voor productie-inference
- Bias-evaluatie per stijl/taalniveau/herkomst

**Backlog**
- Meertalige inputs (Engels, Turks, Arabisch — frequente NT2-talen)
- Multimodale input (PDF + foto + OCR)
- Continuous learning vanuit gemeente-correcties

## Bijdragen

Issues, pull requests en feedback van gemeenten zijn welkom. Specifiek nuttig:
- Gemeenten die de classifier op echte (geanonimiseerde) post willen testen → [info@noaberai.nl](mailto:info@noaberai.nl)
- Juristen die als 2e labeller willen helpen bij gold-set uitbreiding
- Verbeteringen aan de pseudonymiserings-laag
- Additionele data-bronnen (Open Raadsinformatie, WOO-publicaties, etc.)

Bijdragen onder **DCO** (Developer Certificate of Origin) en EUPL-1.2.

## Licentie

[EUPL-1.2](LICENSE) — de Europese open-source licentie van overheidsorganisaties (VNG, ICTU, Common Ground). Compatibel met GPL/MPL voor downstream-projecten.

Model weights op HuggingFace: zelfde EUPL-1.2.
Trainingsdata-pipeline: zelfde EUPL-1.2.
Gepubliceerde datasets: CC-BY-4.0 (eigen creatie) of zoals oorspronkelijke licentie (Rechtspraak: CC0).

## Erkenning

- **RobBERT-2023** — Pieter Delobelle et al., KU Leuven Computational Linguistics
- **Open Data Rechtspraak** — De Rechtspraak (CC0)
- **KOOP / officielebekendmakingen.nl** — Kennis- en Exploitatiecentrum Officiële Overheidspublicaties
- **Common Ground** — VNG / Dimpact / ICTU voor het ecosysteem-denken

## Contact

[info@noaberai.nl](mailto:info@noaberai.nl)

Voor implementatie-ondersteuning, gemeente-pilots, of integratie met bestaande zaaksystemen.
