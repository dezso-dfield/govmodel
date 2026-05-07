# Data Sources — Awb-type-classifier v0.1

**Project:** open-source classifier voor Awb-typering van inkomende digitale gemeentepost.
**Versie:** v0.1 — alleen Awb-type-labeling, geen onderwerp/zaaktype.
**Status:** scoping — nog geen scrapers gebouwd.
**Doel van dit document:** per beoogde databron vastleggen wat we mogen, wat we kunnen halen, en wat het oplevert. Voor de start van de scrape moet alles met een ⚠️-vlag bevestigd zijn.

---

## 1. Wat we precies trainen

**Input:** een vrije-tekst poststuk (e-mail, webformulier-bericht, gescande/OCR'de brief, contactformulier-bericht) gericht aan een gemeente.

**Output (multilabel, sigmoid + abstain):**

| Label | Awb-grondslag | Behandelroute |
|---|---|---|
| `aanvraag` | art. 1:3 lid 3, hfst. 4 | beschikkingsprocedure, 8 wk termijn (verlengbaar) |
| `bezwaar` | hfst. 6 + 7 | bezwaarprocedure, 6 wk termijn |
| `(administratief) beroep` | hfst. 7 (admin. beroep) | doorzending bestuursrechter |
| `klacht` | hfst. 9 | klachtprocedure, 6 wk + 4 wk |
| `melding` | divers (geen Awb-route, vaak materieel) | operationele afhandeling |
| `zienswijze` | art. 3:15, art. 4:8 | meegewogen bij besluit |
| `WOO-verzoek` | Wet open overheid art. 4.1 | 4 wk termijn (verlengbaar) |
| `informatieverzoek (Awb 3:11)` | art. 3:11 | inzage ter inzage gelegde stukken |
| `vraag/overig` | — | reguliere correspondentie |

**Evaluatie**: macro-F1 per label, plus een "behandelroute-correctheid"-score: hoe vaak voorspelt het model de juiste juridische route, ook als labelt het meerdere typen (bv. een document is materieel én bezwaar én klacht — Awb art. 6:5 e.v.).

---

## 2. Juridisch kader voor dataverzameling

| Rechtsregel | Wat het zegt | Implicatie voor ons |
|---|---|---|
| Auteurswet art. 11 | Door overheid uitgevaardigde wetten, besluiten en verordeningen, en in openbaar belang bekendgemaakte verklaringen → **geen auteursrecht** | Wetten.nl-tekst, officiële bekendmakingen, raadsbesluiten zijn vrij te gebruiken |
| Auteurswet art. 15a | Citaatrecht | Korte citaten uit overige bronnen, mits met bronvermelding en gerechtvaardigd doel |
| Auteurswet art. 15o | TDM-exceptie (algemeen) | Tekst- en datamining toegestaan tenzij rechthebbende heeft opt-out (machine-readable). Voor commerciële OS in grijs gebied — meestal acceptabel met respect voor `robots.txt` en redelijke rate-limits |
| Databankenwet | Bescherming op "wezenlijke investering" in databank | Bulk-scrape kan schending zijn ook als items vrij zijn. Mitigatie: redelijke rate, geen 1:1 herpublicatie, attributie |
| AVG art. 5, art. 6, art. 89 | Persoonsgegevens, doelbinding, onderzoeksuitzondering | Pseudonimisering en/of NER-redactie verplicht voor publicatie van eigen dataset |
| AI Act (EU 2024/1689) | Risicocategorisering | Deze classifier = **limited risk** (geen individuele besluitvorming). Documenteren in modelcard. Geen Annex III. |

**Werkregel voor dit project**: alleen bronnen waarvan we (a) auteursrechtelijk vrij zijn óf (b) een TDM-grond hebben mét respect voor robots.txt en attributie. Persoonsgegevens worden vóór opname in de gepubliceerde dataset NER-geredacteerd. Twijfel = ⚠️ vlag, beslissen na juridische check.

---

## 3. Beoogde bronnen — overzicht

Geprioriteerd op (1) labelcoverage, (2) juridische zekerheid, (3) volume.

| # | Bron | Licentie | Mechanisme | Geschat volume v0.1 | Labels |
|---|---|---|---|---|---|
| A | Open Data Rechtspraak (uitspraken.rechtspraak.nl) | CC0 | bulk archives + LJN/ECLI-API | 5–15k bruikbare uitspraken → ~3–8k labelen voorbeelden | bezwaar, beroep, aanvraag, klacht, zienswijze |
| B | Officiële Bekendmakingen (KOOP / officielebekendmakingen.nl) | CC0 | SRU-API + bulk | 1–3k zienswijze-aanleidingen + voorbeeld-formats | zienswijze, aanvraag |
| C | Wetten.overheid.nl (KOOP) | CC0 | API + bulk | n.v.t. (geen voorbeelden, wel few-shot context) | alle (label-definities) |
| D | data.overheid.nl — WOO-besluitenfeed ⚠️ | CC0 (per dataset checken) | catalog + per-publisher | 0,5–2k WOO-verzoeken | WOO-verzoek |
| E | Open Raadsinformatie ⚠️ status onzeker | onduidelijk per gemeente | federated search (openraadsinformatie.nl indien actief) of per iBabs/GO-portal | 2–5k stukken | klacht, vraag, melding |
| F | Nationale Ombudsman (nationaleombudsman.nl) | onderzoeksrapporten doorgaans vrij ⚠️ check | scraper + handmatige selectie | 0,5–1k casuïstiek-fragmenten | klacht, vraag |
| G | Gemeentelijke voorbeeldformulieren (publieke gemeentewebsites) | per gemeente, meestal vrij citaat | gerichte scrape | n.v.t. (formats voor synthetic prompts) | alle |
| H | Synthetic LLM-augmentatie | zelfgegenereerd | Claude/Mistral/Llama prompts | 10–30k voorbeelden (5–10× augmentatie) | alle |
| I | Handgelabelde gold-eval-set | zelfgegenereerd | dubbel labelen, Cohen's κ | 300–500 voorbeelden | alle |

**Beoogde totale trainingsset v0.1: 20–50k voorbeelden, ~70% synthetic, ~30% echt.**

---

## 4. Per bron — detail

### A. Open Data Rechtspraak

**URL:** [uitspraken.rechtspraak.nl](https://uitspraken.rechtspraak.nl) · [data.rechtspraak.nl](https://data.rechtspraak.nl)
**Licentie:** CC0 sinds 2020 voor de Open Data feed.
**Mechanisme:** ECLI-API geeft per uitspraak XML met metadata + tekst. Bulk-archives beschikbaar.
**Selectiecriteria voor v0.1:**
- Bestuursrechtspraak: ABRvS, CRvB, CBb, rechtbanken sector bestuursrecht
- Periode: 2015–heden (post-Awb-revisie 2013)
- Filter op Awb-grondslag-citaten in tekst (`art. 6:4`, `art. 9:`, `art. 4:1`, `art. 3:15` etc.)
**Wat we eruit halen:**
- Letterlijk geciteerde burger-passages ("appellant heeft bij brief van … geschreven: …")
- Het oordeel van de rechter over het type (was het een aanvraag, een bezwaar, een klacht?)
- Negatieve voorbeelden (gemeente zag het als X, rechter oordeelt Y)
**Sterke punt:** dit zijn **gevalideerde labels** — een rechterlijk oordeel over typering. Goud.
**Zwak punt:** survivorship bias — alleen disputen halen de uitspraak. Eenvoudige duidelijke gevallen niet.
**Risico/aandacht:**
- Pseudonimisering: rechtspraak past al X-anonimisering toe. Aanvullend NER nodig.
- Zinscontext: extractie van burger-citaten vereist betrouwbare segmentatie.
**Pipelinetaak:** ECLI-puller → tekstextractie → Awb-grondslag-detector (regex) → citaat-extractie → handmatige sampling → label.

---

### B. Officiële Bekendmakingen (KOOP)

**URL:** [officielebekendmakingen.nl](https://www.officielebekendmakingen.nl) · [SRU-API documentatie via KOOP](https://repository.officiele-overheidspublicaties.nl)
**Licentie:** CC0 (Auteurswet art. 11).
**Mechanisme:** SRU/CQL-API met paginering. Bulk SRU-dumps mogelijk.
**Wat we halen:**
- Ontwerpbesluiten met zienswijze-procedure → context voor zienswijze-voorbeelden
- Verleende vergunningen → voorbeeld-aanvraag-formats
- Beleidsregels en gemeentelijke verordeningen → vocabulaire en context
**Beperking:** dit zijn de uitgaande overheid-stukken, niet de inkomende burgerstukken. Waarde zit in **context-priors** voor synthetic generation, niet directe voorbeelden.
**Pipelinetaak:** SRU-query per documenttype → indexing → koppeling aan synthetic prompts.

---

### C. Wetten.overheid.nl (Awb-tekst)

**URL:** [wetten.overheid.nl](https://wetten.overheid.nl) · KOOP API
**Licentie:** CC0.
**Gebruik:** geen training-voorbeelden, maar **label-definities, randgevallen-tekst en few-shot prompts** voor synthetic generation. De Awb zelf is onze normatieve bron voor wat een "bezwaar" is vs. een "klacht" vs. een "zienswijze".

---

### D. WOO-besluitenfeed via data.overheid.nl ⚠️

**URL:** [data.overheid.nl](https://data.overheid.nl) — zoeken op "Woo-besluiten", per publisher
**Licentie:** doorgaans CC0 maar **per dataset checken** ⚠️
**Status:** sinds 2022 (Wet open overheid) publiceren bestuursorganen WOO-besluiten. Dekking is in opbouw — niet alle gemeenten publiceren machine-leesbaar.
**Wat we halen:**
- Tekst van WOO-verzoeken (deel van het besluit, geanonimiseerd)
- Tekst van besluit (voor oordeel of het als WOO-verzoek werd behandeld)
**Verificatieactie vóór gebruik:** per dataset-publisher de licentie en herdistributie-status bevestigen.
**Pipelinetaak:** data.overheid.nl-catalog harvesten → per dataset metadata + content downloaden → WOO-verzoek-tekst extraheren.

---

### E. Open Raadsinformatie ⚠️ status onzeker

**URL:** historisch [openraadsinformatie.nl](https://openraadsinformatie.nl) (Open State Foundation), per gemeente iBabs/GO/RaadsInformatie-portals.
**Licentie:** per gemeente; meestal publiek toegankelijk maar herpublicatie-licentie verschilt ⚠️
**Verificatieactie vóór gebruik:** controleren of de federatieve API nog actief is en welke gemeenten nu participeren. Anders per gemeente-portal afzonderlijk.
**Wat we halen:**
- Brieven van burgers aan de raad ("ingekomen stukken") — dit zijn vaak letterlijk klachten, vragen, zienswijzen, meldingen
- Geanonimiseerd voor publicatie, maar tekst zelf intact
**Sterke punt:** **echte burger-aan-gemeente-correspondentie**, publiek beschikbaar, doorgaans al geredacteerd door gemeente.
**Zwak punt:** alleen stukken die ter agendering bij de raad gaan — selectiebias richting het politiek geladen segment.

---

### F. Nationale Ombudsman casuïstiek ⚠️

**URL:** [nationaleombudsman.nl](https://www.nationaleombudsman.nl) — rapporten en jaarverslagen
**Licentie:** ⚠️ check per rapport; ombudsman-rapporten zijn doorgaans vrij citeerbaar, maar bulk-extractie en herdistributie vereist verificatie.
**Wat we halen:**
- Casuïstiek-fragmenten waarin burgers schreven en de typering werd verkeerd uitgevoerd
- Edge cases (klacht behandeld als bezwaar etc.)
**Volume:** klein maar hoge kwaliteit.
**Pipelinetaak:** scraper op rapportencatalogus → tekstextractie → handmatige selectie van bruikbare fragmenten.

---

### G. Gemeentelijke voorbeeldformulieren

**Bron:** publieke gemeentewebsites (bezwaarformulier, klachtenformulier, zienswijzeformulier, WOO-verzoekformulier).
**Gebruik:** geen training-voorbeelden, maar **format- en formulierings-priors** voor synthetic generation. Helpt synthetic data realistischer te maken (typische openings- en slotzinnen, juiste juridische formuleringen).

---

### H. Synthetic data via LLM-augmentatie

**Methode:** few-shot prompting van een capabele LLM met:
- Awb-definitie (uit C)
- Echte voorbeelden uit (A) en (E)
- Stijlvariatie-instructies: formeel/informeel, B1/B2/C1, NT2-Nederlands, gefrustreerd/zakelijk, kort/lang, met/zonder typische openingen ("Geachte heer/mevrouw", "Hoi"), met spelfouten/zonder
- Demografische diversiteit-instructies (waar relevant)

**Modelkeuze:**
- **Aanbevolen voor classifier-training:** een lokaal of open-weight model (Mistral, Llama, Qwen) — vermijdt ToS-issues van commerciële API's
- **Aanbevolen voor seed-generatie en handmatige curatie:** Claude/GPT-4 voor kwaliteit, daarna handmatig valideren
- **ToS-attentiepunt:** OpenAI/Anthropic verbieden gebruik van output voor training van *competing LLMs*. Een classifier-encoder is geen competing LLM; juridisch grijs maar pragmatisch verdedigbaar. Documenteren in datasheet.

**Augmentatieratio:** 5–10× ten opzichte van echte voorbeelden. Per echt voorbeeld 5–10 paraphrasings/stijlvariaties.

**Risico:** synthetic data kan sluipende biases introduceren (LLM-clichés, te formele formulering). Eval **uitsluitend op echte data** uit (I).

---

### I. Handgelabelde gold-eval-set

**Doel:** betrouwbare evaluatie. Niet voor training.
**Methode:**
- Sample 300–500 voorbeelden uit (A) + (E) + (F)
- Twee onafhankelijke labellers (bij voorkeur één jurist/Awb-kenner, één generalist)
- Cohen's κ als overeenstemmingsmaat — minimaal 0,75 vereist
- Bij disagreement: derde rondje met discussie
**Publicatie:** als aparte dataset onder CC-BY-4.0 (eigen creatie). Belangrijk voor reproduceerbare benchmarks.

---

## 5. Wat we *niet* gebruiken — en waarom

| Bron | Reden |
|---|---|
| Reddit / Twitter / X / Tweakers / forums | ToS, privacy, niet representatief voor gemeentecorrespondentie |
| Echte zaaksysteem-content uit gemeenten | AVG, niet OS-publicabel zonder DPIA + designpartner-overeenkomst (kandidaat voor v0.2) |
| Stimulansz / Divosa / Schulinck publicaties | Commercieel beschermd, geen TDM-grond |
| Krantenartikelen over gemeentecasussen | Auteursrecht zonder duidelijke TDM-grondslag |
| Chatlogs van gemeentelijke chatbots | Per gemeente verschillend, vrijwel altijd niet OS-publicabel |
| Social media DM's aan gemeenten | Privacy + ToS van platform |

---

## 6. Bias- en representativiteitsrisico's

| Risico | Bron | Mitigatie |
|---|---|---|
| Survivorship bias bij rechtspraak | A | Bewust mengen met (E) en (H) waar typering ongedisputeerd is |
| Politieke selectiebias bij raadsinformatie | E | Diverse gemeenten samplen, niet alleen randstad |
| LLM-stijlbias bij synthetic | H | Verplichte stijl-checks, eval alleen op echte data |
| Onderrepresentatie NT2/laaggeletterd Nederlands | alle | Expliciete stijl-instructies in (H), gold-eval-set bewust meenemen |
| Onderrepresentatie korte/informele berichten | alle | E-mail/contactformulier-formats specifiek genereren in (H) |

Bias-eval verplicht in modelcard, gesegmenteerd per: lengte, formaliteitsniveau, taalniveau (B1/B2/C1), onderwerp.

---

## 7. Privacy-pipeline (verplicht vóór publicatie van dataset)

Elke voorbeeldtekst doorloopt:

1. **NER-redactie** met Dutch-NER model → vervangen van persoonsnamen, adressen, BSN-achtige patronen, kentekens, e-mailadressen, telefoonnummers door placeholders
2. **Regex-veiligheidsnet** voor bekende patronen (BSN, IBAN, postcode+huisnummer)
3. **Steekproef-handmatige check** op 5% van dataset
4. **Risico-acceptatie-statement** in datasheet: rest-risico geaccepteerd, contactpunt voor takedown-verzoeken vermeld

---

## 8. Open vragen — beslissen vóór scrape-bouw

| Vraag | Wie beslist | Deadline |
|---|---|---|
| Open Raadsinformatie federated API nog actief? | te checken | vóór week 1 |
| WOO-feed coverage per publisher: welke gemeenten hebben nu CC0-feeds? | te checken | vóór week 1 |
| Welke LLM voor synthetic? Open-weight (Mistral/Llama) of commercieel (Claude/GPT) of beide? | Joran | vóór week 2 |
| Eén of twee labellers voor gold-set? | Joran | vóór week 3 |
| Naam van het project + GitHub-repo (voorlopig privé) | Joran | vóór week 1 |
| Pseudonimisering: bestaand NL-NER-model (bv. spaCy nl_core_news_lg, of fine-tuned) of dedicated PII-model? | technische keuze | vóór week 2 |
| Datasheet for datasets template gebruiken? (Gebru et al.) | aanbevolen ja | vóór publicatie |

---

## 9. Geschatte tijdlijn

| Week | Activiteit |
|---|---|
| 1 | Verificatie ⚠️-bronnen, repo-skeleton (privé), Rechtspraak-puller |
| 2 | KOOP-puller, eerste handmatige labelingsspreadsheet (50 voorbeelden) |
| 3 | Open Raadsinformatie / Ombudsman scrapers, eerste 200 echte voorbeelden gelabeld |
| 4 | Synthetic data pipeline, NER-pseudonimisering |
| 5 | Eerste training run (RobBERT-2023 + multilabel head), baseline |
| 6 | Gold-eval-set bouwen (300+), bias-eval, modelcard, datasheet |
| 7 | Tuning, herhaalde training, eval-iteratie |
| 8 | Publieke release: HF Hub + GitHub + blog |

---

## 10. Bronnen die nog onderzocht moeten worden

- VNG Realisatie publicaties (gemeente-correspondentie-cases)
- Sociaal raadsliedenwerk-publicaties (anoniem casuïstiek)
- Universitaire taalbeheersings-corpora (RU Nijmegen, UvA, Tilburg) — mogelijke gelabelde NL-overheidstekst
- Common Ground community: bestaande gemeentelijke datasets die niet via standaard catalogi vindbaar zijn
- DSO (Digitaal Stelsel Omgevingswet) — vergunning-correspondentie publiek beschikbaar?

Toevoegen aan dit document zodra onderzocht.
