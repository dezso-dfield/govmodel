# Labeling Guidelines — Awb-type-classifier v0.1

**Doel:** consistent en juridisch correct labelen van inkomende gemeentepost op Awb-type. Dit document is normatief voor labellers én voor de classifier-evaluatie.

**Gebruikers:** labellers (jurist + generalist), reviewers, en model-evaluators.

**Versie:** 0.1 — open voor revisie tijdens de eerste 100 gelabelde voorbeelden, daarna gefroren tot v0.2.

---

## 1. Het labelschema

Negen labels, allemaal **multilabel** (een document kan meerdere labels dragen). Per label: Awb-grondslag, kernkenmerk, behandelroute, en wat het *niet* is.

### 1.1 `aanvraag`

> **Awb art. 1:3 lid 3:** "Onder aanvraag wordt verstaan: een verzoek van een belanghebbende, een besluit te nemen."

**Kernkenmerk:** burger vraagt het bestuursorgaan om een nieuwe beschikking.
**Behandelroute:** beschikkingsprocedure (titel 4.1 Awb), termijn 8 weken (verlengbaar met redelijke termijn).
**Voorbeelden:** WMO-voorzieningenaanvraag, bouwvergunning, paspoort, kwijtschelding gemeentebelasting, subsidie, urgentieverklaring, parkeervergunning, leerlingenvervoer, bijzondere bijstand.
**Wat het niet is:**
- Niet "ik wil opnieuw kijken naar besluit X" → dat is `bezwaar`
- Niet "ik ben het oneens met afwijzing" → dat is `bezwaar`
- Niet "ik wil informatie over hoe ik X aanvraag" → dat is `vraag/overig`

### 1.2 `bezwaar`

> **Awb art. 7:1 jo. 6:4 lid 1:** "Het maken van bezwaar geschiedt door het indienen van een bezwaarschrift bij het bestuursorgaan dat het besluit heeft genomen."

**Kernkenmerk:** de burger is het oneens met een **eerder genomen besluit** en wil heroverweging.
**Behandelroute:** bezwaarprocedure (titel 7.2 Awb), termijn 6 weken na bekendmaking, beslistermijn 6 wk + 6 wk verdaging.
**Vereisten Awb 6:5:** schriftelijk, ondertekend, naam/adres, datering, omschrijving besluit, gronden.
**Belangrijk:** ontbreken van formele kenmerken maakt het *niet-ontvankelijk*, maar nog steeds materieel een `bezwaar`. Verzuim kan hersteld worden (Awb 6:6). **Voor labeling tellen we het materiële karakter, niet de ontvankelijkheid.**
**Voorbeelden:**
- "Ik ben het niet eens met de afwijzing van mijn WMO-aanvraag d.d. 12 maart"
- "Verzoek tot heroverweging van besluit nummer 2024-1234"
- "Hierbij teken ik bezwaar aan tegen…"
- "Ik snap niet waarom mijn vergunning is geweigerd, kunnen jullie hier nog eens naar kijken?" (mits gericht tegen bestaand besluit)
**Wat het niet is:**
- Klacht over de medewerker die het besluit communiceerde → `klacht`
- Reactie op een *ontwerp*-besluit (nog niet definitief) → `zienswijze`
- Beroep bij de rechter → niet relevant voor gemeente-intake

### 1.3 `(administratief) beroep`

> **Awb afd. 7.2 jo. 1:5 lid 2.**

**Kernkenmerk:** beroep op een **ander bestuursorgaan** dan dat het besluit nam (bv. provincie tegen gemeente). Zelden bij gemeenten direct, maar komt voor.
**Behandelroute:** beroepsprocedure bij hoger bestuursorgaan.
**Voor v0.1:** apart label houden, ook al is het volume klein. Misclassificatie als `bezwaar` is een procedurele fout.

### 1.4 `klacht`

> **Awb art. 9:1 lid 1:** "Een ieder heeft het recht om over de wijze waarop een bestuursorgaan zich in een bepaalde aangelegenheid jegens hem of een ander heeft gedragen, een klacht in te dienen."

**Kernkenmerk:** burger klaagt over **gedrag** van het bestuursorgaan of een medewerker — bejegening, communicatie, proces, traagheid, fouten in uitvoering.
**Behandelroute:** klachtprocedure (titel 9.1 Awb), 6 wk + 4 wk verdaging.
**Voorbeelden:**
- "Ik ben drie weken aan het lijntje gehouden door uw afdeling"
- "Een medewerker was onbeschoft tegen mij"
- "Ik krijg geen antwoord op mijn vragen"
- "De brief stond vol fouten en was onbegrijpelijk"
**Wat het niet is:**
- Klacht over de *uitkomst* van een besluit → `bezwaar` (zie ook §3.1)
- Algemene ontevredenheid zonder concreet gedrag → kan `vraag/overig` zijn
- Politieke onvrede ("ik vind het beleid slecht") → meestal `vraag/overig` of buiten Awb-typering

### 1.5 `melding`

**Geen specifieke Awb-grondslag** — meldingen zijn vaak materieel-rechtelijk verankerd in andere wetten of beleidsregels (Wabo, APV, leerplichtwet, OOV-meldingen).

**Kernkenmerk:** burger maakt iets bekend zonder dat een besluit wordt gevraagd; vaak operationeel of informerend.
**Voorbeelden:**
- Melding ingebruikname horecagelegenheid (sluitingstijd-melding)
- Melding incident, overlast, gevaar (niet-MOR — die zou via Signalen gaan)
- Verhuismelding (kan ook `aanvraag` zijn, zie §3.5)
- Melding leerplichtverzuim
- Melding sloop onder de meldingsplicht
**Wat het niet is:**
- MOR (meldingen openbare ruimte buiten/dingen) → buiten scope, verwijzen naar Signalen
- Gewone vraag → `vraag/overig`
- Verzoek om beschikking → `aanvraag`

### 1.6 `zienswijze`

> **Awb art. 3:15** (uitgebreide voorbereidingsprocedure) **/ art. 4:8** (voornemen tot belastend besluit).

**Kernkenmerk:** reactie van een belanghebbende op een **ontwerp**-besluit dat ter inzage ligt, of op een voornemen tot beschikking dat de burger raakt.
**Behandelroute:** wordt meegewogen bij het definitieve besluit. Geen procedurele beslistermijn voor de zienswijze zelf.
**Voorbeelden:**
- Reactie op ontwerp-omgevingsplan
- Reactie op concept-vergunning die ter inzage ligt
- Reactie op voornemen tot intrekking subsidie (4:8)
**Wat het niet is:**
- Reactie op een *definitief* besluit → `bezwaar`
- Reactie zonder concreet voornemen/ontwerp → meestal `vraag/overig`

### 1.7 `WOO-verzoek`

> **Wet open overheid art. 4.1.**

**Kernkenmerk:** verzoek om openbaarmaking van publieke documenten/informatie. Aanvrager hoeft *geen* belanghebbende te zijn.
**Behandelroute:** WOO-procedure, termijn 4 wk + 2 wk verdaging.
**Voorbeelden:**
- "Verzoek tot openbaarmaking van alle correspondentie tussen wethouder X en bedrijf Y"
- "WOO-verzoek inzake besluitvorming over project Z"
- "Ik wil graag alle stukken over dossier 2024-1234"
**Wat het niet is:**
- Verzoek om eigen dossier in te zien → AVG-inzage of `informatieverzoek (Awb 3:11)`, niet WOO
- Algemene informatievraag ("hoe werkt subsidie X?") → `vraag/overig`
- Verzoek om documenten die *al* gepubliceerd zijn op de website → `vraag/overig`

### 1.8 `informatieverzoek (Awb 3:11)`

> **Awb art. 3:11.**

**Kernkenmerk:** verzoek om inzage in stukken die ter inzage zijn gelegd in het kader van een lopende uitgebreide voorbereidingsprocedure.
**Smal label.** Onderscheidend van WOO doordat het specifiek is gekoppeld aan een lopende procedure (3.4).
**Voorbeelden:**
- "Ik wil de onderliggende stukken bij het ontwerp-bestemmingsplan inzien"
- "Verzoek om inzage in de stukken bij de ter inzage gelegde vergunning"

### 1.9 `vraag/overig`

**Restcategorie** voor reguliere correspondentie zonder Awb-procedure-trigger.
**Voorbeelden:**
- "Wanneer is het loket open?"
- "Hoe maak ik een afspraak voor mijn paspoort?"
- "Klopt het dat ik X moet aanvragen voor Y?"
- "Bedankt voor uw hulp"
- Algemene suggesties / ideeën / politieke meningen zonder besluit-context
**Belangrijk:** dit is geen "weet ik niet"-label. Bij echte twijfel → abstain (geen enkel label boven threshold).

---

## 2. Multilabel-regels

Een document kan **meerdere labels tegelijk** dragen wanneer het materieel beide aspecten heeft. Voorbeelden:

| Combinatie | Voorbeeld |
|---|---|
| `bezwaar` + `klacht` | "Ik teken bezwaar aan tegen de afwijzing en bovendien klaag ik dat het zo lang heeft geduurd voor ik antwoord kreeg" |
| `aanvraag` + `bezwaar` | "Ik dien hierbij een nieuwe aanvraag in voor X, en tegen de eerdere afwijzing van Y maak ik bezwaar" |
| `aanvraag` + `klacht` | "Ik wil graag een nieuwe afspraak — overigens vond ik vorige keer de bejegening onder de maat" |
| `melding` + `vraag/overig` | "Ik meld hierbij dat ik ga verhuizen, en kunnen jullie laten weten of mijn afvalpas automatisch meegaat?" |

**Regel:** label álle materieel aanwezige typen. De classifier moet leren dat documenten gemengd kunnen zijn — dat is realiteit, niet edge case.

**Niet-combineerbare paren:**
- `bezwaar` + `zienswijze`: kan niet — bezwaar gaat tegen definitief besluit, zienswijze tegen ontwerp. Wel mogelijk: een document waarin staat "voor het geval dit als ontwerp telt is dit een zienswijze, voor het geval het al definitief is een bezwaar" → in zo'n geval beide labelen, classifier gaat dat niet vaak zien.
- `WOO-verzoek` + `informatieverzoek (Awb 3:11)`: in beginsel wederzijds uitsluitend (verschillende grondslagen), maar bij twijfel beide.

---

## 3. Beslisboom voor grensgevallen

### 3.1 Bezwaar of klacht?

**Vuistregel:** richt het zich tegen een **besluit** of tegen **gedrag**?

```
Is er een eerder besluit/beschikking waar dit document tegen ageert?
├── Ja, en de inhoud van dat besluit wordt betwist     → bezwaar
├── Ja, maar de klacht gaat over hoe het is gegaan
│   (lange behandeling, slechte communicatie, etc.)    → klacht (en eventueel ook bezwaar)
└── Nee, geen specifiek besluit                        → klacht of vraag/overig
```

**Cruciale randgevallen:**
- "Ik vind het schandalig dat ik dit moest aanvragen en dat het zo lang duurde, en ik ben het ook niet eens met de uitkomst" → **`bezwaar` + `klacht`** (multilabel)
- "Ik klaag over dat mijn aanvraag is afgewezen" — woord 'klacht' gebruikt maar materieel tegen besluit → **`bezwaar`** (woordkeuze burger is niet leidend)
- "Ik ben heel ontevreden over de afhandeling van X" zonder besluit-referentie → **`klacht`**

### 3.2 Aanvraag of bezwaar?

**Vuistregel:** is er al een besluit dat heroverwogen moet worden, of vraagt de burger iets nieuws?

```
Is er een eerdere afwijzing/beschikking?
├── Ja, en burger wil heroverweging                                    → bezwaar
├── Ja, maar burger vraagt iets *anders* aan (nieuwe situatie)         → aanvraag
└── Nee, eerste verzoek                                                → aanvraag
```

**Randgevallen:**
- "Ik vraag opnieuw aan omdat de situatie is veranderd" → meestal `aanvraag` (nieuwe feiten, nieuwe aanvraag), tenzij binnen 6 wk na afwijzing en zonder nieuwe feiten — dan kan het `bezwaar` zijn
- "Verzoek tot herziening van besluit X" — formeel geen Awb-instrument, materieel meestal `bezwaar` als binnen termijn, anders `vraag/overig` of nieuwe `aanvraag`

### 3.3 Bezwaar of zienswijze?

**Vuistregel:** is het besluit *definitief* of nog *ontwerp*?

```
Status van het besluit waartegen burger reageert:
├── Definitief genomen besluit                                          → bezwaar
├── Ter inzage gelegd ontwerp (uitgebreide voorbereidingsprocedure 3.4) → zienswijze
└── Voornemen tot belastend besluit (4:8)                               → zienswijze
```

### 3.4 WOO-verzoek of vraag/overig?

**Vuistregel:** vraagt burger om **specifieke documenten/informatie** uit overheidsadministratie, of om algemene uitleg?

```
├── Specifieke documenten over een bestuurlijke aangelegenheid     → WOO-verzoek
├── Algemene informatievraag ("hoe werkt X?")                      → vraag/overig
├── Verzoek om eigen dossier (zaken-eigenaar)                      → vraag/overig
│   (formeel AVG-inzage, voor v0.1 in restcategorie)
└── Verzoek om gepubliceerde info ("kunt u mij artikel sturen?")   → vraag/overig
```

### 3.5 Melding of aanvraag?

**Vuistregel:** is er een **besluit nodig** of niet?

```
├── Burger informeert, geen besluit gevraagd                       → melding
├── Burger vraagt impliciet om beschikking (registratie/akkoord)   → aanvraag
└── Twijfelgeval verhuismelding:
    ├── louter mededeling van wijziging GBA-gegevens              → melding
    └── mits er een beschikkingscomponent is (urgentie etc.)      → aanvraag of beide
```

---

## 4. Edge cases — concrete moeilijkheden

| # | Casus | Labeling |
|---|---|---|
| 1 | Document zonder afzender of datum | Materieel labelen op tekst, in opmerkingen noteren `formeel-incompleet` |
| 2 | Bezwaar buiten 6-weken-termijn | Nog steeds `bezwaar` (ontvankelijkheid is procedureel, niet materieel) |
| 3 | Brief in C1+-niveau, juridische taal van advocaat | Standaard labelen — formaliteitsniveau is geen labelcriterium |
| 4 | Brief in NT2-Nederlands of laaggeletterd | Standaard labelen — taalniveau is geen labelcriterium |
| 5 | E-mail met aanhangend ingevuld formulier | Beide labelen; formulier-inhoud is leidend, e-mail is begeleidend |
| 6 | Boze brief zonder concreet verzoek of klacht | `vraag/overig` met opmerking `borderline-emotioneel`; abstain mogelijk |
| 7 | Brief gericht aan "de gemeente" zonder specifieke afdeling | Standaard labelen |
| 8 | Brief van advocaat namens cliënt | Standaard labelen — vertegenwoordiging maakt niet uit voor type |
| 9 | Document in Engels of andere taal | Markeer met `niet-NL`, label op basis van inhoud (model wordt op NL getraind, deze gevallen zijn out-of-scope voor v0.1 maar moeten herkenbaar zijn) |
| 10 | Burger gebruikt verkeerde terminologie ("hoger beroep" terwijl bezwaar bedoeld wordt) | Materieel labelen, niet op woordkeuze |
| 11 | Heel kort bericht ("graag bellen") | Meestal `vraag/overig` met `borderline-kort` |
| 12 | Mededeling van een kind/jeugdige | Standaard labelen — meerderjarigheid is geen labelcriterium |
| 13 | Document met verzoek én bedreiging/scheldwoorden | Standaard labelen op materieel verzoek; aparte vlag `bedreiging` voor escalatieroute (out-of-scope voor classifier maar markeren) |
| 14 | "Ik wil mijn aanvraag intrekken" | Nieuw materieel kenmerk: intrekking. Voor v0.1: label als `vraag/overig` met opmerking `intrekking` (mogelijk eigen label in v0.2) |

---

## 5. Labelingsprotocol

### 5.1 Rolverdeling

- **Labeller A** — jurist of Awb-deskundige (kennis van procesrecht en grensgevallen)
- **Labeller B** — generalist met KCC-perspectief (kennis van hoe burgers daadwerkelijk schrijven)
- **Reviewer** — derde stem bij disagreement; is bij voorkeur senior bestuursjurist

### 5.2 Werkwijze per voorbeeld

1. Beide labellers zien hetzelfde voorbeeld onafhankelijk
2. Elke labeller kent toe: één of meer labels uit §1, plus confidence (`hoog/middel/laag`), plus vrije opmerking
3. Bij `laag`-confidence verplicht: één regel toelichting waar het twijfelt
4. Resultaten worden geconsolideerd in één spreadsheet/Label Studio-project

### 5.3 Disagreement-resolutie

- Volledige overeenstemming op alle labels → opnemen in dataset
- Verschil op één label → reviewer bekijkt en neemt finale beslissing, motivatie wordt opgenomen
- Verschil op meerdere labels → terug naar discussie, eventueel guidelines aanpassen (zie §7)

### 5.4 Kwaliteitsmetrieken

- **Cohen's κ** per labelpaar, target ≥ 0.75 voor opname in dataset
- Onder 0.75: guidelines herzien, eerste batch opnieuw labelen
- **Per-label F1** tussen labellers — labels onder 0.70 krijgen extra aandacht in §3 (beslisboom-uitbreiding)

### 5.5 Tooling

Voorlopig **spreadsheet** (Google Sheets / Excel) voor de eerste 100 voorbeelden om de guidelines te valideren. Daarna migreren naar **Label Studio** voor de hoofdronde — multilabel-support, audit-trail, en directe export naar HF datasets.

### 5.6 Velden per voorbeeld

| Veld | Type | Verplicht |
|---|---|---|
| `id` | string (uuid) | ja |
| `text` | string | ja |
| `source` | enum (rechtspraak/raadsinformatie/woo/ombudsman/synthetic) | ja |
| `source_url` | string | ja indien echt |
| `labels_A` | list[label] | ja |
| `confidence_A` | enum (hoog/middel/laag) | ja |
| `notes_A` | string | bij `laag` verplicht |
| `labels_B` | list[label] | ja |
| `confidence_B` | enum | ja |
| `notes_B` | string | bij `laag` verplicht |
| `final_labels` | list[label] | ja na resolutie |
| `borderline_flags` | list[flag] | optioneel — zie §4 voor flag-types |
| `language` | iso-code | default `nl` |
| `pseudonymized` | bool | ja |

---

## 6. Voorbeeld-annotaties

Vijf gelabelde voorbeelden als ankerpunt. Reviewers gebruiken deze als basis voor inter-rater alignment.

### Voorbeeld 1
**Tekst:** "Geachte heer/mevrouw, ik heb op 3 maart een aanvraag ingediend voor huishoudelijke hulp. Vandaag kreeg ik een afwijzingsbrief. Ik ben het hier niet mee eens want ik kan echt niet meer zelf stofzuigen na mijn heupoperatie. Graag opnieuw beoordelen. Mvg, J."
**Labels:** `bezwaar`
**Confidence:** hoog
**Reden:** richt zich tegen specifiek besluit (afwijzing), vraagt heroverweging.

### Voorbeeld 2
**Tekst:** "Hallo, ik ben al 4 keer doorverbonden vandaag en niemand kan me helpen met mijn vraag over de parkeervergunning. Dit is echt schandalig. Mag ik weten wie hier verantwoordelijk voor is?"
**Labels:** `klacht`
**Confidence:** hoog
**Reden:** gericht op gedrag (proces, bejegening), geen besluit-referentie.

### Voorbeeld 3
**Tekst:** "Beste gemeente, hierbij verzoek ik om openbaarmaking van alle e-mailcorrespondentie tussen het college en projectontwikkelaar X over project Y in de periode januari–maart 2024."
**Labels:** `WOO-verzoek`
**Confidence:** hoog
**Reden:** verzoek om specifieke overheidsdocumenten, geen eigen dossier.

### Voorbeeld 4
**Tekst:** "Ik wil reageren op het ontwerp-bestemmingsplan dat ter inzage ligt voor de Korenstraat. Mijn bezwaren tegen het plan zijn: (1) verkeerstoename, (2) geluidsoverlast, (3) waardedaling woning."
**Labels:** `zienswijze`
**Confidence:** hoog
**Reden:** reactie op ontwerp ter inzage. Hoewel burger het woord "bezwaren" gebruikt, is het ontwerp nog niet definitief — daarom zienswijze, niet bezwaar.

### Voorbeeld 5
**Tekst:** "Geachte gemeente, hierbij wil ik klagen over de afwijzing van mijn bijstandsaanvraag. Ik begrijp niet waarom dit is afgewezen, en bovendien werd ik door uw medewerker zeer onbeschoft te woord gestaan tijdens het gesprek vorige week. Graag spoedige reactie."
**Labels:** `bezwaar` + `klacht`
**Confidence:** hoog
**Reden:** woord 'klagen' is niet leidend; document bevat zowel ageren tegen afwijzing (= bezwaar) als klacht over bejegening (= klacht). Multilabel.

---

## 7. Versionering en wijzigingen

- Dit document is genummerd. Wijzigingen worden gelogd onderaan.
- Tijdens de eerste **100 voorbeelden** is het document open voor revisie. Daarna gefroren tot v0.2.
- **Wie wijzigt:** alleen na consensus tussen beide labellers + reviewer.
- **Wat triggert revisie:** systematisch lage Cohen's κ (<0.75) op een specifiek label, of meer dan drie keer dezelfde grensgeval-disagreement.

### Changelog
- 2026-05-07 — v0.1 initieel.
