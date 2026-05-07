"""Synthetic data generation voor de Awb-type-classifier.

Per label: definitie + 2 anker-voorbeelden + constraints.
Per style: een korte beschrijving die in de system prompt belandt.

De generator combineert (label × style × n) en levert LabeledExample-objecten.
"""

from __future__ import annotations

import logging
import random
import re
import uuid
from collections.abc import Iterator
from typing import TypedDict

from govmodel.llm_client import LLMClient
from govmodel.schemas import AwbLabel, LabeledExample

logger = logging.getLogger(__name__)


class LabelPromptData(TypedDict):
    definition: str
    examples: list[str]
    constraints: list[str]


# Per Awb-label de definitie, anker-voorbeelden en constraints voor de generator.
LABEL_PROMPTS: dict[AwbLabel, LabelPromptData] = {
    "aanvraag": {
        "definition": (
            "Een aanvraag is een verzoek van een burger aan de gemeente om een nieuwe "
            "beschikking te nemen — bijvoorbeeld een vergunning, WMO-voorziening, "
            "subsidie, paspoort, urgentieverklaring, kwijtschelding, of leerlingenvervoer."
        ),
        "examples": [
            "Geachte gemeente, hierbij vraag ik een gehandicaptenparkeerkaart aan. "
            "Sinds mijn knieoperatie in januari kan ik niet meer dan 50 meter lopen "
            "en heb ik moeite om mijn boodschappen tot de auto te krijgen. Bijgevoegd "
            "zijn de medische verklaring en een kopie van mijn ID. Ik hoor graag "
            "wanneer ik een reactie kan verwachten.",
            "Hallo, ik wil graag huishoudelijke hulp via de WMO aanvragen. Mijn "
            "moeder van 84 woont nog zelfstandig maar het stofzuigen en dweilen lukt "
            "niet meer. Kunt u mij vertellen hoe ik dit kan regelen?",
        ],
        "constraints": [
            "Vraag impliciet of expliciet om een nieuwe beschikking",
            "Verwijs NIET naar een eerder besluit van de gemeente",
            "Geen klachten over gedrag of behandeling",
        ],
    },
    "bezwaar": {
        "definition": (
            "Een bezwaar is een schriftelijke reactie waarin de burger zich verzet "
            "tegen een eerder genomen besluit van de gemeente. De burger noemt het "
            "besluit (afwijzing, korting, intrekking, etc.) en vraagt om heroverweging."
        ),
        "examples": [
            "Geachte heer/mevrouw, ik heb op 3 maart een aanvraag ingediend voor "
            "huishoudelijke hulp. Vandaag kreeg ik een afwijzingsbrief. Ik ben het "
            "hier niet mee eens want ik kan echt niet meer zelf stofzuigen na mijn "
            "heupoperatie. Graag opnieuw beoordelen.",
            "Hierbij teken ik bezwaar aan tegen uw besluit van 12 februari met "
            "kenmerk 2024-1234, waarbij mijn omgevingsvergunning is geweigerd. De "
            "gronden voor mijn bezwaar zijn dat de welstandscommissie onvoldoende "
            "heeft gemotiveerd waarom mijn dakkapel niet zou passen in het straatbeeld.",
        ],
        "constraints": [
            "VERWIJS expliciet naar een eerder besluit van de gemeente (afwijzing, korting, weigering, intrekking)",
            "Vraag om heroverweging — niet om een nieuwe aanvraag",
            "Geen aparte zorgvraag of verzoek dat losstaat van het bestreden besluit",
        ],
    },
    "beroep": {
        "definition": (
            "Een (administratief) beroep is een beroep ingesteld bij een ander "
            "bestuursorgaan dan dat het oorspronkelijke besluit nam — bijvoorbeeld "
            "beroep bij de provincie tegen een gemeentelijk besluit."
        ),
        "examples": [
            "Geacht college van Gedeputeerde Staten, hierbij stel ik administratief "
            "beroep in tegen het besluit van het college van burgemeester en wethouders "
            "van de gemeente X d.d. 5 januari, waarbij mijn aanvraag is afgewezen. "
            "De motivering van het bestreden besluit is naar mijn oordeel onvoldoende.",
        ],
        "constraints": [
            "Richt je tot een ander bestuursorgaan (provincie, ministerie) dan de gemeente die het besluit nam",
            "Verwijs naar het bestreden besluit en de oorspronkelijke besluitvormer",
        ],
    },
    "klacht": {
        "definition": (
            "Een klacht gaat over GEDRAG van het bestuursorgaan of een medewerker — "
            "bejegening, lange behandeling, slechte communicatie, niet teruggebeld "
            "worden, fouten in uitvoering. NIET over de uitkomst van een besluit."
        ),
        "examples": [
            "Hallo, ik ben al 4 keer doorverbonden vandaag en niemand kan me helpen "
            "met mijn vraag over de parkeervergunning. Dit is echt schandalig. Mag "
            "ik weten wie hier verantwoordelijk voor is?",
            "Beste gemeente, ik wil een klacht indienen over de behandelend ambtenaar "
            "in mijn dossier. Drie weken geleden zou ik teruggebeld worden. Inmiddels "
            "heb ik al twee keer gemaild zonder antwoord. Dit kan toch niet?",
        ],
        "constraints": [
            "Klacht gaat over GEDRAG of PROCES (bejegening, traagheid, communicatie)",
            "GEEN klacht over de inhoudelijke uitkomst van een besluit (dat is bezwaar)",
            "Geen nieuwe aanvraag of verzoek om besluit",
        ],
    },
    "melding": {
        "definition": (
            "Een melding is een mededeling aan de gemeente waarbij geen besluit "
            "wordt gevraagd — vaak operationeel of informerend, zoals een melding "
            "leerplichtverzuim, ingebruikname-melding, sloopmelding, of melding "
            "ondernemer. NIET MOR (meldingen openbare ruimte buiten — die gaan via Signalen)."
        ),
        "examples": [
            "Hierbij meld ik dat wij vanaf 1 september ons horecapand aan de Kerkstraat 12 "
            "in gebruik nemen. De openingstijden zijn conform de APV. Bijgevoegd het "
            "ondernemingsplan en een kopie van het brandveiligheidscertificaat.",
            "Beste gemeente, ik meld hierbij dat mijn zoon (12 jaar) deze week niet "
            "naar school is geweest wegens ziekte. Bijgaand een verklaring van de huisarts.",
        ],
        "constraints": [
            "Mededeling, geen verzoek om besluit",
            "Geen melding over kapot wegdek/zwerfafval/openbare-ruimte (dat is MOR)",
            "Geen klacht of bezwaar",
        ],
    },
    "zienswijze": {
        "definition": (
            "Een zienswijze is een reactie van een belanghebbende op een ontwerp-besluit "
            "dat ter inzage ligt (Awb 3:15) of op een voornemen tot belastend besluit "
            "(Awb 4:8). Het besluit is dus nog NIET definitief."
        ),
        "examples": [
            "Hierbij dien ik mijn zienswijze in op het ter inzage gelegde ontwerp-"
            "omgevingsplan voor de Korenstraat. Mijn bezwaren tegen het plan zijn: "
            "(1) verkeerstoename, (2) geluidsoverlast, (3) waardedaling woning. "
            "Ik vraag u deze punten in de definitieve besluitvorming mee te wegen.",
            "Geachte college, ik heb kennisgenomen van uw voornemen tot intrekking van "
            "mijn subsidie. Hierbij dien ik mijn zienswijze in: de aangevoerde gronden "
            "zijn naar mijn oordeel feitelijk onjuist, en wel om de volgende redenen...",
        ],
        "constraints": [
            "Reactie op een ONTWERP-besluit of VOORNEMEN — niet op een definitief besluit",
            "Geen verzoek om nieuwe beschikking",
            "Geen klacht over bejegening",
        ],
    },
    "woo_verzoek": {
        "definition": (
            "Een WOO-verzoek (Wet open overheid art. 4.1) is een verzoek om "
            "openbaarmaking van publieke documenten/informatie. De aanvrager hoeft "
            "geen belanghebbende te zijn. Het gaat om OVERHEIDSDOCUMENTEN, niet om "
            "het eigen dossier van de aanvrager."
        ),
        "examples": [
            "Beste gemeente, hierbij verzoek ik op grond van de Wet open overheid om "
            "openbaarmaking van alle e-mailcorrespondentie tussen het college en "
            "projectontwikkelaar X over project Y in de periode januari–maart 2024.",
            "Geachte gemeente, ik wil graag op grond van de Woo inzage in alle "
            "documenten betreffende de besluitvorming rond de aanleg van de rotonde "
            "op het kruispunt Hoofdstraat/Kerkweg.",
        ],
        "constraints": [
            "Verzoek om OVERHEIDSDOCUMENTEN of -informatie",
            "Niet om eigen dossier van aanvrager",
            "Verwijzing naar Wet open overheid, Woo, of openbaarmaking",
        ],
    },
    "informatieverzoek_3_11": {
        "definition": (
            "Een informatieverzoek op grond van Awb 3:11 is een verzoek om inzage "
            "in stukken die ter inzage zijn gelegd in het kader van een lopende "
            "uitgebreide voorbereidingsprocedure (afdeling 3.4 Awb)."
        ),
        "examples": [
            "Geachte gemeente, ik wil graag de onderliggende stukken inzien bij het "
            "ontwerp-bestemmingsplan dat momenteel ter inzage ligt voor de wijk "
            "Hoogkamp. Met name de verkeersrapporten en de geluidsstudie.",
        ],
        "constraints": [
            "Verzoek om inzage in stukken die ter inzage zijn gelegd",
            "Gekoppeld aan een lopende uitgebreide voorbereidingsprocedure",
            "Geen algemeen WOO-verzoek",
        ],
    },
    "vraag_overig": {
        "definition": (
            "Reguliere correspondentie zonder Awb-procedure-trigger: openingstijden, "
            "afspraken, algemene informatievragen, bedankbrieven, of suggesties zonder "
            "besluit-context."
        ),
        "examples": [
            "Hallo, kunt u mij vertellen wanneer het loket Burgerzaken op zaterdag "
            "open is? Ik wil graag mijn paspoort komen ophalen.",
            "Beste gemeente, ik wilde u graag bedanken voor de snelle hulp van "
            "Marijke vorige week bij mijn verhuizing. Fijn dat ze zo meedacht.",
        ],
        "constraints": [
            "Geen verzoek om besluit (anders is het een aanvraag)",
            "Geen verwijzing naar eerder besluit (anders is het bezwaar)",
            "Geen klacht over gedrag (anders is het klacht)",
            "Geen verzoek om openbaarmaking van documenten (anders is het WOO)",
        ],
    },
}


# Stijl-profielen — bepalen toon, taalniveau, lengte, en spelvariatie.
STYLE_PROFILES: dict[str, str] = {
    "formal_short": (
        "formele toon, B2/C1-niveau, ongeveer 40-80 woorden, foutloos Nederlands, "
        "zakelijke aanhef en afsluiting"
    ),
    "formal_long": (
        "formele toon, C1-niveau, 200-350 woorden, juridisch correcte formuleringen, "
        "duidelijke argumentatie in genummerde of opgesomde punten"
    ),
    "informal_short": (
        "informele toon, B1-niveau, 30-60 woorden, spreektaal-Nederlands, "
        "directe aanspreekvorm, mag een spelfoutje bevatten"
    ),
    "informal_medium": (
        "informele toon, B1-niveau, 80-150 woorden, spreektaal, persoonlijk verhaal, "
        "korte zinnen, mag spelfouten bevatten"
    ),
    "angry_medium": (
        "boze, soms emotionele toon, 100-200 woorden, B1/B2-niveau, mag spelfouten "
        "en uitroeptekens bevatten, frustratie is voelbaar"
    ),
    "polite_questioning": (
        "vriendelijk vragend, beleefd, 50-100 woorden, B2-niveau, geen verwijten, "
        "open vragen aan het eind"
    ),
    "nt2_simple": (
        "Nederlands van iemand die het nog leert (NT2-niveau A2/B1), zeer korte zinnen, "
        "50-100 woorden, mag duidelijke grammatica- en spelfouten bevatten"
    ),
    "elderly_handwritten_style": (
        "stijl van een oudere burger die formeel schrijft maar wat ouderwets, "
        "100-200 woorden, B2-niveau, gebruikt wat archaïsche formuleringen, "
        "geen typefouten maar wel lange zinnen"
    ),
    "legal_dense": (
        "stijl van een advocaat of jurist die namens een cliënt schrijft, "
        "150-300 woorden, juridisch jargon, expliciete verwijzingen naar "
        "wetsartikelen waar passend (bv. art. 6:5 Awb), formele aanhef "
        "('Namens cliënt...'), geen spelfouten"
    ),
    "rambling_long": (
        "uitvoerige, soms uitweidende stijl, 250-450 woorden, B1/B2-niveau, "
        "vermeldt persoonlijke achtergrond en context naast het hoofdpunt, "
        "incidentele zijwegen ('zoals u zich misschien herinnert...'), "
        "wel grammaticaal correct"
    ),
    "chaotic_typing": (
        "snel getypt, weinig hoofdletters, geen interpunctie of slordige "
        "interpunctie, 60-130 woorden, duidelijk laaggeletterd of in haast, "
        "veel typefouten en spelfouten, mag woorden samentrekken ('mn' voor "
        "'mijn', 'kdoe' voor 'ik doe')"
    ),
}


# Multilabel-prompt: voor specifieke combinaties zoals bezwaar+klacht.
MULTILABEL_COMBOS: list[tuple[tuple[AwbLabel, ...], str]] = [
    (
        ("bezwaar", "klacht"),
        "Schrijf een brief die ZOWEL bezwaar maakt tegen een eerder besluit "
        "(bv. afwijzing of weigering) ALS klaagt over hoe de gemeente dit heeft "
        "behandeld (bv. trage reactie, slechte communicatie, foute brief). "
        "Beide aspecten moeten duidelijk aanwezig zijn.",
    ),
    (
        ("aanvraag", "klacht"),
        "Schrijf een brief waarin de burger een NIEUWE aanvraag indient (bv. "
        "een vergunning, voorziening, subsidie) EN tegelijk klaagt over een "
        "eerdere ervaring met de gemeente (bv. trage afhandeling, onbeleefde "
        "medewerker). Beide aspecten duidelijk.",
    ),
    (
        ("aanvraag", "vraag_overig"),
        "Schrijf een brief waarin de burger zowel een aanvraag indient als "
        "een aanvullende informatievraag stelt (bv. 'wanneer hoor ik iets, "
        "en wat moet ik nog aanleveren?')",
    ),
    (
        ("zienswijze", "vraag_overig"),
        "Schrijf een brief waarin de burger een zienswijze indient op een "
        "ontwerp-besluit en daarnaast vraagt om aanvullende informatie of "
        "een gesprek over het traject.",
    ),
]


def build_multilabel_messages(
    labels: tuple[AwbLabel, ...], style: str, instruction: str
) -> list[dict[str, str]]:
    """Bouw chat-messages voor een multilabel-combinatie."""
    style_desc = STYLE_PROFILES[style]
    label_definitions = "\n".join(
        f"- **{lbl}**: {LABEL_PROMPTS[lbl]['definition']}" for lbl in labels
    )

    system = (
        "Je bent een Nederlandse burger die schrijft aan de gemeente. "
        f"Schrijfstijl: {style_desc}. "
        "Schrijf ALLEEN de brieftekst, geen meta-uitleg, geen toelichting, geen markdown."
    )

    user = (
        f"Definities van de relevante typen:\n{label_definitions}\n\n"
        f"Opdracht: {instruction}\n\n"
        "Verzin een realistisch onderwerp en schrijf het bericht volgens de opgegeven schrijfstijl."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_multilabel_examples(
    client: LLMClient,
    n_per_combo: int = 4,
    styles: list[str] | None = None,
    temperature: float = 0.85,
    max_tokens: int = 800,
    rng: random.Random | None = None,
) -> Iterator[LabeledExample]:
    """Genereer voorbeelden voor multilabel-combinaties."""
    rng = rng or random.Random()
    styles = styles or ["formal_long", "informal_medium", "angry_medium", "rambling_long"]

    for labels, instruction in MULTILABEL_COMBOS:
        for style in styles:
            for _ in range(n_per_combo):
                messages = build_multilabel_messages(labels, style, instruction)
                try:
                    raw = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Multilabel generation failed (%s/%s): %s", labels, style, e)
                    continue
                text = clean_generated_text(raw)
                if not is_valid_generation(text):
                    continue
                yield LabeledExample(
                    id=str(uuid.uuid4()),
                    text=text,
                    source="synthetic",
                    labels=list(labels),
                    confidence=None,
                    notes=f"multilabel; style={style}; model={client.model}",
                    language="nl",
                    pseudonymized=True,
                )


def build_messages(
    label: AwbLabel, style: str, seed_example: str
) -> list[dict[str, str]]:
    """Bouw de chat-messages voor één generation-aanroep."""
    label_data = LABEL_PROMPTS[label]
    style_desc = STYLE_PROFILES[style]
    constraints_str = "\n- ".join(label_data["constraints"])

    system = (
        "Je bent een Nederlandse burger die schrijft aan de gemeente. "
        f"Schrijfstijl: {style_desc}. "
        "Schrijf ALLEEN de brieftekst, geen meta-uitleg, geen toelichting, geen markdown."
    )

    user = (
        f"Schrijf een bericht aan de gemeente van het type: {label}.\n\n"
        f"Definitie: {label_data['definition']}\n\n"
        f"Voorbeeld van zo'n bericht:\n\"\"\"\n{seed_example}\n\"\"\"\n\n"
        f"Eisen voor jouw nieuwe bericht:\n- {constraints_str}\n"
        f"- Verzin een ander onderwerp en andere details dan in het voorbeeld\n"
        f"- Houd je strikt aan de opgegeven schrijfstijl\n\n"
        "Schrijf nu het bericht:"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def clean_generated_text(text: str) -> str:
    """Lichte opschoning van model-output."""
    text = text.strip()
    # Verwijder markdown code fences
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text)
    # Verwijder omringende quotes
    text = text.strip().strip('"').strip("'").strip()
    # Verwijder typische meta-prefixen
    for prefix in ("Brief:", "Bericht:", "Tekst:", "Hier is mijn bericht:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_valid_generation(text: str, min_chars: int = 30, max_chars: int = 4000) -> bool:
    """Basisvalidatie op gegenereerde tekst."""
    if not text:
        return False
    if len(text) < min_chars or len(text) > max_chars:
        return False
    # Filter outputs die alleen meta-uitleg blijken te zijn
    if text.lower().startswith(("ik kan", "als ai", "natuurlijk", "zeker,")):
        return False
    return True


def generate_examples(
    client: LLMClient,
    label: AwbLabel,
    style: str,
    n: int = 3,
    temperature: float = 0.85,
    max_tokens: int = 600,
    rng: random.Random | None = None,
) -> Iterator[LabeledExample]:
    """Genereer N voorbeelden voor één (label, style) combinatie."""
    if label not in LABEL_PROMPTS:
        raise ValueError(f"Onbekend label: {label}")
    if style not in STYLE_PROFILES:
        raise ValueError(f"Onbekende stijl: {style}")

    seed_examples = LABEL_PROMPTS[label]["examples"]
    rng = rng or random.Random()

    for _ in range(n):
        seed = rng.choice(seed_examples)
        messages = build_messages(label, style, seed)
        try:
            raw = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            logger.warning("Generation failed (%s/%s): %s", label, style, e)
            continue

        text = clean_generated_text(raw)
        if not is_valid_generation(text):
            logger.debug("Filtered invalid generation: %r", text[:80])
            continue

        yield LabeledExample(
            id=str(uuid.uuid4()),
            text=text,
            source="synthetic",
            labels=[label],
            confidence=None,
            notes=f"style={style}; model={client.model}",
            language="nl",
            pseudonymized=True,
        )
