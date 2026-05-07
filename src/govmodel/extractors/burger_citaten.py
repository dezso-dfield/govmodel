"""Extract directe burger-citaten uit Rechtspraak-uitspraken.

Strategie: zoek quoted text in de uitspraak-tekst die wordt voorafgegaan door
een actor+verb-patroon dat aangeeft dat dit een citaat van de burger is.

Voorbeeld in tekst:
    "Eiseres heeft in haar bezwaarschrift het volgende aangevoerd:
    'Ik ben het niet eens met de hoogte van het schadebedrag.'"

→ extracted: 'Ik ben het niet eens met de hoogte van het schadebedrag.'

Rechters paraphraseren meestal, maar bij betwiste woordkeuze citeren ze direct.
Dat zijn de momenten waarop we échte burgertaal vinden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Aanduiding van burger in juridische taal
ACTORS = (
    r"(?:appellant(?:e|en)?|eiser(?:es|s)?|verzoeker(?:s)?|verzoekster|"
    r"klager(?:s)?|klaagster|indiener(?:s)?|betrokkene|belanghebbende|"
    r"bezwaarmaker|bezwaarmaakster)"
)

# Werkwoorden of formuleringen die introduceren wat de burger schreef/zei.
# Bewust losjes — kan op zichzelf staan (`schrijft`) of als deelwoord
# (`...heeft ... geschreven`). We checken actor en verb apart binnen het
# context-window, niet in één strakke regex.
INTRO_VERBS = (
    r"(?:schrijft|stelt|voert\s+aan|verzoekt|meldt|"
    r"geschreven|aangevoerd|gesteld|verzocht|gemeld|geantwoord|"
    r"opgemerkt|naar\s+voren\s+gebracht|geklaagd|verklaard|"
    r"het\s+volgende|als\s+volgt|onder\s+meer)"
)

# Quote-tekens die in NL-juridische teksten voorkomen
QUOTE_OPEN = r"[\"‘“„»‹]"
QUOTE_CLOSE = r"[\"’”«›]"

# Hoofdpatroon: <quote> 50-1500 tekens met no-quote inside
QUOTE_PATTERN = re.compile(
    rf"{QUOTE_OPEN}([^\"’”«›]{{50,1500}}?){QUOTE_CLOSE}",
    re.DOTALL,
)

# Actor en verb worden los gezocht in het context-window (300 chars vóór quote).
# Strakke combo-regex faalde te vaak omdat 'heeft ... geschreven' veel woorden
# tussen kan hebben.
ACTOR_RE = re.compile(rf"\b{ACTORS}\b", re.IGNORECASE)
VERB_RE = re.compile(rf"\b{INTRO_VERBS}\b", re.IGNORECASE)


def has_burger_context(context_window: str) -> bool:
    """Burger-citaat-context: zowel een actor als een intro-verb in het venster."""
    return bool(ACTOR_RE.search(context_window)) and bool(VERB_RE.search(context_window))

# Heuristieken voor uitsluiting van rechter-language binnen 'citaat'
RECHTER_LANGUAGE_INSIDE_QUOTE = re.compile(
    r"\b(?:appellant|eiseres|de\s+rechtbank|de\s+afdeling|college|verweerder|"
    r"is\s+van\s+oordeel|naar\s+het\s+oordeel|overweegt|conclude(?:ert|ren))\b",
    re.IGNORECASE,
)

# Awb-artikel detectie binnen context — voor labelvoorstel
AWB_TO_LABEL = {
    "1:3": "aanvraag",
    "4:1": "aanvraag",
    "6:4": "bezwaar",
    "6:5": "bezwaar",
    "7:1": "bezwaar",
    "9:1": "klacht",
    "9:8": "klacht",
    "3:11": "informatieverzoek_3_11",
    "3:15": "zienswijze",
    "4:8": "zienswijze",
}
AWB_REF_RE = re.compile(r"\b(?:art(?:ikel|\.?)?\s*)(\d+):(\d+)", re.IGNORECASE)
WOO_REF_RE = re.compile(r"\b(?:Wet\s+open\s+overheid|Woo\b)", re.IGNORECASE)


@dataclass
class CitationCandidate:
    quote: str
    preceding_context: str
    quote_start_in_text: int
    suggested_labels: list[str]
    suggestion_reason: str
    quality_score: float


def suggest_labels(context: str) -> tuple[list[str], str]:
    """Stel mogelijke Awb-labels voor op basis van Awb/WOO-citaten in de
    omringende rechter-tekst. Niet definitief — labeller verifieert."""
    suggestions: list[str] = []
    reasons: list[str] = []

    for m in AWB_REF_RE.finditer(context):
        art = f"{m.group(1)}:{m.group(2)}"
        if art in AWB_TO_LABEL:
            label = AWB_TO_LABEL[art]
            if label not in suggestions:
                suggestions.append(label)
                reasons.append(f"art {art}")

    if WOO_REF_RE.search(context):
        if "woo_verzoek" not in suggestions:
            suggestions.append("woo_verzoek")
            reasons.append("Woo-referentie")

    # Aanvullende heuristieken
    ctx_lower = context.lower()
    if "bezwaarschrift" in ctx_lower and "bezwaar" not in suggestions:
        suggestions.append("bezwaar")
        reasons.append("woord 'bezwaarschrift'")
    if "klachtschrift" in ctx_lower and "klacht" not in suggestions:
        suggestions.append("klacht")
        reasons.append("woord 'klachtschrift'")
    if "zienswijze" in ctx_lower and "zienswijze" not in suggestions:
        suggestions.append("zienswijze")
        reasons.append("woord 'zienswijze'")

    return suggestions, "; ".join(reasons) if reasons else "geen specifiek signaal"


def quality_score(quote: str, context: str) -> float:
    """Heuristische kwaliteitsscore 0..1 voor 'is dit echt burgertaal'.

    Hoger = waarschijnlijker direct burger-citaat. Niet betrouwbaar — vooral
    voor sortering, labeller verifieert handmatig.
    """
    score = 0.5

    # Negatief: rechter-jargon binnen 'citaat'
    rechter_hits = len(RECHTER_LANGUAGE_INSIDE_QUOTE.findall(quote))
    score -= 0.15 * rechter_hits

    # Positief: eerste-persoon ('ik', 'mijn', 'wij')
    first_person_hits = len(re.findall(r"\b(?:ik|mijn|mij|me|wij|ons|onze)\b", quote, re.IGNORECASE))
    score += min(0.3, 0.05 * first_person_hits)

    # Positief: lengte richting 100-500 (sweet spot voor brief-fragment)
    n = len(quote)
    if 100 <= n <= 500:
        score += 0.2
    elif 80 <= n < 100 or 500 < n <= 800:
        score += 0.1

    # Positief: brief-achtige openings/sluitingen
    if re.search(r"\b(?:geachte|geacht\s+college|hierbij|met\s+vriendelijke\s+groet|hoogachtend)\b",
                 quote, re.IGNORECASE):
        score += 0.15

    # Negatief: lijkt op wettekst-citaat
    if re.search(r"\bartikel\s+\d+", quote, re.IGNORECASE) and len(quote) < 250:
        score -= 0.1

    return max(0.0, min(1.0, score))


def extract_citations(text: str) -> list[CitationCandidate]:
    """Vind alle waarschijnlijke burger-citaten in een uitspraak-tekst."""
    if not text:
        return []

    candidates: list[CitationCandidate] = []
    for m in QUOTE_PATTERN.finditer(text):
        quote = m.group(1).strip()

        # Filters
        if len(quote) < 50:
            continue

        # Context vóór de quote (max 300 chars)
        ctx_start = max(0, m.start() - 300)
        context_before = text[ctx_start:m.start()]

        # Moet zowel actor als intro-verb hebben in het context-venster
        if not has_burger_context(context_before):
            continue

        # Wijdere context voor labelsuggestie (300 voor + 200 na)
        wider_start = max(0, m.start() - 300)
        wider_end = min(len(text), m.end() + 200)
        wider_context = text[wider_start:wider_end]

        suggestions, reason = suggest_labels(wider_context)
        score = quality_score(quote, wider_context)

        candidates.append(CitationCandidate(
            quote=quote,
            preceding_context=context_before.strip()[-250:],
            quote_start_in_text=m.start(),
            suggested_labels=suggestions,
            suggestion_reason=reason,
            quality_score=score,
        ))

    return candidates
