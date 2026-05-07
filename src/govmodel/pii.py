"""PII-pseudonimisering voor Nederlandstalige correspondentie.

Detecteert en vervangt persoonsidentificerende patronen via regex + validatie:
- BSN (met 11-proef)
- IBAN (NL, met mod-97-validatie)
- E-mailadressen
- Telefoonnummers (NL-formats)
- Postcodes (4 cijfers + 2 letters, optioneel met huisnummer)

Voor *namen, locaties en organisaties* is regex onvoldoende; voeg in v0.2 een
spaCy of fine-tuned PII-NER toe. Voor v0.1 leunen we op:
1. Reeds gepseudonimiseerde input (Rechtspraak doet dit al; synthetic gebruikt
   placeholders zoals [Jouw naam])
2. Deze regex-laag als extra veiligheidsnet voor gestructureerde PII
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class Redaction:
    pattern_name: str
    original: str
    placeholder: str
    start: int
    end: int


class _Pattern(NamedTuple):
    name: str
    regex: re.Pattern[str]
    placeholder: str


# Email — RFC 5322 simplified
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# IBAN-NL: NL kk BBBB 0000 0000 00 (with optional spaces)
IBAN_NL_RE = re.compile(
    r"\bNL\d{2}\s?[A-Z]{4}(?:\s?\d{4}){2}\s?\d{2}\b",
    re.IGNORECASE,
)

# Telefoonnummers NL — alleen werkelijk uitgegeven kengetallen, om false
# positives op zaaknummers/dossier-IDs te beperken.
# Mobiel: 06; Vast: 010, 013, 015, 020, 023-026, 030, 033, 035-038, 040,
# 043-046, 050, 053, 055, 058, 070-079; Service: 0800, 0900, 084, 087, 088.
# Lookbehind/-ahead op \w in plaats van \b: \b werkt niet voor een patroon
# dat met '+' begint.
PHONE_RE = re.compile(
    r"(?<!\w)(?:"
    # internationaal +31 of 0031, gevolgd door geldig kengetal
    r"(?:\+31|0031)\s?-?\s?6\s?-?\s?\d{8}"
    r"|(?:\+31|0031)\s?-?\s?(?:1[035]|2[0-6]|3[035-8]|4[03-6]|5[035-8]|7[0-9]|8[478])\s?-?\s?\d{6,7}"
    # nationaal mobiel
    r"|06\s?-?\s?\d{8}"
    # nationaal vast
    r"|0(?:1[035]|2[0-6]|3[035-8]|4[03-6]|5[035-8]|7[0-9]|8[478])\s?-?\s?\d{6,7}"
    # service
    r"|0(?:800|900)\s?-?\s?\d{4,7}"
    r")(?!\w)"
)

# Postcode: 1234AB of 1234 AB, optioneel met huisnummer erachter
POSTCODE_RE = re.compile(
    r"\b(\d{4})\s?([A-Z]{2})(\s+\d{1,5}[a-zA-Z]?)?\b"
)

# BSN: 9 cijfers (nog te valideren met 11-proef in een tweede stap)
BSN_RE = re.compile(r"\b\d{9}\b")


def is_valid_bsn(digits: str) -> bool:
    """11-proef voor BSN. Niet alle 9-cijferige getallen zijn BSN's."""
    if len(digits) != 9 or not digits.isdigit():
        return False
    weights = (9, 8, 7, 6, 5, 4, 3, 2, -1)
    s = sum(int(d) * w for d, w in zip(digits, weights, strict=True))
    return s % 11 == 0


_IBAN_LETTER_TO_DIGIT = {chr(ord("A") + i): str(10 + i) for i in range(26)}


def is_valid_iban_nl(iban: str) -> bool:
    """Mod-97 validatie voor IBAN."""
    cleaned = re.sub(r"\s+", "", iban).upper()
    if not re.fullmatch(r"NL\d{2}[A-Z]{4}\d{10}", cleaned):
        return False
    rearranged = cleaned[4:] + cleaned[:4]
    numeric = "".join(_IBAN_LETTER_TO_DIGIT.get(c, c) for c in rearranged)
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


def _find_emails(text: str) -> list[Redaction]:
    return [
        Redaction("EMAIL", m.group(0), "[EMAIL]", m.start(), m.end())
        for m in EMAIL_RE.finditer(text)
    ]


def _find_ibans(text: str) -> list[Redaction]:
    out: list[Redaction] = []
    for m in IBAN_NL_RE.finditer(text):
        if is_valid_iban_nl(m.group(0)):
            out.append(Redaction("IBAN", m.group(0), "[IBAN]", m.start(), m.end()))
    return out


def _find_phones(text: str) -> list[Redaction]:
    return [
        Redaction("TELEFOON", m.group(0), "[TELEFOON]", m.start(), m.end())
        for m in PHONE_RE.finditer(text)
    ]


def _find_postcodes(text: str) -> list[Redaction]:
    out: list[Redaction] = []
    for m in POSTCODE_RE.finditer(text):
        out.append(Redaction("POSTCODE", m.group(0), "[POSTCODE]", m.start(), m.end()))
    return out


def _find_bsns(text: str, exclude_ranges: list[tuple[int, int]]) -> list[Redaction]:
    """BSN match alleen als (a) 11-proef geldig en (b) niet binnen al gevonden range
    (bv. midden in een telefoonnummer of IBAN)."""
    out: list[Redaction] = []
    for m in BSN_RE.finditer(text):
        if any(s <= m.start() < e for s, e in exclude_ranges):
            continue
        if is_valid_bsn(m.group(0)):
            out.append(Redaction("BSN", m.group(0), "[BSN]", m.start(), m.end()))
    return out


def find_pii(text: str) -> list[Redaction]:
    """Vind alle PII-redacties in volgorde van start-offset."""
    redactions: list[Redaction] = []
    redactions.extend(_find_emails(text))
    redactions.extend(_find_ibans(text))
    redactions.extend(_find_phones(text))
    redactions.extend(_find_postcodes(text))

    # BSN als laatste, met exclusion van eerdere ranges (telefoon kan 9 cijfers bevatten)
    excluded = [(r.start, r.end) for r in redactions]
    redactions.extend(_find_bsns(text, excluded))

    redactions.sort(key=lambda r: (r.start, -r.end))
    return _resolve_overlaps(redactions)


def _resolve_overlaps(redactions: list[Redaction]) -> list[Redaction]:
    """Bij overlap: behoud de langste/eerste, gooi de overlappende weg."""
    out: list[Redaction] = []
    last_end = -1
    for r in redactions:
        if r.start >= last_end:
            out.append(r)
            last_end = r.end
    return out


def pseudonymize(text: str) -> tuple[str, list[Redaction]]:
    """Vervang gevonden PII door placeholders. Geeft (geredacteerde tekst, redacties).

    De originele waarden in de Redactions-lijst zijn alleen bedoeld voor audit;
    publiceer ze nooit samen met de geredacteerde dataset.
    """
    redactions = find_pii(text)
    if not redactions:
        return text, []

    # Vervang van achteren naar voren zodat indices niet schuiven
    pieces: list[str] = []
    cursor = len(text)
    for r in reversed(redactions):
        pieces.append(text[r.end:cursor])
        pieces.append(r.placeholder)
        cursor = r.start
    pieces.append(text[:cursor])
    redacted = "".join(reversed(pieces))
    return redacted, redactions
