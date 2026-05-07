from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

AwbLabel = Literal[
    "aanvraag",
    "bezwaar",
    "beroep",
    "klacht",
    "melding",
    "zienswijze",
    "woo_verzoek",
    "informatieverzoek_3_11",
    "vraag_overig",
]


class RechtspraakUitspraak(BaseModel):
    """Een uitspraak van de Open Data Rechtspraak (data.rechtspraak.nl).

    Velden volgen de RDF/dcterms-metadata van de rechtspraak-1.0 schema.
    """

    ecli: str
    content_url: str
    instance: str | None = None
    decision_date: date | None = None
    subject: str | None = None
    case_number: str | None = None
    summary: str | None = None
    full_text: str | None = None
    awb_articles_cited: list[str] = Field(default_factory=list)
    woo_cited: bool = False
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LabeledExample(BaseModel):
    """Een voorbeeld in de uiteindelijke trainings/eval-set.

    Gemeenschappelijk schema voor alle bronnen (rechtspraak, raadsinformatie,
    WOO, ombudsman, synthetic).
    """

    id: str
    text: str
    source: Literal[
        "rechtspraak", "raadsinformatie", "woo", "ombudsman", "synthetic", "gold"
    ]
    source_url: str | None = None
    labels: list[AwbLabel] = Field(default_factory=list)
    confidence: Literal["hoog", "middel", "laag"] | None = None
    notes: str | None = None
    borderline_flags: list[str] = Field(default_factory=list)
    language: str = "nl"
    pseudonymized: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
