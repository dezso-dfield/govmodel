"""Meta-klacht synthetic templates (R&D item G2 from plan.md).

The v0.1 classifier scored macro-F1 0.44 on real `klacht` examples — way
below the 0.89 it scored on synthetic-realistic. A big chunk of that gap
is **meta-complaints**: complaints about how a previous complaint was
handled. The synthetic set Qwen generated never contains them, so the
model never learned the pattern.

This module produces prompt templates the next synthetic-generation run
can feed to `LLMClient` to fill that gap. It does NOT call the LLM here;
generation is a separate script step (see `scripts/generate_synthetic.py`).
Keeping prompts here (and unit-tested) means the next person who wants
to extend coverage can reuse / adapt the templates without re-inventing
prompt structure.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

# 8 meta-complaint scenarios drawn from real Rechtspraak themes.
# Each entry's `theme` describes the original grievance; the LLM is
# prompted to write a complaint *about how* that earlier complaint was
# handled, not about the underlying issue.
_SCENARIOS: tuple[tuple[str, str], ...] = (
    (
        "trage_afhandeling",
        "Een eerdere klacht over een vergunningaanvraag is na 12 weken nog niet beantwoord.",
    ),
    (
        "geen_reactie",
        "Een eerdere klacht over geluidsoverlast werd zes maanden geleden ingediend, "
        "nooit een ontvangstbevestiging gekregen.",
    ),
    (
        "ontwijkend_antwoord",
        "De gemeente heeft de klacht beantwoord met algemene tekst zonder op de inhoud "
        "in te gaan.",
    ),
    (
        "verkeerde_persoon",
        "De klacht over een ambtenaar werd doorgestuurd naar dezelfde ambtenaar.",
    ),
    (
        "verkeerde_afdeling",
        "Een klacht over de afdeling Burgerzaken werd herhaaldelijk doorgeschoven "
        "tussen afdelingen.",
    ),
    (
        "geen_excuses",
        "De klacht is afgewezen zonder enige erkenning van wat misging.",
    ),
    (
        "niet_in_klachtenregister",
        "De klacht is wel in behandeling genomen maar niet geregistreerd, dus geen "
        "formele afhandeling.",
    ),
    (
        "klachtprocedure_verkort",
        "De gemeente sluit een klachtprocedure af zonder de hoorzitting waar de "
        "burger om vroeg.",
    ),
)

_STYLES: tuple[str, ...] = (
    "formal_legalistic",
    "polite_frustrated",
    "angry_emotional",
    "matter_of_fact",
    "NT2_B1_simplified",
)


@dataclass(frozen=True)
class MetaKlachtPromptSpec:
    scenario_key: str
    style: str
    expected_labels: list[str]
    prompt: str


_BASE_SYSTEM_PROMPT = (
    "Je bent een hulp die realistische Nederlandse burgerbrieven aan een "
    "gemeente schrijft. Je schrijft alleen de brief — geen toelichting, "
    "geen disclaimers, geen markdown. Geen verzonnen PII (geen BSN, IBAN, "
    "echte telefoonnummers, echte adressen of namen)."
)


def _user_prompt(scenario_text: str, style: str) -> str:
    style_hint = {
        "formal_legalistic": (
            "Schrijf in formele, juridisch-aanvoelende stijl. Gebruik "
            "verwijzingen naar 'klachtenregeling', 'beklag' en 'Nationale "
            "ombudsman'."
        ),
        "polite_frustrated": (
            "Schrijf beleefd maar duidelijk gefrustreerd. De toon is van "
            "iemand die nog steeds geduld probeert te bewaren."
        ),
        "angry_emotional": (
            "Schrijf direct en boos. De burger gebruikt soms hoofdletters "
            "voor nadruk. Uitroepen en pittige zinnen zijn welkom."
        ),
        "matter_of_fact": (
            "Schrijf droog en zakelijk, alsof het een memo is. Korte zinnen, "
            "geen emotie, alleen feiten."
        ),
        "NT2_B1_simplified": (
            "Schrijf in eenvoudig Nederlands (taalniveau B1). Korte zinnen, "
            "basale woorden, vermijd formele connectoren."
        ),
    }[style]
    return (
        f"Schrijf een **meta-klacht**: een brief van een burger aan zijn "
        f"gemeente waarin de burger klaagt over de manier waarop een "
        f"eerdere klacht is afgehandeld. De oorspronkelijke klacht ging "
        f"over: '{scenario_text}'. {style_hint}\n\n"
        f"Belangrijk: de brief moet expliciet verwijzen naar de eerdere "
        f"klacht (datum, kenmerk fictief). De brief is een klacht — "
        f"GEEN bezwaar, GEEN aanvraag — over de **klachtafhandeling**. "
        f"Schrijf 4–8 alinea's.\n\nBrief:"
    )


def iter_prompts(*, examples_per_combo: int = 2) -> Iterator[MetaKlachtPromptSpec]:
    """Yields one prompt per (scenario × style × n_per_combo) combination.

    8 scenarios × 5 styles × `examples_per_combo` examples → that's the
    target meta-klacht corpus. With `examples_per_combo=3` you get 120
    rows, which is in the 50–100 range plan.md calls for in G2.
    """
    for scenario_key, scenario_text in _SCENARIOS:
        for style in _STYLES:
            for _ in range(examples_per_combo):
                yield MetaKlachtPromptSpec(
                    scenario_key=scenario_key,
                    style=style,
                    expected_labels=["klacht"],
                    prompt=_user_prompt(scenario_text, style),
                )


def system_prompt() -> str:
    return _BASE_SYSTEM_PROMPT


def scenarios() -> tuple[tuple[str, str], ...]:
    return _SCENARIOS


def styles() -> tuple[str, ...]:
    return _STYLES
