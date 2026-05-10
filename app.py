"""Gradio demo voor govmodel — Awb-typering classifier.

Voor lokaal draaien:
    pip install gradio transformers torch
    python app.py

Voor HuggingFace Spaces:
    Plaats dit bestand + requirements.txt in een Space-repo.
    HF Spaces detecteert Gradio automatisch.

Logging (standaard naar logs/predictions.jsonl):
    GOVMODEL_PREDICTION_LOG=path/to/log.jsonl  # override pad
    GOVMODEL_LOG_RAW_TEXT=1                    # ook ruwe tekst loggen (privacy!)
    GOVMODEL_ABSTAIN_THRESHOLD=0.20            # abstain als top-score < deze
"""

from __future__ import annotations

import logging
import os

import gradio as gr

from govmodel.inference import (
    ABSTAIN_THRESHOLD,
    DEFAULT_MODEL_ID,
    Classifier,
    PredictionLogger,
)
from govmodel.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger("govmodel.app")

LABEL_DESCRIPTIONS = {
    "aanvraag": "Verzoek om beschikking (vergunning, voorziening, subsidie)",
    "bezwaar": "Verzet tegen eerder besluit van de gemeente",
    "beroep": "Administratief beroep bij ander bestuursorgaan",
    "klacht": "Klacht over gedrag van bestuursorgaan/medewerker",
    "melding": "Mededeling zonder besluit-vraag (verhuizing, sloop, leerplicht)",
    "zienswijze": "Reactie op ontwerp-besluit of voornemen",
    "woo_verzoek": "Verzoek om openbaarmaking documenten (Wet open overheid)",
    "informatieverzoek_3_11": "Inzage stukken in lopende uitgebreide voorbereidingsprocedure",
    "vraag_overig": "Reguliere correspondentie zonder Awb-procedure",
}

EXAMPLES = [
    ["Hierbij teken ik bezwaar aan tegen uw besluit van 12 maart 2024 (kenmerk WMO-2024-0567) waarbij mijn aanvraag voor een traplift is afgewezen. De gronden voor mijn bezwaar zijn dat ik op 78-jarige leeftijd al 42 jaar in deze woning woon en een verhuizing voor mij onacceptabel is."],
    ["Beste gemeente, op grond van de Wet open overheid verzoek ik om openbaarmaking van alle besluiten en bijbehorende stukken van het college over de aanbesteding van het zwembad De Plons in 2024."],
    ["Hallo, ik ben al 5 keer doorverbonden vandaag en niemand kan me helpen met mijn vraag over de parkeervergunning. Dit is echt schandalig. Mag ik weten wie hier verantwoordelijk voor is?"],
    ["Geachte gemeente, hierbij vraag ik een omgevingsvergunning aan voor het plaatsen van een dakkapel aan de achterzijde van mijn woning. Bijgaand de bouwtekening en de welstandsnota."],
    ["Hierbij dien ik mijn zienswijze in op het ter inzage gelegde ontwerp-bestemmingsplan 'De Hoek'. Mijn bezwaren: (1) verkeerstoename, (2) geluidsoverlast, (3) waardedaling woning."],
]

logger.info("starting app", extra={"model_id": DEFAULT_MODEL_ID})
classifier = Classifier()
prediction_logger = PredictionLogger()


def classify(text: str, threshold: float = 0.5) -> tuple[dict, str]:
    if not text or not text.strip():
        return {}, "_(geen invoer)_"

    pred = classifier.classify(text, threshold=threshold)
    prediction_logger.log(pred, raw_text=text)
    logger.info(
        "classified",
        extra={
            "request_id": pred.request_id,
            "top_label": pred.top_label,
            "top_score": round(pred.top_score, 3),
            "abstain": pred.abstain,
            "latency_ms": round(pred.latency_ms, 1),
        },
    )

    multilabel_results = pred.all_scores

    explanation_lines = []
    if pred.abstain:
        explanation_lines.append(
            f"⚠️ **Onvoldoende vertrouwen** — top-score `{pred.top_score:.0%}` "
            f"ligt onder de abstain-drempel `{ABSTAIN_THRESHOLD:.0%}`."
        )
        explanation_lines.append("Aanbeveling: handmatige triage door medewerker.")
        explanation_lines.append("")

    explanation_lines.append(f"**Top single-label:** `{pred.top_label}` ({pred.top_score:.0%})")
    explanation_lines.append(f"_{LABEL_DESCRIPTIONS.get(pred.top_label, '')}_")
    explanation_lines.append("")
    if pred.above_threshold:
        n = len(pred.above_threshold)
        explanation_lines.append(
            f"**Boven threshold {threshold}** ({n} label{'s' if n != 1 else ''}):"
        )
        for lbl, p in pred.above_threshold:
            explanation_lines.append(f"- `{lbl}` — {p:.0%}")
    elif not pred.abstain:
        explanation_lines.append(
            f"_Geen labels boven threshold {threshold}. Top suggestie: "
            f"`{pred.top_label}` ({pred.top_score:.0%})._"
        )

    explanation_lines.append("")
    explanation_lines.append(f"_request_id: `{pred.request_id}` · "
                             f"latency: {pred.latency_ms:.0f} ms_")
    explanation_lines.append("")
    explanation_lines.append("---")
    explanation_lines.append(
        "⚠️ Deze classifier is een **research baseline**. Output dient als "
        "advies, niet als besluit. Mens in the loop verplicht."
    )

    return multilabel_results, "\n".join(explanation_lines)


with gr.Blocks(title="govmodel — Awb-typering classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # govmodel — Awb-typering classifier

        Open-source NL classifier voor inkomende gemeentepost. Plak een burgerbrief / e-mail
        en het model voorspelt onder welk Awb-type het valt. Multilabel: één bericht
        kan tegelijk bezwaar én klacht zijn.

        **Status:** v0.1 research baseline · **Licentie:** EUPL-1.2 · **Contact:** info@noaberai.nl
        """
    )

    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(
                label="Burgerbrief / e-mail",
                placeholder="Plak hier de tekst van een burger aan een gemeente...",
                lines=10,
            )
            threshold = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                label="Multilabel threshold",
                info="Lager = meer labels gerapporteerd; hoger = alleen sterke signalen",
            )
            run_btn = gr.Button("Classificeer", variant="primary")

        with gr.Column():
            scores = gr.Label(label="Awb-type voorspellingen", num_top_classes=9)
            explanation = gr.Markdown(label="Toelichting")

    gr.Examples(
        examples=EXAMPLES,
        inputs=text_in,
        label="Voorbeelden (klik om te proberen)",
    )

    run_btn.click(classify, inputs=[text_in, threshold], outputs=[scores, explanation])
    text_in.submit(classify, inputs=[text_in, threshold], outputs=[scores, explanation])

    gr.Markdown(
        f"""
        ---
        ### Bekende beperkingen
        - **Klacht-detectie** werkt voor reguliere klachten; juridisch-complexe en meta-klachten zijn een v0.2-prioriteit.
        - **6 labels** (aanvraag, beroep, melding, zienswijze, informatieverzoek_3_11, vraag_overig)
          zijn nog **niet getoetst op echte burgerteksten** — alleen op handgeschreven en synthetic eval-sets.
        - Niet bedoeld voor automatische besluitvorming (AI Act *limited risk*).
        - **Abstain-mechanisme:** als geen label boven {ABSTAIN_THRESHOLD:.0%} komt,
          adviseert het model handmatige triage in plaats van een gok.

        Volledige model card: [`MODEL_CARD.md`](https://huggingface.co/NoaberAI/govmodel-awb-classifier-v0.1/blob/main/README.md)
        """
    )


if __name__ == "__main__":
    demo.launch()
