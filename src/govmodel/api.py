"""FastAPI service for govmodel.

Endpoints:
    GET  /                          → analytics dashboard (HTML)
    GET  /api/health                → liveness + model status
    POST /api/classify              → {"text": ..., "threshold": 0.5} → Prediction
    POST /api/classify/{rid}/feedback → {"user_label": ...} for active-learning
    GET  /api/analytics/summary     → analytics payload (JSON, used by dashboard)
    GET  /api/predictions           → recent N JSONL rows (for inspectors)

Run:
    uvicorn govmodel.api:app --host 127.0.0.1 --port 8001 --reload

Env vars:
    GOVMODEL_MODEL_ID           HF model id or local path (default v0.1)
    GOVMODEL_PREDICTION_LOG     JSONL log path (default logs/predictions.jsonl)
    GOVMODEL_ABSTAIN_THRESHOLD  abstain if top_score < this (default 0.20)
    GOVMODEL_API_TOKEN          require Bearer token for POST /api/classify*
    GOVMODEL_CORS_ORIGINS       comma-separated origins (default empty = same-origin only)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from govmodel import analytics
from govmodel.calibration import ece_from_log
from govmodel.errors import GovmodelError, ModelLoadError
from govmodel.inference import (
    DEFAULT_MODEL_ID,
    DEFAULT_THRESHOLD,
    Classifier,
    PredictionLogger,
)
from govmodel.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger("govmodel.api")

app = FastAPI(
    title="govmodel API",
    version="0.2.0-dev",
    description="Awb-typering classifier with abstain mechanism + analytics dashboard.",
)

_origins = [o.strip() for o in os.environ.get("GOVMODEL_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

DASHBOARD_HTML = Path(__file__).parent / "dashboard" / "index.html"

_classifier: Classifier | None = None
_prediction_logger = PredictionLogger()


def get_classifier() -> Classifier:
    global _classifier
    if _classifier is None:
        try:
            _classifier = Classifier()
        except ModelLoadError as e:
            logger.error("model load failed", extra={"err": str(e)})
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"model not available: {e}",
            ) from e
    return _classifier


def require_token(request: Request) -> None:
    expected = os.environ.get("GOVMODEL_API_TOKEN")
    if not expected:
        return  # auth disabled
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != expected:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.05, le=0.95)
    request_id: str | None = None


class FeedbackRequest(BaseModel):
    user_label: str = Field(..., min_length=1, max_length=100)


@app.exception_handler(GovmodelError)
async def _govmodel_error_handler(_: Request, exc: GovmodelError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": type(exc).__name__, "detail": str(exc)})


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    if not DASHBOARD_HTML.exists():
        return HTMLResponse("<h1>govmodel</h1><p>Dashboard niet gevonden.</p>", status_code=200)
    return HTMLResponse(DASHBOARD_HTML.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": _classifier is not None,
        "model_id": DEFAULT_MODEL_ID,
        "abstain_threshold": float(os.environ.get("GOVMODEL_ABSTAIN_THRESHOLD", "0.20")),
        "auth_required": bool(os.environ.get("GOVMODEL_API_TOKEN")),
    }


@app.post("/api/classify", dependencies=[Depends(require_token)])
async def classify(req: ClassifyRequest) -> dict[str, Any]:
    clf = get_classifier()
    pred = clf.classify(req.text, threshold=req.threshold, request_id=req.request_id)
    _prediction_logger.log(pred, raw_text=req.text)
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
    return pred.to_public_dict()


@app.post("/api/classify/{request_id}/feedback", dependencies=[Depends(require_token)])
async def feedback(request_id: str, req: FeedbackRequest) -> dict[str, Any]:
    """Append a user-correction row so the analytics dashboard can show
    agreement and confusion. Doesn't modify earlier rows; the latest entry
    for a request_id wins when summarising."""
    record = {
        "ts": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "request_id": request_id,
        "user_label": req.user_label,
        "feedback_only": True,
    }
    _prediction_logger.path.parent.mkdir(parents=True, exist_ok=True)
    with _prediction_logger.path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("feedback received", extra={"request_id": request_id, "user_label": req.user_label})
    return {"ok": True}


@app.get("/api/analytics/summary")
async def analytics_summary(window_hours: float = 24.0) -> dict[str, Any]:
    return analytics.summary(window_hours=window_hours)


@app.get("/api/analytics/calibration")
async def analytics_calibration() -> dict[str, Any]:
    """ECE + reliability curve over (top_score, top_label == user_label)
    pairs in the prediction log. Empty until feedback rows arrive."""
    return ece_from_log(_prediction_logger.path)


@app.get("/api/predictions", response_class=PlainTextResponse)
async def recent_predictions(limit: int = 100) -> str:
    """Return the last N rows of the JSONL log as raw JSONL (for piping)."""
    path = _prediction_logger.path
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-limit:])
