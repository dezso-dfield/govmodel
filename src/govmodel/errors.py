"""Typed exception hierarchy for govmodel.

Distinct types let callers (Gradio, FastAPI service, tests) decide on
retry/abort/alert behaviour without grepping error strings.
"""
from __future__ import annotations


class GovmodelError(Exception):
    """Base class for govmodel-specific failures."""


class LLMResponseError(GovmodelError):
    """LLM server returned a malformed or HTTP-error response."""


class ModelLoadError(GovmodelError):
    """Failed to load the classifier model or tokenizer."""


class InsufficientConfidenceError(GovmodelError):
    """No label crosses its threshold — caller should route to a human."""
