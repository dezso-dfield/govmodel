"""Tests for the hardened LLMClient.

Focus on the parts that don't need a running LLM server: input validation,
JSON-error handling, and env-var configuration.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from govmodel.errors import LLMResponseError
from govmodel.llm_client import LLMClient, _validate_messages


def test_validate_messages_accepts_well_formed():
    _validate_messages([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("bad", [
    [],
    "not a list",
    [{"role": "user"}],                          # missing content
    [{"content": "x"}],                          # missing role
    [{"role": "alien", "content": "x"}],         # invalid role
    [{"role": "user", "content": 42}],           # non-string content
])
def test_validate_messages_rejects_malformed(bad):
    with pytest.raises(LLMResponseError):
        _validate_messages(bad)  # type: ignore[arg-type]


def test_validate_messages_rejects_oversize():
    with pytest.raises(LLMResponseError, match="exceeds limit"):
        _validate_messages([{"role": "user", "content": "x" * 33_000}])


def test_from_env_uses_env_vars(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://example.test:9999/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_S", "5")
    with LLMClient.from_env() as client:
        assert client.base_url == "http://example.test:9999/v1"
        assert client.model == "test-model"


def _make_response(*, status_code: int, body: object) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=body if isinstance(body, (bytes, str)) else json.dumps(body),
        request=httpx.Request("POST", "http://example/v1/chat/completions"),
    )


def test_chat_raises_on_http_error():
    response = _make_response(status_code=500, body=b"server down")
    with LLMClient(base_url="http://example") as client, \
         patch.object(client._client, "post", return_value=response), \
         pytest.raises(LLMResponseError, match="500"):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_on_non_json_response():
    response = _make_response(status_code=200, body=b"<html>500</html>")
    with LLMClient(base_url="http://example") as client, \
         patch.object(client._client, "post", return_value=response), \
         pytest.raises(LLMResponseError, match="not JSON"):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_on_missing_choices():
    response = _make_response(status_code=200, body={"unexpected": "shape"})
    with LLMClient(base_url="http://example") as client, \
         patch.object(client._client, "post", return_value=response), \
         pytest.raises(LLMResponseError, match="missing choices"):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_returns_content_on_happy_path():
    body = {"choices": [{"message": {"content": "hallo wereld"}}]}
    response = _make_response(status_code=200, body=body)
    with LLMClient(base_url="http://example") as client, \
         patch.object(client._client, "post", return_value=response):
        assert client.chat([{"role": "user", "content": "hi"}]) == "hallo wereld"
