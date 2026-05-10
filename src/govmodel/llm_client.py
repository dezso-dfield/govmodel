"""OpenAI-compatible client voor lokale LLM-server (LM Studio, Ollama, vLLM, ...).

Configurable via env vars (preferred over hardcoding a LAN IP):
    LLM_BASE_URL      e.g. http://localhost:1234/v1   (default if unset)
    LLM_MODEL         e.g. qwen/qwen3-4b-2507
    LLM_API_KEY       optional bearer token
    LLM_TIMEOUT_S     request timeout in seconds (default 180)

Usage:
    with LLMClient.from_env() as client:
        text = client.chat([{"role": "user", "content": "Hallo"}])
"""

from __future__ import annotations

import logging
import os
from types import TracebackType

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from govmodel.errors import LLMResponseError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "qwen/qwen3-4b-2507"
DEFAULT_TIMEOUT_S = 180.0
MAX_MESSAGE_CHARS = 32_000  # rough guard against runaway prompts


class LLMClient:
    """Minimale OpenAI-compatible chat completion client.

    Werkt met elke OpenAI-API-compatibele server: LM Studio, Ollama (via
    /v1-endpoint), vLLM, llama.cpp server, etc.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=timeout_s, headers=headers)

    @classmethod
    def from_env(cls) -> LLMClient:
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            timeout_s=float(os.environ.get("LLM_TIMEOUT_S", DEFAULT_TIMEOUT_S)),
            api_key=os.environ.get("LLM_API_KEY") or None,
        )

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=20),
        retry=retry_if_exception_type(httpx.RequestError),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 500,
        **extra: object,
    ) -> str:
        """Stuur een chat-completion request, retourneer de eerste choice-tekst.

        Raises:
            LLMResponseError: when the server returns a malformed response or
                an HTTP error. Distinct from network errors which `tenacity`
                retries automatically.
        """
        _validate_messages(messages)

        body: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body.update(extra)
        response = self._client.post(f"{self.base_url}/chat/completions", json=body)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMResponseError(
                f"LLM server returned {response.status_code}: {response.text[:200]}"
            ) from e

        try:
            data = response.json()
        except ValueError as e:
            raise LLMResponseError(f"LLM response was not JSON: {response.text[:200]}") from e

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMResponseError(
                f"LLM response missing choices[0].message.content: {data!r}"
            ) from e

        if not isinstance(content, str):
            raise LLMResponseError(
                f"LLM response content is {type(content).__name__}, expected str"
            )
        return content


def _validate_messages(messages: list[dict[str, str]]) -> None:
    if not isinstance(messages, list) or not messages:
        raise LLMResponseError("messages must be a non-empty list")
    total_chars = 0
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise LLMResponseError(f"messages[{i}] missing role/content")
        if m["role"] not in {"system", "user", "assistant", "tool"}:
            raise LLMResponseError(f"messages[{i}] role={m['role']!r} not allowed")
        if not isinstance(m["content"], str):
            raise LLMResponseError(f"messages[{i}].content must be str")
        total_chars += len(m["content"])
    if total_chars > MAX_MESSAGE_CHARS:
        raise LLMResponseError(
            f"messages total {total_chars} chars exceeds limit {MAX_MESSAGE_CHARS}"
        )
