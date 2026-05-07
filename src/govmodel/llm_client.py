"""OpenAI-compatible client voor lokale LLM-server (LM Studio, Ollama, vLLM, ...).

Gebruik:
    with LLMClient() as client:
        text = client.chat([{"role": "user", "content": "Hallo"}])
"""

from __future__ import annotations

import logging
from types import TracebackType

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://192.168.56.1:1234/v1"
DEFAULT_MODEL = "qwen/qwen3-4b-2507"
DEFAULT_TIMEOUT_S = 180.0


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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 500,
        **extra: object,
    ) -> str:
        """Stuur een chat-completion request, retourneer de eerste choice-tekst."""
        body: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body.update(extra)
        response = self._client.post(f"{self.base_url}/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
