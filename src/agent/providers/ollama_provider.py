"""Proveedor Ollama local.

Llama al endpoint HTTP `/api/chat` de Ollama. No necesita API key. Util para
demos locales o entornos sin acceso a LLMs comerciales.
"""

from __future__ import annotations

import logging

import httpx

from src.agent.providers.base import LLMProvider, LLMProviderError

log = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 60.0) -> None:
        self._host = host.rstrip("/")
        self.model = model
        self._timeout = timeout

    def generate(self, *, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = httpx.post(
                f"{self._host}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.error("Ollama HTTP error: %s", exc)
            raise LLMProviderError(str(exc)) from exc

        data = response.json()
        return str(data.get("message", {}).get("content", "")).strip()
