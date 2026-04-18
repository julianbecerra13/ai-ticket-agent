"""Proveedor Anthropic Claude."""

from __future__ import annotations

import logging

from anthropic import Anthropic, APIError

from src.agent.providers.base import LLMProvider, LLMProviderError

log = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        self._client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, *, system: str, user: str) -> str:
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.2,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except APIError as exc:
            log.error("Anthropic API error: %s", exc)
            raise LLMProviderError(str(exc)) from exc

        parts = [block.text for block in message.content if getattr(block, "type", "") == "text"]
        return "".join(parts).strip()
