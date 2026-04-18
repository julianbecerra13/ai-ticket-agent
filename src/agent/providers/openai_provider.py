"""Proveedor OpenAI."""

from __future__ import annotations

import logging

from openai import APIError, OpenAI

from src.agent.providers.base import LLMProvider, LLMProviderError

log = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, *, system: str, user: str) -> str:
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except APIError as exc:
            log.error("OpenAI API error: %s", exc)
            raise LLMProviderError(str(exc)) from exc

        content = completion.choices[0].message.content or ""
        return content.strip()
