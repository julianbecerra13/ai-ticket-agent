"""Fabrica que devuelve el proveedor LLM configurado, o None si no hay ninguno."""

from __future__ import annotations

import logging

from src.agent.providers.anthropic_provider import AnthropicProvider
from src.agent.providers.base import LLMProvider
from src.agent.providers.mock_provider import MockProvider
from src.agent.providers.ollama_provider import OllamaProvider
from src.agent.providers.openai_provider import OpenAIProvider
from src.config import LLMProviderName, Settings, get_settings

log = logging.getLogger(__name__)


def build_provider(settings: Settings | None = None) -> LLMProvider | None:
    s = settings or get_settings()
    resolved = s.resolve_llm_provider()

    if resolved is None:
        log.info("No hay proveedor LLM configurado. El agente quedara desactivado.")
        return None

    match resolved:
        case LLMProviderName.ANTHROPIC:
            assert s.anthropic_api_key
            log.info("LLM activo: Anthropic (%s)", s.anthropic_model)
            return AnthropicProvider(api_key=s.anthropic_api_key, model=s.anthropic_model)
        case LLMProviderName.OPENAI:
            assert s.openai_api_key
            log.info("LLM activo: OpenAI (%s)", s.openai_model)
            return OpenAIProvider(api_key=s.openai_api_key, model=s.openai_model)
        case LLMProviderName.OLLAMA:
            assert s.ollama_host
            log.info("LLM activo: Ollama (%s @ %s)", s.ollama_model, s.ollama_host)
            return OllamaProvider(host=s.ollama_host, model=s.ollama_model)


def build_mock_provider() -> LLMProvider:
    return MockProvider()
