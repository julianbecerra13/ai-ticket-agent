"""Tests de los proveedores LLM (foco en MockProvider y factory)."""

from __future__ import annotations

import json

import pytest
from src.agent.providers.factory import build_provider
from src.agent.providers.mock_provider import MockProvider
from src.config import LLMProviderName, Settings
from src.db.models import AgentActionType


@pytest.fixture
def mock() -> MockProvider:
    return MockProvider()


def _user_prompt(category: str, urgency: str) -> str:
    return (
        "TICKET\n------\nUsuario: u1\nAsunto: x\nCuerpo: y\n\n"
        f"CLASIFICACION ML\n----------------\nCategoria: {category} (confianza 0.90)\n"
        f"Urgencia: {urgency} (confianza 0.90)\n\nHISTORICO RECIENTE DEL USUARIO\n------------------------------\n"
    )


def test_mock_escala_en_urgencia_critica(mock: MockProvider) -> None:
    raw = mock.generate(system="", user=_user_prompt("tecnico", "critica"))
    data = json.loads(raw)
    assert data["action"] == AgentActionType.ESCALATE.value


def test_mock_auto_respond_en_caso_sencillo(mock: MockProvider) -> None:
    raw = mock.generate(system="", user=_user_prompt("cuenta", "baja"))
    data = json.loads(raw)
    assert data["action"] == AgentActionType.AUTO_RESPOND.value
    assert data["response_text"]


def test_factory_devuelve_none_sin_config() -> None:
    s = Settings(
        anthropic_api_key=None,
        openai_api_key=None,
        ollama_host=None,
        llm_provider=None,
    )
    assert build_provider(s) is None


def test_factory_detecta_anthropic_por_key() -> None:
    s = Settings(
        anthropic_api_key="sk-ant-fake",
        openai_api_key=None,
        ollama_host=None,
        llm_provider=None,
    )
    provider = build_provider(s)
    assert provider is not None
    assert provider.name == LLMProviderName.ANTHROPIC.value


def test_factory_respeta_override() -> None:
    s = Settings(
        anthropic_api_key="sk-ant-fake",
        openai_api_key="sk-fake",
        ollama_host=None,
        llm_provider=LLMProviderName.OPENAI,
    )
    provider = build_provider(s)
    assert provider is not None
    assert provider.name == LLMProviderName.OPENAI.value
