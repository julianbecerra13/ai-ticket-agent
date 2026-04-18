"""Tests del orquestador TicketAgent."""

from __future__ import annotations

from src.agent.agent import TicketAgent
from src.agent.providers.base import LLMProvider, LLMProviderError
from src.db.models import AgentActionType, Ticket, TicketCategory, TicketUrgency
from src.ml.classifier import PredictionResult


class _StaticProvider(LLMProvider):
    name = "static"
    model = "static-v1"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self.response


class _FlakyProvider(LLMProvider):
    name = "flaky"
    model = "flaky-v1"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, system: str, user: str) -> str:
        self.calls += 1
        if self.calls == 1:
            raise LLMProviderError("rate limit")
        return '{"action": "auto_respond", "reasoning": "ok", "response_text": "hola"}'


def _ticket() -> Ticket:
    return Ticket(id=1, user_id="u", subject="test", body="cuerpo del ticket")


def _prediction() -> PredictionResult:
    return PredictionResult(
        category=TicketCategory.CUENTA,
        urgency=TicketUrgency.BAJA,
        confidence_category=0.9,
        confidence_urgency=0.8,
    )


def test_agente_devuelve_decision_valida() -> None:
    provider = _StaticProvider(
        '{"action": "auto_respond", "reasoning": "ok", "response_text": "hola"}'
    )
    decision = TicketAgent(provider).decide(ticket=_ticket(), prediction=_prediction())
    assert decision.action == AgentActionType.AUTO_RESPOND
    assert decision.response_text == "hola"
    assert provider.calls == 1


def test_agente_reintenta_con_json_invalido() -> None:
    provider = _StaticProvider("no es json valido")
    decision = TicketAgent(provider).decide(ticket=_ticket(), prediction=_prediction())
    assert decision.action == AgentActionType.ESCALATE
    assert provider.calls == 2
    assert "[fallback]" in decision.reasoning


def test_agente_acepta_json_envuelto_en_markdown() -> None:
    provider = _StaticProvider('```json\n{"action": "escalate", "reasoning": "caso sensible"}\n```')
    decision = TicketAgent(provider).decide(ticket=_ticket(), prediction=_prediction())
    assert decision.action == AgentActionType.ESCALATE
    assert decision.reasoning == "caso sensible"


def test_agente_reintenta_tras_error_de_proveedor() -> None:
    provider = _FlakyProvider()
    decision = TicketAgent(provider).decide(ticket=_ticket(), prediction=_prediction())
    assert decision.action == AgentActionType.AUTO_RESPOND
    assert provider.calls == 2


def test_agente_fallback_si_proveedor_siempre_falla() -> None:
    class _AlwaysFails(LLMProvider):
        name = "bad"
        model = "bad-v1"

        def generate(self, *, system: str, user: str) -> str:
            raise LLMProviderError("offline")

    decision = TicketAgent(_AlwaysFails()).decide(ticket=_ticket(), prediction=_prediction())
    assert decision.action == AgentActionType.ESCALATE
    assert "[fallback]" in decision.reasoning
