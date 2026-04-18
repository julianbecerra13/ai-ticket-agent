"""Orquestador del agente.

Recibe un ticket + su prediccion + el historico, arma el prompt, llama al
proveedor LLM y parsea la respuesta. Si el LLM devuelve un JSON invalido,
reintenta una vez. Si vuelve a fallar, se devuelve una decision de
`escalate` por defecto con el motivo registrado.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from src.agent.prompts import SYSTEM_PROMPT, render_user_prompt
from src.agent.providers.base import LLMProvider, LLMProviderError
from src.db.models import AgentActionType, Ticket, TicketCategory, TicketUrgency
from src.ml.classifier import PredictionResult

log = logging.getLogger(__name__)


class _RawDecision(BaseModel):
    action: AgentActionType
    reasoning: str = Field(min_length=1)
    response_text: str | None = None


@dataclass
class AgentDecisionResult:
    action: AgentActionType
    reasoning: str
    response_text: str | None
    provider_name: str
    provider_model: str


class TicketAgent:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def decide(
        self,
        *,
        ticket: Ticket,
        prediction: PredictionResult,
        recent_history: list[Ticket] | None = None,
    ) -> AgentDecisionResult:
        history_text = _format_history(recent_history or [])
        user_prompt = render_user_prompt(
            user_id=ticket.user_id,
            subject=ticket.subject,
            body=ticket.body,
            category=prediction.category.value,
            urgency=prediction.urgency.value,
            confidence_category=prediction.confidence_category,
            confidence_urgency=prediction.confidence_urgency,
            recent_history=history_text,
        )

        for attempt in range(2):
            try:
                raw = self._provider.generate(system=SYSTEM_PROMPT, user=user_prompt)
            except LLMProviderError as exc:
                log.warning("Proveedor LLM fallo (intento %s): %s", attempt + 1, exc)
                if attempt == 0:
                    continue
                return self._fallback(f"Error del proveedor LLM: {exc}")

            parsed = _parse(raw)
            if parsed is not None:
                return AgentDecisionResult(
                    action=parsed.action,
                    reasoning=parsed.reasoning,
                    response_text=parsed.response_text,
                    provider_name=self._provider.name,
                    provider_model=self._provider.model,
                )
            log.warning(
                "JSON invalido del LLM (intento %s). Respuesta: %s",
                attempt + 1,
                raw[:200] if raw else "<vacia>",
            )

        return self._fallback("LLM devolvio JSON invalido tras dos intentos.")

    def _fallback(self, reason: str) -> AgentDecisionResult:
        return AgentDecisionResult(
            action=AgentActionType.ESCALATE,
            reasoning=f"[fallback] {reason}",
            response_text=None,
            provider_name=self._provider.name,
            provider_model=self._provider.model,
        )


def _parse(raw: str) -> _RawDecision | None:
    if not raw:
        return None
    snippet = _strip_markdown(raw)
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    try:
        return _RawDecision.model_validate(data)
    except ValidationError:
        return None


def _strip_markdown(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()


def _format_history(history: list[Ticket]) -> str:
    if not history:
        return ""
    lines = []
    for t in history[:5]:
        lines.append(f"- [{t.created_at:%Y-%m-%d}] {t.subject}")
    return "\n".join(lines)


def _coerce_category(value: str) -> TicketCategory:
    try:
        return TicketCategory(value)
    except ValueError:
        return TicketCategory.INFORMACION


def _coerce_urgency(value: str) -> TicketUrgency:
    try:
        return TicketUrgency(value)
    except ValueError:
        return TicketUrgency.MEDIA
