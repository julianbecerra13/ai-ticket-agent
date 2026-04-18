"""Repositorios: punto unico de acceso a datos.

Mantienen la API superior (servicios, rutas) libre de detalles de SQLAlchemy.
Cualquier query nueva deberia vivir aqui, nunca dispersa en los endpoints.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from src.db.models import (
    Action,
    ActionStatus,
    AgentActionType,
    AgentDecision,
    Prediction,
    Ticket,
    TicketCategory,
    TicketUrgency,
)


class TicketRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, *, user_id: str, subject: str, body: str) -> Ticket:
        ticket = Ticket(user_id=user_id, subject=subject, body=body)
        self.db.add(ticket)
        self.db.flush()
        return ticket

    def get(self, ticket_id: int) -> Ticket | None:
        stmt = (
            select(Ticket)
            .options(joinedload(Ticket.prediction), joinedload(Ticket.decision))
            .where(Ticket.id == ticket_id)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def list_recent(self, limit: int = 50) -> list[Ticket]:
        stmt = (
            select(Ticket)
            .options(joinedload(Ticket.prediction), joinedload(Ticket.decision))
            .order_by(Ticket.created_at.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())


class PredictionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def save(
        self,
        *,
        ticket_id: int,
        category: TicketCategory,
        urgency: TicketUrgency,
        confidence_category: float,
        confidence_urgency: float,
    ) -> Prediction:
        prediction = Prediction(
            ticket_id=ticket_id,
            category=category,
            urgency=urgency,
            confidence_category=confidence_category,
            confidence_urgency=confidence_urgency,
        )
        self.db.add(prediction)
        self.db.flush()
        return prediction


class AgentDecisionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def save(
        self,
        *,
        ticket_id: int,
        action: AgentActionType,
        reasoning: str,
        llm_provider: str,
        llm_model: str,
        response_text: str | None = None,
    ) -> AgentDecision:
        decision = AgentDecision(
            ticket_id=ticket_id,
            action=action,
            reasoning=reasoning,
            response_text=response_text,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        self.db.add(decision)
        self.db.flush()

        self.db.add(
            Action(
                decision_id=decision.id,
                type=action,
                payload={"response_text": response_text} if response_text else {},
                status=ActionStatus.EXECUTED,
                executed_at=datetime.now(timezone.utc),
            )
        )
        self.db.flush()
        return decision


class MetricsRepository:
    """Agrega datos para el endpoint /metrics y el dashboard."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def summary(self, since_hours: int | None = None) -> dict:
        stmt_tickets = select(func.count(Ticket.id))
        stmt_decisions = select(AgentDecision.action, func.count(AgentDecision.id))
        stmt_categories = select(Prediction.category, func.count(Prediction.id))
        stmt_urgencies = select(Prediction.urgency, func.count(Prediction.id))

        if since_hours is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
            stmt_tickets = stmt_tickets.where(Ticket.created_at >= cutoff)
            stmt_decisions = stmt_decisions.join(Ticket).where(Ticket.created_at >= cutoff)
            stmt_categories = stmt_categories.join(Ticket).where(Ticket.created_at >= cutoff)
            stmt_urgencies = stmt_urgencies.join(Ticket).where(Ticket.created_at >= cutoff)

        total_tickets = self.db.execute(stmt_tickets).scalar_one()
        by_action = {
            action.value: count
            for action, count in self.db.execute(stmt_decisions.group_by(AgentDecision.action))
        }
        by_category = {
            cat.value: count
            for cat, count in self.db.execute(stmt_categories.group_by(Prediction.category))
        }
        by_urgency = {
            urg.value: count
            for urg, count in self.db.execute(stmt_urgencies.group_by(Prediction.urgency))
        }

        auto_resolved = by_action.get(AgentActionType.AUTO_RESPOND.value, 0)
        total_decisions = sum(by_action.values())
        auto_rate = (auto_resolved / total_decisions) if total_decisions else 0.0

        return {
            "total_tickets": total_tickets,
            "total_decisions": total_decisions,
            "auto_resolved": auto_resolved,
            "auto_resolution_rate": round(auto_rate, 3),
            "by_action": by_action,
            "by_category": by_category,
            "by_urgency": by_urgency,
            "window_hours": since_hours,
        }
