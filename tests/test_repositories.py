"""Tests de los repositorios (foco en agregaciones de metricas)."""

from __future__ import annotations

from src.db.models import AgentActionType, TicketCategory, TicketUrgency
from src.db.repositories import (
    AgentDecisionRepository,
    MetricsRepository,
    PredictionRepository,
    TicketRepository,
)


def test_metricas_vacias(db_session) -> None:
    summary = MetricsRepository(db_session).summary()
    assert summary["total_tickets"] == 0
    assert summary["auto_resolution_rate"] == 0.0


def test_metricas_cuentan_acciones(db_session) -> None:
    tr = TicketRepository(db_session)
    pr = PredictionRepository(db_session)
    dr = AgentDecisionRepository(db_session)

    for i in range(3):
        ticket = tr.create(user_id=f"u{i}", subject="s", body="b")
        pr.save(
            ticket_id=ticket.id,
            category=TicketCategory.CUENTA,
            urgency=TicketUrgency.BAJA,
            confidence_category=0.9,
            confidence_urgency=0.9,
        )
        dr.save(
            ticket_id=ticket.id,
            action=AgentActionType.AUTO_RESPOND,
            reasoning="ok",
            response_text="hola",
            llm_provider="mock",
            llm_model="mock-v1",
        )
    db_session.commit()

    summary = MetricsRepository(db_session).summary()
    assert summary["total_tickets"] == 3
    assert summary["auto_resolved"] == 3
    assert summary["auto_resolution_rate"] == 1.0
    assert summary["by_category"]["cuenta"] == 3
