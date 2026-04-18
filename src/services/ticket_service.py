"""Servicio de tickets: orquesta clasificador, agente y repositorios."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.agent.agent import TicketAgent
from src.db.models import Ticket
from src.db.repositories import (
    AgentDecisionRepository,
    PredictionRepository,
    TicketRepository,
)
from src.ml.classifier import TicketClassifier

log = logging.getLogger(__name__)


@dataclass
class ProcessedTicket:
    ticket: Ticket
    classified: bool
    decided: bool


class TicketService:
    def __init__(
        self,
        *,
        db: Session,
        classifier: TicketClassifier | None,
        agent: TicketAgent | None,
    ) -> None:
        self.db = db
        self.classifier = classifier
        self.agent = agent

    def create_and_process(
        self,
        *,
        user_id: str,
        subject: str,
        body: str,
    ) -> ProcessedTicket:
        ticket_repo = TicketRepository(self.db)
        prediction_repo = PredictionRepository(self.db)
        decision_repo = AgentDecisionRepository(self.db)

        ticket = ticket_repo.create(user_id=user_id, subject=subject, body=body)
        classified = False
        decided = False

        if self.classifier is not None and self.classifier.is_ready:
            prediction = self.classifier.predict(subject=subject, body=body)
            prediction_repo.save(
                ticket_id=ticket.id,
                category=prediction.category,
                urgency=prediction.urgency,
                confidence_category=prediction.confidence_category,
                confidence_urgency=prediction.confidence_urgency,
            )
            classified = True

            if self.agent is not None:
                recent = [
                    t for t in ticket_repo.list_recent(limit=10) if t.user_id == user_id and t.id != ticket.id
                ]
                decision = self.agent.decide(
                    ticket=ticket, prediction=prediction, recent_history=recent
                )
                decision_repo.save(
                    ticket_id=ticket.id,
                    action=decision.action,
                    reasoning=decision.reasoning,
                    response_text=decision.response_text,
                    llm_provider=decision.provider_name,
                    llm_model=decision.provider_model,
                )
                decided = True
        else:
            log.warning("Clasificador no disponible; el ticket %s queda sin procesar.", ticket.id)

        self.db.commit()
        return ProcessedTicket(ticket=ticket_repo.get(ticket.id), classified=classified, decided=decided)
