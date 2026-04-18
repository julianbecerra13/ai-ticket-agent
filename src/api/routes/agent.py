"""Endpoint /agent/decide: pide una decision sobre un ticket existente."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.agent.agent import TicketAgent
from src.api.dependencies import get_agent, get_classifier
from src.api.schemas.ticket import AgentDecisionOut
from src.db.repositories import AgentDecisionRepository, PredictionRepository, TicketRepository
from src.db.session import get_db
from src.ml.classifier import PredictionResult, TicketClassifier

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/decide/{ticket_id}", response_model=AgentDecisionOut)
def decide(
    ticket_id: int,
    db: Session = Depends(get_db),
    classifier: TicketClassifier | None = Depends(get_classifier),
    agent: TicketAgent | None = Depends(get_agent),
) -> AgentDecisionOut:
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No hay proveedor LLM configurado. Configura ANTHROPIC_API_KEY, OPENAI_API_KEY u OLLAMA_HOST.",
        )
    if classifier is None or not classifier.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clasificador no entrenado. Ejecuta `make train` antes de invocar al agente.",
        )

    ticket = TicketRepository(db).get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket no encontrado.")

    if ticket.prediction is None:
        prediction = classifier.predict(subject=ticket.subject, body=ticket.body)
        PredictionRepository(db).save(
            ticket_id=ticket.id,
            category=prediction.category,
            urgency=prediction.urgency,
            confidence_category=prediction.confidence_category,
            confidence_urgency=prediction.confidence_urgency,
        )
    else:
        prediction = PredictionResult(
            category=ticket.prediction.category,
            urgency=ticket.prediction.urgency,
            confidence_category=ticket.prediction.confidence_category,
            confidence_urgency=ticket.prediction.confidence_urgency,
        )

    recent = [t for t in TicketRepository(db).list_recent(limit=10) if t.user_id == ticket.user_id and t.id != ticket.id]
    result = agent.decide(ticket=ticket, prediction=prediction, recent_history=recent)

    decision_repo = AgentDecisionRepository(db)
    if ticket.decision is not None:
        db.delete(ticket.decision)
        db.flush()

    saved = decision_repo.save(
        ticket_id=ticket.id,
        action=result.action,
        reasoning=result.reasoning,
        response_text=result.response_text,
        llm_provider=result.provider_name,
        llm_model=result.provider_model,
    )
    db.commit()
    return AgentDecisionOut.model_validate(saved)
