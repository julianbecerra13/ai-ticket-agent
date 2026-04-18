"""Endpoint /health.

Muestra el estado de la aplicacion y es la primera parada de cualquier
recruiter o SRE que pruebe el servicio.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.agent.agent import TicketAgent
from src.api.dependencies import get_agent, get_classifier
from src.config import get_settings
from src.db.session import get_db
from src.ml.classifier import TicketClassifier

router = APIRouter()


@router.get("/health", tags=["health"])
def health(
    db: Session = Depends(get_db),
    classifier: TicketClassifier | None = Depends(get_classifier),
    agent: TicketAgent | None = Depends(get_agent),
) -> dict:
    settings = get_settings()

    database_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        database_ok = False

    return {
        "status": "ok" if database_ok else "degraded",
        "database": "connected" if database_ok else "unreachable",
        "classifier": "loaded" if classifier and classifier.is_ready else "not_trained",
        "llm_provider": agent._provider.name if agent else None,
        "llm_model": agent._provider.model if agent else None,
        "llm_status": "ready" if agent else "not_configured",
        "resolved_provider": settings.resolve_llm_provider(),
    }
