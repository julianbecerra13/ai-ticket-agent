"""Dependencias de FastAPI: clasificador, agente, servicios.

Cacheamos el clasificador y el proveedor LLM a nivel proceso porque son caros
de instanciar. La sesion de BD si se crea por request.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import Depends
from sqlalchemy.orm import Session

from src.agent.agent import TicketAgent
from src.agent.providers.factory import build_provider
from src.config import settings
from src.db.session import get_db
from src.ml.classifier import TicketClassifier
from src.services.ticket_service import TicketService

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_classifier() -> TicketClassifier | None:
    path = Path(settings.model_path)
    if not path.exists():
        log.warning("No se encontro modelo entrenado en %s. Ejecuta `make train`.", path)
        return None
    try:
        return TicketClassifier.load(path)
    except Exception as exc:
        log.error("Error cargando el clasificador: %s", exc)
        return None


@lru_cache(maxsize=1)
def get_agent() -> TicketAgent | None:
    provider = build_provider()
    if provider is None:
        return None
    return TicketAgent(provider=provider)


def get_ticket_service(
    db: Session = Depends(get_db),
    classifier: TicketClassifier | None = Depends(get_classifier),
    agent: TicketAgent | None = Depends(get_agent),
) -> TicketService:
    return TicketService(db=db, classifier=classifier, agent=agent)


def reset_caches() -> None:
    """Limpia las caches de clasificador/agente (util en tests)."""
    get_classifier.cache_clear()
    get_agent.cache_clear()
