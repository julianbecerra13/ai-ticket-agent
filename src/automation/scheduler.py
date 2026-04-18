"""Scheduler opcional basado en APScheduler.

Se deja desactivado por defecto; la forma recomendada de activarlo es correr
`scripts/run_scheduler.py` aparte de la API. Se documenta en el README.
"""

from __future__ import annotations

import logging
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.agent.agent import TicketAgent
from src.agent.providers.factory import build_provider
from src.automation.ingestor import CsvIngestor
from src.config import settings
from src.db.session import SessionLocal
from src.ml.classifier import TicketClassifier
from src.services.ticket_service import TicketService

log = logging.getLogger(__name__)


def run_ingestor_job(csv_path: str | Path = "data/raw/batch.csv") -> None:
    path = Path(csv_path)
    if not path.exists():
        log.info("No hay CSV en %s, se omite este ciclo.", path)
        return

    classifier = None
    model_path = Path(settings.model_path)
    if model_path.exists():
        classifier = TicketClassifier.load(model_path)

    provider = build_provider()
    agent = TicketAgent(provider=provider) if provider else None

    with SessionLocal() as db:
        service = TicketService(db=db, classifier=classifier, agent=agent)
        report = CsvIngestor(service).ingest(path)
        log.info("Reporte: %s", report)


def build_scheduler(interval_minutes: int = 10) -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_ingestor_job,
        trigger=IntervalTrigger(minutes=interval_minutes),
        name="csv-ingestor",
        replace_existing=True,
    )
    return scheduler
