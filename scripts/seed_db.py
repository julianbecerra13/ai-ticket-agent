"""Carga 20 tickets de ejemplo en la base para demos sin configurar nada mas."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from src.agent.agent import TicketAgent
from src.agent.providers.factory import build_provider
from src.config import settings
from src.db.session import SessionLocal
from src.logging_config import configure_logging
from src.ml.classifier import TicketClassifier
from src.ml.dataset import generate_dataset
from src.services.ticket_service import TicketService

log = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    model_path = Path(settings.model_path)
    classifier = None
    if model_path.exists():
        classifier = TicketClassifier.load(model_path)
    else:
        log.warning("No hay modelo entrenado; los tickets no se clasificaran.")

    provider = build_provider()
    agent = TicketAgent(provider=provider) if provider else None

    sample = generate_dataset(samples_per_template=1).head(20)
    rng = random.Random(7)
    users = [f"user_{i}" for i in range(1, 6)]

    with SessionLocal() as db:
        service = TicketService(db=db, classifier=classifier, agent=agent)
        inserted = 0
        for _, row in sample.iterrows():
            service.create_and_process(
                user_id=rng.choice(users),
                subject=row["subject"],
                body=row["body"],
            )
            inserted += 1
        log.info("Insertados %s tickets de ejemplo.", inserted)


if __name__ == "__main__":
    main()
