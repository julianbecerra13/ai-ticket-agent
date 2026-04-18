"""Ingestor de tickets desde archivo CSV.

Formato esperado:
    user_id,subject,body

Cualquier columna extra se ignora. Filas con subject o body vacios se saltan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from src.services.ticket_service import TicketService

log = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    total_rows: int
    processed: int
    skipped: int
    classified: int
    decided: int


class CsvIngestor:
    def __init__(self, service: TicketService) -> None:
        self._service = service

    def ingest(self, csv_path: str | Path) -> IngestionReport:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el CSV: {path}")

        df = pd.read_csv(path)
        required = {"subject", "body"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en el CSV: {sorted(missing)}")

        if "user_id" not in df.columns:
            df["user_id"] = "anonimo"

        total = len(df)
        processed = 0
        skipped = 0
        classified = 0
        decided = 0

        for _, row in df.iterrows():
            subject = str(row["subject"]).strip() if not pd.isna(row["subject"]) else ""
            body = str(row["body"]).strip() if not pd.isna(row["body"]) else ""
            user_id = str(row["user_id"]).strip() if not pd.isna(row["user_id"]) else "anonimo"

            if not subject or not body:
                skipped += 1
                continue

            result = self._service.create_and_process(user_id=user_id, subject=subject, body=body)
            processed += 1
            if result.classified:
                classified += 1
            if result.decided:
                decided += 1

        log.info(
            "Ingesta terminada: %s procesados, %s saltados (total %s).",
            processed,
            skipped,
            total,
        )
        return IngestionReport(
            total_rows=total,
            processed=processed,
            skipped=skipped,
            classified=classified,
            decided=decided,
        )


def ingest_csv(csv_path: str | Path, db: Session, service: TicketService) -> IngestionReport:
    return CsvIngestor(service).ingest(csv_path)
