"""Tests del ingestor CSV."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.agent.agent import TicketAgent
from src.agent.providers.mock_provider import MockProvider
from src.automation.ingestor import CsvIngestor
from src.ml.classifier import TicketClassifier
from src.ml.dataset import generate_dataset
from src.services.ticket_service import TicketService


@pytest.fixture(scope="module")
def _classifier() -> TicketClassifier:
    clf = TicketClassifier()
    clf.train(generate_dataset(samples_per_template=5))
    return clf


def test_ingesta_de_csv(tmp_path: Path, db_session, _classifier) -> None:
    csv = tmp_path / "batch.csv"
    csv.write_text(
        "user_id,subject,body\n"
        "u1,No puedo entrar,olvide mi contrasena\n"
        "u2,Cobro duplicado,Me cobraron dos veces\n"
        ",,\n"
        "u3,Consulta,Como exporto mis datos\n",
        encoding="utf-8",
    )

    service = TicketService(
        db=db_session, classifier=_classifier, agent=TicketAgent(MockProvider())
    )
    report = CsvIngestor(service).ingest(csv)

    assert report.total_rows == 4
    assert report.processed == 3
    assert report.skipped == 1
    assert report.classified == 3
    assert report.decided == 3


def test_csv_inexistente_lanza(tmp_path: Path, db_session, _classifier) -> None:
    service = TicketService(db=db_session, classifier=_classifier, agent=None)
    with pytest.raises(FileNotFoundError):
        CsvIngestor(service).ingest(tmp_path / "no-existe.csv")


def test_csv_sin_columnas_requeridas(tmp_path: Path, db_session, _classifier) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text("foo,bar\n1,2\n", encoding="utf-8")
    service = TicketService(db=db_session, classifier=_classifier, agent=None)
    with pytest.raises(ValueError):
        CsvIngestor(service).ingest(csv)
