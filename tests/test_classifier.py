"""Tests del clasificador TF-IDF + LogisticRegression."""

from __future__ import annotations

import pytest
from src.db.models import TicketCategory, TicketUrgency
from src.ml.classifier import TicketClassifier
from src.ml.dataset import generate_dataset


@pytest.fixture(scope="module")
def trained():
    clf = TicketClassifier()
    clf.train(generate_dataset(samples_per_template=6))
    return clf


def test_clasificador_entrena_y_predice(trained: TicketClassifier) -> None:
    result = trained.predict(
        subject="No puedo iniciar sesion",
        body="Me dice credenciales invalidas aunque la contrasena es correcta.",
    )
    assert result.category == TicketCategory.CUENTA
    assert 0.0 < result.confidence_category <= 1.0
    assert 0.0 < result.confidence_urgency <= 1.0


def test_prediccion_critica_para_sistema_caido(trained: TicketClassifier) -> None:
    result = trained.predict(
        subject="URGENTE: La pagina esta caida",
        body="Todo el portal esta fuera de servicio para mi equipo.",
    )
    assert result.urgency in (TicketUrgency.ALTA, TicketUrgency.CRITICA)


def test_save_and_load(trained: TicketClassifier, tmp_path) -> None:
    path = tmp_path / "clf.pkl"
    trained.save(path)
    restored = TicketClassifier.load(path)
    assert restored.is_ready
    r1 = trained.predict(subject="Como descargo la factura", body="Donde esta la factura?")
    r2 = restored.predict(subject="Como descargo la factura", body="Donde esta la factura?")
    assert r1.category == r2.category


def test_predict_falla_sin_entrenar() -> None:
    clf = TicketClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(subject="x", body="y")
