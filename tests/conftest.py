"""Fixtures compartidas por la suite de tests.

Usamos SQLite en memoria para que los tests no dependan de Postgres y corran
en cualquier maquina. El `TicketAgent` se inyecta con `MockProvider`, evitando
llamadas a proveedores reales.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.agent.agent import TicketAgent
from src.agent.providers.mock_provider import MockProvider
from src.api.dependencies import get_agent, get_classifier, reset_caches
from src.api.main import create_app
from src.db.models import Base
from src.db.session import get_db
from src.ml.classifier import TicketClassifier
from src.ml.dataset import generate_dataset


@pytest.fixture(scope="session")
def _engine():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(_engine) -> Iterator[Session]:
    TestSession = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    connection = _engine.connect()
    transaction = connection.begin()
    session = TestSession(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture(scope="session")
def trained_classifier(tmp_path_factory) -> TicketClassifier:
    df = generate_dataset(samples_per_template=6)
    clf = TicketClassifier()
    clf.train(df)
    tmp = tmp_path_factory.mktemp("models") / "clf.pkl"
    clf.save(tmp)
    return TicketClassifier.load(tmp)


@pytest.fixture
def mock_agent() -> TicketAgent:
    return TicketAgent(provider=MockProvider())


@pytest.fixture
def app(_engine, trained_classifier, mock_agent):
    reset_caches()
    app = create_app()

    TestSession = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)

    def _override_db() -> Iterator[Session]:
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_classifier] = lambda: trained_classifier
    app.dependency_overrides[get_agent] = lambda: mock_agent
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def app_without_llm(_engine, trained_classifier):
    reset_caches()
    app = create_app()

    TestSession = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)

    def _override_db() -> Iterator[Session]:
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_classifier] = lambda: trained_classifier
    app.dependency_overrides[get_agent] = lambda: None
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client_without_llm(app_without_llm) -> TestClient:
    return TestClient(app_without_llm)


@pytest.fixture(autouse=True)
def _tmp_cwd(tmp_path: Path, monkeypatch):
    """Evita que los tests escriban en el directorio del proyecto."""
    monkeypatch.chdir(tmp_path)
