"""Fixtures compartidas por la suite de tests.

Usamos SQLite en memoria con `StaticPool` (una sola conexion compartida entre
fixtures y la app), y limpiamos todas las tablas entre tests con un fixture
`autouse`. Esto evita el lio de transacciones anidadas y garantiza
aislamiento real por test.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

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
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="session")
def _session_factory(_engine) -> sessionmaker[Session]:
    return sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


@pytest.fixture(autouse=True)
def _clean_db(_engine):
    yield
    with _engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture
def db_session(_session_factory) -> Iterator[Session]:
    session = _session_factory()
    try:
        yield session
    finally:
        session.close()


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
def app(_session_factory, trained_classifier, mock_agent):
    reset_caches()
    app = create_app()

    def _override_db() -> Iterator[Session]:
        session = _session_factory()
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
def app_without_llm(_session_factory, trained_classifier):
    reset_caches()
    app = create_app()

    def _override_db() -> Iterator[Session]:
        session = _session_factory()
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
    monkeypatch.chdir(tmp_path)
