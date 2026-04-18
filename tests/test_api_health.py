"""Tests del endpoint /health."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_con_todo_activo(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["classifier"] == "loaded"
    assert data["llm_status"] == "ready"
    assert data["llm_provider"] == "mock"


def test_health_sin_llm(client_without_llm: TestClient) -> None:
    response = client_without_llm.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["classifier"] == "loaded"
    assert data["llm_status"] == "not_configured"
    assert data["llm_provider"] is None
