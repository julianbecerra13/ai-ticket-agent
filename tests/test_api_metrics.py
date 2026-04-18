"""Tests del endpoint /metrics."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_vacio(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_tickets" in data
    assert "by_action" in data
    assert data["auto_resolution_rate"] >= 0.0


def test_metrics_reflejan_tickets_creados(client: TestClient) -> None:
    for _ in range(3):
        client.post(
            "/tickets",
            json={
                "user_id": "u-metricas",
                "subject": "consulta",
                "body": "quiero saber donde veo mis facturas",
            },
        )
    response = client.get("/metrics")
    data = response.json()
    assert data["total_tickets"] >= 3
    assert sum(data["by_category"].values()) >= 3
