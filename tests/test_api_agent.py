"""Tests del endpoint /agent/decide."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_decide_sin_llm_devuelve_503(client_without_llm: TestClient) -> None:
    created = client_without_llm.post(
        "/tickets",
        json={"user_id": "u", "subject": "a", "body": "ayuda"},
    ).json()
    ticket_id = created["ticket"]["id"]

    response = client_without_llm.post(f"/agent/decide/{ticket_id}")
    assert response.status_code == 503
    assert "LLM" in response.json()["detail"]


def test_decide_sobre_ticket_existente(client: TestClient) -> None:
    created = client.post(
        "/tickets",
        json={"user_id": "u", "subject": "pregunta", "body": "donde veo mis facturas"},
    ).json()
    ticket_id = created["ticket"]["id"]

    response = client.post(f"/agent/decide/{ticket_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["action"] in {"auto_respond", "escalate", "request_info", "close_duplicate"}
    assert data["llm_provider"] == "mock"


def test_decide_ticket_inexistente(client: TestClient) -> None:
    response = client.post("/agent/decide/99999")
    assert response.status_code == 404
