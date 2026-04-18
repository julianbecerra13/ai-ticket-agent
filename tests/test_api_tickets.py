"""Tests de los endpoints de /tickets."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_crear_ticket_con_agente(client: TestClient) -> None:
    payload = {
        "user_id": "u1",
        "subject": "No puedo entrar",
        "body": "Olvide mi contrasena, necesito recuperar acceso.",
    }
    response = client.post("/tickets", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["classified"] is True
    assert data["decided"] is True
    assert data["ticket"]["prediction"]["category"] in {"cuenta", "tecnico", "informacion", "facturacion", "queja"}
    assert data["ticket"]["decision"]["action"] in {"auto_respond", "escalate", "request_info", "close_duplicate"}


def test_crear_ticket_sin_llm_queda_solo_clasificado(client_without_llm: TestClient) -> None:
    payload = {
        "user_id": "u2",
        "subject": "Consulta sobre factura",
        "body": "Donde veo la factura del mes pasado.",
    }
    response = client_without_llm.post("/tickets", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["classified"] is True
    assert data["decided"] is False
    assert data["ticket"]["decision"] is None
    assert data["notice"] and "LLM" in data["notice"]


def test_get_ticket_por_id(client: TestClient) -> None:
    created = client.post(
        "/tickets",
        json={"user_id": "u3", "subject": "Duda", "body": "Como exporto mis datos"},
    ).json()
    ticket_id = created["ticket"]["id"]

    response = client.get(f"/tickets/{ticket_id}")
    assert response.status_code == 200
    assert response.json()["id"] == ticket_id


def test_get_ticket_inexistente(client: TestClient) -> None:
    response = client.get("/tickets/99999")
    assert response.status_code == 404


def test_listado_devuelve_lista(client: TestClient) -> None:
    client.post(
        "/tickets",
        json={"user_id": "u4", "subject": "Pregunta", "body": "que hora es?"},
    )
    response = client.get("/tickets?limit=5")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
