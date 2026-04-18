"""Endpoints de tickets."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.dependencies import get_ticket_service
from src.api.schemas.ticket import ProcessedTicketOut, TicketCreate, TicketOut
from src.db.repositories import TicketRepository
from src.db.session import get_db
from src.services.ticket_service import TicketService

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post("", response_model=ProcessedTicketOut, status_code=status.HTTP_201_CREATED)
def create_ticket(
    payload: TicketCreate,
    service: TicketService = Depends(get_ticket_service),
) -> ProcessedTicketOut:
    result = service.create_and_process(
        user_id=payload.user_id, subject=payload.subject, body=payload.body
    )
    notice = None
    if not result.classified:
        notice = "Clasificador no entrenado. Ejecuta `make train`."
    elif not result.decided:
        notice = "Ticket clasificado pero sin decision: no hay proveedor LLM configurado."
    return ProcessedTicketOut(
        ticket=TicketOut.model_validate(result.ticket),
        classified=result.classified,
        decided=result.decided,
        notice=notice,
    )


@router.get("/{ticket_id}", response_model=TicketOut)
def get_ticket(
    ticket_id: int,
    db: Session = Depends(get_db),
) -> TicketOut:
    ticket = TicketRepository(db).get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket no encontrado.")
    return TicketOut.model_validate(ticket)


@router.get("", response_model=list[TicketOut])
def list_tickets(
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[TicketOut]:
    limit = max(1, min(limit, 200))
    tickets = TicketRepository(db).list_recent(limit=limit)
    return [TicketOut.model_validate(t) for t in tickets]
