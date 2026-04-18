"""Modelos SQLAlchemy del dominio de tickets.

Estructura:
- Ticket: la peticion entrante tal como la manda el cliente.
- Prediction: salida del clasificador ML (categoria + urgencia).
- AgentDecision: decision del agente LLM sobre que hacer con el ticket.
- Action: accion concreta ejecutada a raiz de la decision.

Relaciones principales:
    Ticket 1 ─── 1 Prediction
    Ticket 1 ─── 1 AgentDecision ─── N Action
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum, StrEnum

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _enum(enum_cls: type[Enum], name: str) -> SAEnum:
    """Crea un tipo Enum portable (VARCHAR + CHECK) compatible con SQLite y Postgres."""
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        values_callable=lambda e: [m.value for m in e],
        length=32,
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class TicketCategory(StrEnum):
    TECNICO = "tecnico"
    FACTURACION = "facturacion"
    CUENTA = "cuenta"
    INFORMACION = "informacion"
    QUEJA = "queja"


class TicketUrgency(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"
    CRITICA = "critica"


class AgentActionType(StrEnum):
    AUTO_RESPOND = "auto_respond"
    ESCALATE = "escalate"
    REQUEST_INFO = "request_info"
    CLOSE_DUPLICATE = "close_duplicate"


class ActionStatus(StrEnum):
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    subject: Mapped[str] = mapped_column(String(255))
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    prediction: Mapped[Prediction | None] = relationship(
        back_populates="ticket", uselist=False, cascade="all, delete-orphan"
    )
    decision: Mapped[AgentDecision | None] = relationship(
        back_populates="ticket", uselist=False, cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), unique=True, index=True
    )
    category: Mapped[TicketCategory] = mapped_column(_enum(TicketCategory, "ticket_category"))
    urgency: Mapped[TicketUrgency] = mapped_column(_enum(TicketUrgency, "ticket_urgency"))
    confidence_category: Mapped[float] = mapped_column(Float)
    confidence_urgency: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    ticket: Mapped[Ticket] = relationship(back_populates="prediction")


class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), unique=True, index=True
    )
    action: Mapped[AgentActionType] = mapped_column(_enum(AgentActionType, "agent_action_type"))
    reasoning: Mapped[str] = mapped_column(Text)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_provider: Mapped[str] = mapped_column(String(32))
    llm_model: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    ticket: Mapped[Ticket] = relationship(back_populates="decision")
    actions: Mapped[list[Action]] = relationship(
        back_populates="decision", cascade="all, delete-orphan"
    )


class Action(Base):
    __tablename__ = "actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    decision_id: Mapped[int] = mapped_column(
        ForeignKey("agent_decisions.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[AgentActionType] = mapped_column(
        _enum(AgentActionType, "agent_action_type_action")
    )
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[ActionStatus] = mapped_column(
        _enum(ActionStatus, "action_status"), default=ActionStatus.PENDING
    )
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    decision: Mapped[AgentDecision] = relationship(back_populates="actions")
