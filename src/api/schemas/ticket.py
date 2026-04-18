"""Schemas Pydantic para tickets."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.db.models import AgentActionType, TicketCategory, TicketUrgency


class TicketCreate(BaseModel):
    user_id: str = Field(default="anonimo", max_length=64)
    subject: str = Field(min_length=1, max_length=255)
    body: str = Field(min_length=1)


class PredictionOut(BaseModel):
    category: TicketCategory
    urgency: TicketUrgency
    confidence_category: float
    confidence_urgency: float

    model_config = ConfigDict(from_attributes=True)


class AgentDecisionOut(BaseModel):
    action: AgentActionType
    reasoning: str
    response_text: str | None
    llm_provider: str
    llm_model: str

    model_config = ConfigDict(from_attributes=True)


class TicketOut(BaseModel):
    id: int
    user_id: str
    subject: str
    body: str
    created_at: datetime
    prediction: PredictionOut | None = None
    decision: AgentDecisionOut | None = None

    model_config = ConfigDict(from_attributes=True)


class ProcessedTicketOut(BaseModel):
    ticket: TicketOut
    classified: bool
    decided: bool
    notice: str | None = None
