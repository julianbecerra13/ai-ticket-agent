"""Endpoint /metrics: agregados sobre los tickets procesados."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.schemas.metrics import MetricsSummary
from src.db.repositories import MetricsRepository
from src.db.session import get_db

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", response_model=MetricsSummary)
def get_metrics(
    window_hours: int | None = Query(default=None, ge=1, le=24 * 30),
    db: Session = Depends(get_db),
) -> MetricsSummary:
    summary = MetricsRepository(db).summary(since_hours=window_hours)
    return MetricsSummary(**summary)
