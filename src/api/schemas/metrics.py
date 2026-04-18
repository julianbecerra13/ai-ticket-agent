"""Schema del endpoint /metrics."""

from __future__ import annotations

from pydantic import BaseModel


class MetricsSummary(BaseModel):
    total_tickets: int
    total_decisions: int
    auto_resolved: int
    auto_resolution_rate: float
    by_action: dict[str, int]
    by_category: dict[str, int]
    by_urgency: dict[str, int]
    window_hours: int | None = None
