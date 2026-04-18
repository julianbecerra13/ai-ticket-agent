"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-18 00:00:00

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tickets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_tickets_user_id", "tickets", ["user_id"])
    op.create_index("ix_tickets_created_at", "tickets", ["created_at"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "ticket_id",
            sa.Integer(),
            sa.ForeignKey("tickets.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("urgency", sa.String(length=16), nullable=False),
        sa.Column("confidence_category", sa.Float(), nullable=False),
        sa.Column("confidence_urgency", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_predictions_ticket_id", "predictions", ["ticket_id"])

    op.create_table(
        "agent_decisions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "ticket_id",
            sa.Integer(),
            sa.ForeignKey("tickets.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("llm_provider", sa.String(length=32), nullable=False),
        sa.Column("llm_model", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_agent_decisions_ticket_id", "agent_decisions", ["ticket_id"])

    op.create_table(
        "actions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "decision_id",
            sa.Integer(),
            sa.ForeignKey("agent_decisions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_actions_decision_id", "actions", ["decision_id"])


def downgrade() -> None:
    op.drop_index("ix_actions_decision_id", table_name="actions")
    op.drop_table("actions")

    op.drop_index("ix_agent_decisions_ticket_id", table_name="agent_decisions")
    op.drop_table("agent_decisions")

    op.drop_index("ix_predictions_ticket_id", table_name="predictions")
    op.drop_table("predictions")

    op.drop_index("ix_tickets_created_at", table_name="tickets")
    op.drop_index("ix_tickets_user_id", table_name="tickets")
    op.drop_table("tickets")
