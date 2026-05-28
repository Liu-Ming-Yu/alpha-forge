"""Operational readiness evidence.

Revision ID: 010
Revises: 009
Create Date: 2026-04-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "010"
down_revision: str | None = "009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "signal_gate_observations",
        sa.Column("signal_name", sa.Text(), nullable=False),
        sa.Column("signal_type", sa.Text(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("daily_ic", sa.Float(), nullable=False),
        sa.Column("observations", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("drawdown", sa.Float(), nullable=False, server_default="0"),
        sa.Column("turnover", sa.Float(), nullable=False, server_default="0"),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("signal_name", "signal_type", "as_of"),
    )
    op.create_index(
        "ix_signal_gate_type_name_as_of",
        "signal_gate_observations",
        ["signal_type", "signal_name", "as_of"],
    )

    op.create_table(
        "runtime_heartbeats",
        sa.Column("component", sa.Text(), primary_key=True),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "broker_health_observations",
        sa.Column("observation_id", sa.Uuid(), primary_key=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("detail", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_broker_health_observed_at",
        "broker_health_observations",
        ["observed_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_broker_health_observed_at", table_name="broker_health_observations")
    op.drop_table("broker_health_observations")
    op.drop_table("runtime_heartbeats")
    op.drop_index("ix_signal_gate_type_name_as_of", table_name="signal_gate_observations")
    op.drop_table("signal_gate_observations")
