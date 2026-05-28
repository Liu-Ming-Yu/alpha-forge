"""Broker paper gate evidence.

Revision ID: 013
Revises: 012
Create Date: 2026-04-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "013"
down_revision: str | None = "012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "broker_smoke_observations",
        sa.Column("observation_id", sa.UUID(), nullable=False),
        sa.Column("observed_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("host", sa.Text(), nullable=False),
        sa.Column("port", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("account_status", sa.Text(), nullable=False),
        sa.Column("positions_status", sa.Text(), nullable=False),
        sa.Column("open_orders_status", sa.Text(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("observation_id"),
    )
    op.create_index(
        "ix_broker_smoke_observed_at",
        "broker_smoke_observations",
        ["observed_at"],
    )

    op.create_table(
        "paper_lifecycle_observations",
        sa.Column("observation_id", sa.UUID(), nullable=False),
        sa.Column("observed_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("host", sa.Text(), nullable=False),
        sa.Column("port", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.Integer(), nullable=False),
        sa.Column("instrument_id", sa.UUID(), nullable=False),
        sa.Column("broker_order_id", sa.Text(), nullable=False),
        sa.Column("max_notional_usd", sa.Numeric(20, 8), nullable=False),
        sa.Column("limit_price", sa.Numeric(20, 8), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("ack_status", sa.Text(), nullable=False),
        sa.Column("cancel_status", sa.Text(), nullable=False),
        sa.Column("stale_open_order_count", sa.Integer(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("observation_id"),
    )
    op.create_index(
        "ix_paper_lifecycle_observed_at",
        "paper_lifecycle_observations",
        ["observed_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_paper_lifecycle_observed_at", table_name="paper_lifecycle_observations")
    op.drop_table("paper_lifecycle_observations")
    op.drop_index("ix_broker_smoke_observed_at", table_name="broker_smoke_observations")
    op.drop_table("broker_smoke_observations")
