"""Initial schema — order intents, fills, account snapshots, positions, audit log.

Revision ID: 001
Revises: None
Create Date: 2026-04-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "order_intents",
        sa.Column("order_id", sa.Uuid(), primary_key=True),
        sa.Column("strategy_run_id", sa.Uuid(), nullable=False),
        sa.Column("portfolio_target_id", sa.Uuid(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("side", sa.Text(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("order_type", sa.Text(), nullable=False),
        sa.Column("time_in_force", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("limit_price", sa.Numeric()),
        sa.Column("cash_reservation_id", sa.Uuid()),
        sa.Column("is_terminal", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("terminal_reason", sa.Text()),
    )

    op.create_table(
        "fill_events",
        sa.Column("fill_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "order_id",
            sa.Uuid(),
            sa.ForeignKey("order_intents.order_id"),
            nullable=False,
        ),
        sa.Column("broker_order_id", sa.Text(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("side", sa.Text(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("fill_price", sa.Numeric(), nullable=False),
        sa.Column("commission", sa.Numeric(), nullable=False),
        sa.Column("currency", sa.Text(), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("supersedes_id", sa.Uuid()),
    )

    op.create_table(
        "account_snapshots",
        sa.Column("snapshot_id", sa.Uuid(), primary_key=True),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("settled_cash", sa.Numeric(), nullable=False),
        sa.Column("unsettled_cash", sa.Numeric(), nullable=False),
        sa.Column("reserved_cash", sa.Numeric(), nullable=False),
        sa.Column("available_cash", sa.Numeric(), nullable=False),
        sa.Column("net_asset_value", sa.Numeric(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False, server_default="broker"),
    )

    op.create_table(
        "position_snapshots",
        sa.Column(
            "snapshot_id",
            sa.Uuid(),
            sa.ForeignKey("account_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("average_cost", sa.Numeric(), nullable=False),
        sa.Column("market_price", sa.Numeric(), nullable=False),
        sa.Column("market_value", sa.Numeric(), nullable=False),
        sa.Column("unrealised_pnl", sa.Numeric(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.Text(), nullable=False, server_default="broker"),
        sa.PrimaryKeyConstraint("snapshot_id", "instrument_id"),
    )

    op.create_table(
        "audit_log",
        sa.Column(
            "entry_id",
            sa.Uuid(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column(
            "event_payload",
            postgresql.JSONB(),
            nullable=False,
        ),
        sa.Column(
            "context",
            postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index("ix_fill_events_order_id", "fill_events", ["order_id"])
    op.create_index("ix_order_intents_strategy_run", "order_intents", ["strategy_run_id"])
    op.create_index("ix_account_snapshots_as_of", "account_snapshots", ["as_of"])
    op.create_index("ix_audit_log_event_type", "audit_log", ["event_type"])
    op.create_index("ix_audit_log_recorded_at", "audit_log", ["recorded_at"])


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("position_snapshots")
    op.drop_table("account_snapshots")
    op.drop_table("fill_events")
    op.drop_table("order_intents")
