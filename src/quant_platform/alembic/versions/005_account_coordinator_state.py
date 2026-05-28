"""Durable AccountStateCoordinator state.

Revision ID: 005
Revises: 004
Create Date: 2026-04-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pending_settlement_lots",
        sa.Column("lot_id", sa.Text(), primary_key=True),
        sa.Column("fill_id", sa.Text(), nullable=False),
        sa.Column("order_id", sa.Text(), nullable=False),
        sa.Column("instrument_id", sa.Text(), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("settlement_date", sa.Date(), nullable=False),
        sa.Column("gross_proceeds", sa.Numeric(20, 8), nullable=False),
        sa.Column("commission", sa.Numeric(20, 8), nullable=False),
        sa.Column("net_proceeds", sa.Numeric(20, 8), nullable=False),
        sa.Column("currency", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_pending_settlement_lots_settlement_date",
        "pending_settlement_lots",
        ["settlement_date"],
    )
    op.create_index(
        "ix_pending_settlement_lots_run_id",
        "pending_settlement_lots",
        ["run_id"],
    )

    op.create_table(
        "completed_order_hints",
        sa.Column("order_id", sa.Text(), primary_key=True),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_pending_settlement_lots_run_id",
        table_name="pending_settlement_lots",
    )
    op.drop_index(
        "ix_pending_settlement_lots_settlement_date",
        table_name="pending_settlement_lots",
    )
    op.drop_table("completed_order_hints")
    op.drop_table("pending_settlement_lots")
