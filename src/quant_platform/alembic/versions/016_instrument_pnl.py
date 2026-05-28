"""Per-instrument P&L attribution table.

Revision ID: 016
Revises: 015
Create Date: 2026-04-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "016"
down_revision: str | None = "015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "instrument_pnl",
        sa.Column("pnl_id", sa.UUID(), nullable=False),
        sa.Column("strategy_run_id", sa.UUID(), nullable=False),
        sa.Column("instrument_id", sa.UUID(), nullable=False),
        sa.Column("as_of", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("realized_pnl", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("unrealized_pnl", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("weight", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("contribution", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("pnl_id"),
    )
    op.create_index(
        "ix_instrument_pnl_run_as_of",
        "instrument_pnl",
        ["strategy_run_id", "as_of"],
    )
    op.create_index(
        "ix_instrument_pnl_instrument_as_of",
        "instrument_pnl",
        ["instrument_id", "as_of"],
    )


def downgrade() -> None:
    op.drop_index("ix_instrument_pnl_instrument_as_of", table_name="instrument_pnl")
    op.drop_index("ix_instrument_pnl_run_as_of", table_name="instrument_pnl")
    op.drop_table("instrument_pnl")
