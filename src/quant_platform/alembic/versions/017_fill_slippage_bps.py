"""Add slippage_bps column to fill_events for transaction cost analysis.

Revision ID: 017
Revises: 016
Create Date: 2026-04-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "017"
down_revision: str | None = "016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "fill_events",
        sa.Column("slippage_bps", sa.Float(), nullable=True),
    )
    op.create_index(
        "ix_fill_events_slippage_bps",
        "fill_events",
        ["slippage_bps"],
    )


def downgrade() -> None:
    op.drop_index("ix_fill_events_slippage_bps", table_name="fill_events")
    op.drop_column("fill_events", "slippage_bps")
