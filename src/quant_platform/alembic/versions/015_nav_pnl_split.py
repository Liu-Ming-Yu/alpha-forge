"""Add realized_pnl and unrealized_pnl columns to nav_snapshots.

Revision ID: 015
Revises: 014
Create Date: 2026-04-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "015"
down_revision: str | None = "014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "nav_snapshots",
        sa.Column("realized_pnl", sa.Numeric(precision=20, scale=8), nullable=True),
    )
    op.add_column(
        "nav_snapshots",
        sa.Column("unrealized_pnl", sa.Numeric(precision=20, scale=8), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("nav_snapshots", "unrealized_pnl")
    op.drop_column("nav_snapshots", "realized_pnl")
