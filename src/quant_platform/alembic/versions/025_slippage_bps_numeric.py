"""Promote fill_events.slippage_bps from FLOAT to NUMERIC(20, 8).

Slippage is a Decimal in the application layer; the prior FLOAT column
forced an IEEE-754 round-trip and accumulated rounding error across many
fills. NUMERIC(20, 8) matches the precision used elsewhere for slippage
(see migration 018) and lets the application bind Decimal end-to-end.

Revision ID: 025
Revises: 024
Create Date: 2026-05-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "025"
down_revision: str | None = "024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "fill_events",
        "slippage_bps",
        existing_type=sa.Float(),
        type_=sa.Numeric(20, 8),
        existing_nullable=True,
        postgresql_using="slippage_bps::numeric(20, 8)",
    )


def downgrade() -> None:
    op.alter_column(
        "fill_events",
        "slippage_bps",
        existing_type=sa.Numeric(20, 8),
        type_=sa.Float(),
        existing_nullable=True,
        postgresql_using="slippage_bps::double precision",
    )
