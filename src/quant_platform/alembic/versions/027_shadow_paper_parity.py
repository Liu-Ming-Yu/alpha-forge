"""Add shadow-vs-paper parity observations.

Revision ID: 027
Revises: 026
Create Date: 2026-05-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "027"
down_revision: str | None = "026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "shadow_paper_parity_observations",
        sa.Column("parity_id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column("signal_name", sa.Text(), nullable=False),
        sa.Column("signal_type", sa.Text(), nullable=False),
        sa.Column("trading_day", sa.Date(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("instruments_compared", sa.Integer(), nullable=False),
        sa.Column("missing_instruments", sa.Integer(), nullable=False),
        sa.Column("max_target_weight_diff_bps", sa.Float(), nullable=False),
        sa.Column("order_side_mismatches", sa.Integer(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("instruments_compared >= 0", name="ck_shadow_paper_instruments"),
        sa.CheckConstraint("missing_instruments >= 0", name="ck_shadow_paper_missing"),
        sa.CheckConstraint(
            "max_target_weight_diff_bps >= 0",
            name="ck_shadow_paper_target_diff",
        ),
        sa.CheckConstraint(
            "order_side_mismatches >= 0",
            name="ck_shadow_paper_side_mismatches",
        ),
    )
    op.create_index(
        "ix_shadow_paper_parity_signal_day",
        "shadow_paper_parity_observations",
        ["signal_type", "signal_name", "trading_day"],
        unique=True,
    )
    op.create_index(
        "ix_shadow_paper_parity_signal_latest",
        "shadow_paper_parity_observations",
        ["signal_type", "signal_name", "as_of"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_shadow_paper_parity_signal_latest",
        table_name="shadow_paper_parity_observations",
    )
    op.drop_index(
        "ix_shadow_paper_parity_signal_day",
        table_name="shadow_paper_parity_observations",
    )
    op.drop_table("shadow_paper_parity_observations")
