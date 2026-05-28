"""Production readiness state.

Adds durable NAV snapshots for live/paper performance governance and daily
text-signal IC observations for automated promotion gates.

Revision ID: 007
Revises: 006
Create Date: 2026-04-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "nav_snapshots",
        sa.Column("snapshot_id", sa.Uuid(), primary_key=True),
        sa.Column("strategy_run_id", sa.Uuid(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("net_asset_value", sa.Numeric(20, 8), nullable=False),
        sa.Column("gross_exposure", sa.Numeric(20, 8), nullable=False, server_default="0"),
        sa.Column("cash", sa.Numeric(20, 8), nullable=False, server_default="0"),
        sa.Column("source", sa.Text(), nullable=False, server_default="runtime"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_nav_snapshots_strategy_run_as_of",
        "nav_snapshots",
        ["strategy_run_id", "as_of"],
    )

    op.create_table(
        "text_signal_ic_observations",
        sa.Column("strategy_name", sa.Text(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("daily_ic", sa.Float(), nullable=False),
        sa.Column("observations", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("strategy_name", "as_of"),
    )
    op.create_index(
        "ix_text_signal_ic_strategy_as_of",
        "text_signal_ic_observations",
        ["strategy_name", "as_of"],
    )


def downgrade() -> None:
    op.drop_index("ix_text_signal_ic_strategy_as_of", table_name="text_signal_ic_observations")
    op.drop_table("text_signal_ic_observations")
    op.drop_index("ix_nav_snapshots_strategy_run_as_of", table_name="nav_snapshots")
    op.drop_table("nav_snapshots")
