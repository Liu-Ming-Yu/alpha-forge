"""Multi-engine governance state.

Revision ID: 008
Revises: 007
Create Date: 2026-04-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "strategy_runs",
        sa.Column("run_id", sa.Uuid(), primary_key=True),
        sa.Column("strategy_name", sa.Text(), nullable=False),
        sa.Column("strategy_version", sa.Text(), nullable=False),
        sa.Column("run_type", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("config_snapshot", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_strategy_runs_name_created", "strategy_runs", ["strategy_name", "created_at"]
    )

    op.create_table(
        "engine_budgets",
        sa.Column("engine_name", sa.Text(), primary_key=True),
        sa.Column("engine_version", sa.Text(), nullable=False),
        sa.Column("run_mode", sa.Text(), nullable=False),
        sa.Column("capital_weight", sa.Numeric(20, 8), nullable=False),
        sa.Column("max_gross", sa.Numeric(20, 8), nullable=False),
        sa.Column("max_turnover", sa.Numeric(20, 8), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "combined_portfolio_targets",
        sa.Column("target_id", sa.Uuid(), primary_key=True),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("weights_json", postgresql.JSONB(), nullable=False),
        sa.Column("cash_target_weight", sa.Numeric(20, 8), nullable=False),
        sa.Column("construction_notes", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_combined_targets_as_of", "combined_portfolio_targets", ["as_of"])

    op.create_table(
        "engine_target_contributions",
        sa.Column("contribution_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "combined_target_id",
            sa.Uuid(),
            sa.ForeignKey("combined_portfolio_targets.target_id"),
            nullable=False,
        ),
        sa.Column("engine_name", sa.Text(), nullable=False),
        sa.Column("strategy_run_id", sa.Uuid(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("weights_json", postgresql.JSONB(), nullable=False),
        sa.Column("capital_weight", sa.Numeric(20, 8), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_engine_target_contributions_target",
        "engine_target_contributions",
        ["combined_target_id", "engine_name"],
    )

    op.create_table(
        "order_allocations",
        sa.Column("allocation_id", sa.Uuid(), primary_key=True),
        sa.Column("order_id", sa.Uuid(), nullable=False),
        sa.Column("engine_name", sa.Text(), nullable=False),
        sa.Column("strategy_run_id", sa.Uuid(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("allocated_weight", sa.Numeric(20, 8), nullable=False),
        sa.Column("allocated_notional", sa.Numeric(20, 8)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_order_allocations_order", "order_allocations", ["order_id"])


def downgrade() -> None:
    op.drop_index("ix_order_allocations_order", table_name="order_allocations")
    op.drop_table("order_allocations")
    op.drop_index("ix_engine_target_contributions_target", table_name="engine_target_contributions")
    op.drop_table("engine_target_contributions")
    op.drop_index("ix_combined_targets_as_of", table_name="combined_portfolio_targets")
    op.drop_table("combined_portfolio_targets")
    op.drop_table("engine_budgets")
    op.drop_index("ix_strategy_runs_name_created", table_name="strategy_runs")
    op.drop_table("strategy_runs")
