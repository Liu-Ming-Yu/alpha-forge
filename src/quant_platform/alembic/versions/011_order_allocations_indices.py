"""Add indices to order_allocations for engine/run and instrument queries.

Without these indices, queries by (engine_name, strategy_run_id) and
instrument_id require sequential scans on order_allocations, which is a
hot table during live execution and post-trade attribution.

Revision ID: 011
Revises: 010
Create Date: 2026-04-28
"""

from collections.abc import Sequence

from alembic import op

revision: str = "011"
down_revision: str | None = "010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_order_allocations_engine_run",
        "order_allocations",
        ["engine_name", "strategy_run_id"],
    )
    op.create_index(
        "ix_order_allocations_instrument",
        "order_allocations",
        ["instrument_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_order_allocations_instrument", table_name="order_allocations")
    op.drop_index("ix_order_allocations_engine_run", table_name="order_allocations")
