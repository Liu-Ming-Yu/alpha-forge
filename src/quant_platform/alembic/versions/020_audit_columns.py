"""Add updated_at / updated_by audit columns to financial-state tables.

Tracks the last modification timestamp and actor for orders, positions,
cash-ledger snapshots, kill-switch state, and operator actions.

Greenfield DB: all new columns use NOT NULL with a server-side default of
now() / '' so no backfill of existing rows is needed.

Revision ID: 020
Revises: 019
Create Date: 2026-04-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "020"
down_revision: str | None = "019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tables that represent mutable financial state and need a full audit trail.
_TABLES_BOTH = (
    "order_intents",
    "account_snapshots",
    "position_snapshots",
    "operator_actions",
)

# kill_switch_state already has updated_at from migration 004; only add updated_by.
_TABLES_UPDATED_BY_ONLY = ("kill_switch_state",)


def upgrade() -> None:
    for table in _TABLES_BOTH:
        op.add_column(
            table,
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )
        op.add_column(
            table,
            sa.Column(
                "updated_by",
                sa.Text(),
                nullable=True,
            ),
        )

    for table in _TABLES_UPDATED_BY_ONLY:
        op.add_column(
            table,
            sa.Column(
                "updated_by",
                sa.Text(),
                nullable=True,
            ),
        )

    # Index on updated_at for efficient "modified since" queries on orders.
    op.create_index(
        "ix_order_intents_updated_at",
        "order_intents",
        ["updated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_order_intents_updated_at", table_name="order_intents")

    for table in _TABLES_UPDATED_BY_ONLY:
        op.drop_column(table, "updated_by")

    for table in reversed(_TABLES_BOTH):
        op.drop_column(table, "updated_by")
        op.drop_column(table, "updated_at")
