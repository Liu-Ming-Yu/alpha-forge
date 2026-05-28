"""Add operator_actions audit table.

Records durable evidence of every operator-initiated action: kill switch
activations, manual position overrides, model promotions, and configuration
changes.  This is distinct from the general audit_log (which records
system-generated events) — operator_actions records human decisions.

Revision ID: 012
Revises: 011
Create Date: 2026-04-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "012"
down_revision: str | None = "011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "operator_actions",
        sa.Column("id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("occurred_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("action_type", sa.Text(), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_operator_actions_occurred_at",
        "operator_actions",
        ["occurred_at"],
    )
    op.create_index(
        "ix_operator_actions_action_type",
        "operator_actions",
        ["action_type", "occurred_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_operator_actions_action_type", table_name="operator_actions")
    op.drop_index("ix_operator_actions_occurred_at", table_name="operator_actions")
    op.drop_table("operator_actions")
