"""Durable kill-switch state table.

Revision ID: 004
Revises: 003
Create Date: 2026-04-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "kill_switch_state",
        sa.Column("id", sa.Text(), primary_key=True, server_default=sa.text("'default'")),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("activated_by", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.execute(
        "INSERT INTO kill_switch_state (id, active) VALUES ('default', false) "
        "ON CONFLICT (id) DO NOTHING"
    )


def downgrade() -> None:
    op.drop_table("kill_switch_state")
