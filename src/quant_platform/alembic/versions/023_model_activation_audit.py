"""Add model_activation_audit table for operator-visible model promotion history.

Revision ID: 023
Revises: 022
Create Date: 2026-05-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "023"
down_revision: str | None = "022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_activation_audit",
        sa.Column("audit_id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column("strategy_name", sa.Text(), nullable=False),
        sa.Column("to_model_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("from_model_id", sa.UUID(as_uuid=True), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("activated_by", sa.Text(), nullable=False),
    )
    op.create_index(
        "ix_model_activation_audit_strategy_name",
        "model_activation_audit",
        ["strategy_name", "activated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_model_activation_audit_strategy_name", table_name="model_activation_audit")
    op.drop_table("model_activation_audit")
