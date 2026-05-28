"""Add available_at to durable feature vectors.

Revision ID: 022
Revises: 021
Create Date: 2026-05-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "022"
down_revision: str | None = "021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "feature_vectors",
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute("UPDATE feature_vectors SET available_at = as_of WHERE available_at IS NULL")
    op.alter_column("feature_vectors", "available_at", nullable=False)
    op.create_index(
        "ix_feature_vectors_available_lookup",
        "feature_vectors",
        ["instrument_id", "feature_set_version", "available_at", "as_of"],
    )


def downgrade() -> None:
    op.drop_index("ix_feature_vectors_available_lookup", table_name="feature_vectors")
    op.drop_column("feature_vectors", "available_at")
