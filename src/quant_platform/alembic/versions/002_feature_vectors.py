"""Add feature_vectors table for durable feature storage.

Revision ID: 002
Revises: 001
Create Date: 2026-04-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feature_vectors",
        sa.Column("vector_id", sa.Uuid(), primary_key=True),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_set_version", sa.Text(), nullable=False),
        sa.Column(
            "features",
            postgresql.JSONB(),
            nullable=False,
        ),
        sa.Column("strategy_run_id", sa.Uuid(), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "instrument_id",
            "feature_set_version",
            "as_of",
            name="uq_feature_vector_natural_key",
        ),
    )

    op.create_index(
        "ix_feature_vectors_lookup",
        "feature_vectors",
        ["instrument_id", "feature_set_version", "as_of"],
    )


def downgrade() -> None:
    op.drop_table("feature_vectors")
