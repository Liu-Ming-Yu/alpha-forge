"""Add model-registry tables for durable strategy/model/feature-job tracking.

Revision ID: 003
Revises: 002
Create Date: 2026-04-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "registered_models",
        sa.Column("model_id", sa.Uuid(), primary_key=True),
        sa.Column("strategy_name", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("feature_set_version", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_index(
        "ix_registered_models_strategy",
        "registered_models",
        ["strategy_name", "active"],
    )
    op.create_index(
        "ix_registered_models_strategy_created",
        "registered_models",
        ["strategy_name", "created_at"],
    )

    op.create_table(
        "feature_jobs",
        sa.Column("job_id", sa.Uuid(), primary_key=True),
        sa.Column("model_id", sa.Uuid(), nullable=False),
        sa.Column("strategy_name", sa.Text(), nullable=False),
        sa.Column("feature_set_version", sa.Text(), nullable=False),
        sa.Column("interval_seconds", sa.Float(), nullable=False),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "consecutive_failures",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.ForeignKeyConstraint(["model_id"], ["registered_models.model_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_feature_jobs_due", "feature_jobs", ["enabled", "next_run_at"])


def downgrade() -> None:
    op.drop_index("ix_feature_jobs_due", table_name="feature_jobs")
    op.drop_table("feature_jobs")
    op.drop_index("ix_registered_models_strategy_created", table_name="registered_models")
    op.drop_index("ix_registered_models_strategy", table_name="registered_models")
    op.drop_table("registered_models")
