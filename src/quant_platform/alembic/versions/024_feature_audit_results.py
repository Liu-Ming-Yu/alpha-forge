"""Add feature_audit_results table for feature-level governance.

Revision ID: 024
Revises: 023
Create Date: 2026-05-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "024"
down_revision: str | None = "023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feature_audit_results",
        sa.Column("audit_id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column("feature_name", sa.Text(), nullable=False),
        sa.Column("feature_version", sa.Text(), nullable=False),
        sa.Column("feature_set_version", sa.Text(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sample_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sample_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("gate_results_json", sa.JSON(), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("schema_hash", sa.Text(), nullable=False),
        sa.Column("code_commit", sa.Text(), nullable=False),
        sa.Column("blockers_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_feature_audit_results_latest",
        "feature_audit_results",
        ["feature_name", "feature_version", "as_of"],
    )
    op.create_index(
        "ix_feature_audit_results_status",
        "feature_audit_results",
        ["status", "passed", "as_of"],
    )


def downgrade() -> None:
    op.drop_index("ix_feature_audit_results_status", table_name="feature_audit_results")
    op.drop_index("ix_feature_audit_results_latest", table_name="feature_audit_results")
    op.drop_table("feature_audit_results")
