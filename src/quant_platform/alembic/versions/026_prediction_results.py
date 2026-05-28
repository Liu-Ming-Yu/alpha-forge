"""Add prediction_results for governed forecast evidence.

Revision ID: 026
Revises: 025
Create Date: 2026-05-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "026"
down_revision: str | None = "025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "metric_rollup_snapshots",
        sa.Column("snapshot_id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column("metric_name", sa.Text(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rollup_window", sa.Text(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("labels_json", postgresql.JSONB(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_metric_rollup_snapshots_latest",
        "metric_rollup_snapshots",
        ["metric_name", "as_of"],
    )
    op.create_table(
        "prediction_results",
        sa.Column("prediction_id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column("strategy_run_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("instrument_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon", sa.Text(), nullable=False),
        sa.Column("expected_return", sa.Float(), nullable=False),
        sa.Column("rank_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("feature_schema_hash", sa.Text(), nullable=False),
        sa.Column("calibration_bucket", sa.Text(), nullable=False),
        sa.Column("blockers_json", postgresql.JSONB(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_prediction_confidence"),
    )
    op.create_index(
        "ix_prediction_results_source_latest",
        "prediction_results",
        ["source", "model_version", "as_of"],
    )
    op.create_index(
        "ix_prediction_results_run_instrument",
        "prediction_results",
        ["strategy_run_id", "instrument_id", "as_of"],
    )


def downgrade() -> None:
    op.drop_index("ix_prediction_results_run_instrument", table_name="prediction_results")
    op.drop_index("ix_prediction_results_source_latest", table_name="prediction_results")
    op.drop_table("prediction_results")
    op.drop_index("ix_metric_rollup_snapshots_latest", table_name="metric_rollup_snapshots")
    op.drop_table("metric_rollup_snapshots")
