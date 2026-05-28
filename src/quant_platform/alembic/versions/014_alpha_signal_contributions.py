"""Alpha signal contribution attribution.

Revision ID: 014
Revises: 013
Create Date: 2026-04-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "014"
down_revision: str | None = "013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "signal_contributions",
        sa.Column("contribution_id", sa.UUID(), nullable=False),
        sa.Column("score_id", sa.UUID(), nullable=False),
        sa.Column("strategy_run_id", sa.UUID(), nullable=False),
        sa.Column("instrument_id", sa.UUID(), nullable=False),
        sa.Column("as_of", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_model_version", sa.Text(), nullable=False),
        sa.Column("raw_score", sa.Float(), nullable=False),
        sa.Column("normalized_score", sa.Float(), nullable=False),
        sa.Column("blend_weight", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("feature_vector_id", sa.UUID(), nullable=True),
        sa.Column("promotion_state", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("contribution_id"),
    )
    op.create_index(
        "ix_signal_contributions_score_id",
        "signal_contributions",
        ["score_id"],
    )
    op.create_index(
        "ix_signal_contributions_run_instrument_as_of",
        "signal_contributions",
        ["strategy_run_id", "instrument_id", "as_of"],
    )
    op.create_index(
        "ix_signal_contributions_source_as_of",
        "signal_contributions",
        ["source", "as_of"],
    )

    op.add_column("text_events", sa.Column("provider", sa.Text(), nullable=True))
    op.add_column("text_events", sa.Column("dedupe_key", sa.Text(), nullable=True))
    op.add_column("text_events", sa.Column("content_hash", sa.Text(), nullable=True))
    op.add_column("text_events", sa.Column("ingestion_status", sa.Text(), nullable=True))
    op.add_column(
        "text_events", sa.Column("source_published_at", sa.TIMESTAMP(timezone=True), nullable=True)
    )
    op.create_index(
        "ix_text_events_provider_status", "text_events", ["provider", "ingestion_status"]
    )
    op.create_index("ix_text_events_dedupe_key", "text_events", ["dedupe_key"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_text_events_dedupe_key", table_name="text_events")
    op.drop_index("ix_text_events_provider_status", table_name="text_events")
    op.drop_column("text_events", "source_published_at")
    op.drop_column("text_events", "ingestion_status")
    op.drop_column("text_events", "content_hash")
    op.drop_column("text_events", "dedupe_key")
    op.drop_column("text_events", "provider")
    op.drop_index("ix_signal_contributions_source_as_of", table_name="signal_contributions")
    op.drop_index(
        "ix_signal_contributions_run_instrument_as_of",
        table_name="signal_contributions",
    )
    op.drop_index("ix_signal_contributions_score_id", table_name="signal_contributions")
    op.drop_table("signal_contributions")
