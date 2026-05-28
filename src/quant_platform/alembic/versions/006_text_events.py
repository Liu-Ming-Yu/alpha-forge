"""Text event store — Phase 5 LLM text feature layer.

Creates the ``text_events`` table used by ``PostgresTextEventStore`` to
persist pointers to raw text sources (earnings transcripts, SEC filings,
news headlines, guidance revisions) and their object-store artifact URIs.

The ``metadata`` column is JSONB to support arbitrary key-value pairs
without requiring schema changes for new provenance fields.

Idempotency: all ``store_event()`` calls use INSERT ... ON CONFLICT DO NOTHING
on the ``id`` primary key; reads are bounded by the ``occurred_at`` index.

Revision ID: 006
Revises: 005
Create Date: 2026-04-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "text_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("instrument_id", sa.Uuid(), nullable=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_text_events_instrument_id_occurred_at",
        "text_events",
        ["instrument_id", "occurred_at"],
    )
    op.create_index(
        "ix_text_events_event_type_occurred_at",
        "text_events",
        ["event_type", "occurred_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_text_events_event_type_occurred_at",
        table_name="text_events",
    )
    op.drop_index(
        "ix_text_events_instrument_id_occurred_at",
        table_name="text_events",
    )
    op.drop_table("text_events")
