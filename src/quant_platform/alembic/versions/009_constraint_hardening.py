"""Constraint hardening — broker execution idempotency, kill-switch singleton,
settlement lot UUID types, and order-intent reservation invariant.

Revision ID: 009
Revises: 008
Create Date: 2026-04-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "009"
down_revision: str | None = "008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # B-3a: broker execution identity for idempotent partial-fill ingestion.
    op.execute("ALTER TABLE fill_events ADD COLUMN IF NOT EXISTS broker_execution_id TEXT")
    op.create_index(
        "uq_fill_events_broker_execution",
        "fill_events",
        ["broker_order_id", "broker_execution_id"],
        unique=True,
        if_not_exists=True,
        postgresql_where=sa.text("broker_execution_id IS NOT NULL"),
    )

    # B-3b: enforce kill-switch singleton row
    op.execute(
        """
        DO $$
        BEGIN
            ALTER TABLE kill_switch_state
            ADD CONSTRAINT ck_kill_switch_singleton CHECK (id = 'default');
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
        """
    )

    # B-3c: pending_settlement_lots lot_id → uuid (was text PK)
    op.execute(
        "ALTER TABLE pending_settlement_lots ALTER COLUMN lot_id TYPE uuid USING lot_id::uuid"
    )

    # B-3c: completed_order_hints order_id → uuid (was text PK)
    op.execute(
        "ALTER TABLE completed_order_hints ALTER COLUMN order_id TYPE uuid USING order_id::uuid"
    )

    # B-3d: order_intents must have a cash_reservation_id unless terminal
    op.execute(
        """
        DO $$
        BEGIN
            ALTER TABLE order_intents
            ADD CONSTRAINT ck_intent_reservation
            CHECK (cash_reservation_id IS NOT NULL OR is_terminal = TRUE);
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE order_intents DROP CONSTRAINT IF EXISTS ck_intent_reservation")
    op.execute(
        "ALTER TABLE completed_order_hints ALTER COLUMN order_id TYPE text USING order_id::text"
    )
    op.execute(
        "ALTER TABLE pending_settlement_lots ALTER COLUMN lot_id TYPE text USING lot_id::text"
    )
    op.execute("ALTER TABLE kill_switch_state DROP CONSTRAINT IF EXISTS ck_kill_switch_singleton")
    op.drop_index("uq_fill_events_broker_execution", table_name="fill_events", if_exists=True)
    op.execute("ALTER TABLE fill_events DROP COLUMN IF EXISTS broker_execution_id")
