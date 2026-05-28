"""Allow open sell intents without cash reservations.

Revision ID: 028
Revises: 027
Create Date: 2026-05-18
"""

from collections.abc import Sequence

from alembic import op

revision: str = "028"
down_revision: str | None = "027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE order_intents DROP CONSTRAINT IF EXISTS ck_intent_reservation")
    op.execute(
        """
        DO $$
        BEGIN
            ALTER TABLE order_intents
            ADD CONSTRAINT ck_intent_reservation
            CHECK (
                side <> 'buy'
                OR cash_reservation_id IS NOT NULL
                OR is_terminal = TRUE
            );
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE order_intents DROP CONSTRAINT IF EXISTS ck_intent_reservation")
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
