"""V2 hardening constraints and readiness state.

Revision ID: 019
Revises: 018
Create Date: 2026-04-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "019"
down_revision: str | None = "018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    _add_quality_check("security_master_records")
    _add_quality_check("corporate_action_events")
    _add_quality_check("universe_snapshots")
    _add_quality_check("bar_datasets")
    _add_quality_check("feature_datasets")

    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_security_master_natural
        ON security_master_records (instrument_id, source, as_of, available_at)
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_symbol_history_natural
        ON symbol_history (instrument_id, symbol, valid_from)
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_universe_snapshot_natural
        ON universe_snapshots (universe_name, as_of, available_at)
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_bar_dataset_natural
        ON bar_datasets (layer, vendor, bar_seconds, start_at, end_at, as_of)
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_feature_dataset_natural
        ON feature_datasets (feature_set_version, as_of, available_at, schema_hash)
        """
    )

    op.create_table(
        "liquidity_profiles_v2",
        sa.Column("profile_id", sa.Uuid(), primary_key=True),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("adv_shares_20d", sa.Numeric(20, 8), nullable=False),
        sa.Column("adv_usd_20d", sa.Numeric(20, 8), nullable=False),
        sa.Column("last_close", sa.Numeric(20, 8), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("quality_status", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_liquidity_profiles_v2_instrument_available",
        "liquidity_profiles_v2",
        ["instrument_id", "available_at"],
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_liquidity_profiles_v2_natural
        ON liquidity_profiles_v2 (instrument_id, source, as_of, available_at)
        """
    )
    _add_quality_check("liquidity_profiles_v2")

    op.create_table(
        "dataset_quorum_evidence",
        sa.Column("evidence_id", sa.Uuid(), primary_key=True),
        sa.Column("dataset_kind", sa.Text(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("vendors_json", postgresql.JSONB(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("required_vendor_count", sa.Integer(), nullable=False),
        sa.Column("max_disagreement_bps", sa.Numeric(20, 8), nullable=False),
        sa.Column("details_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_dataset_quorum_kind_asof",
        "dataset_quorum_evidence",
        ["dataset_kind", "as_of"],
    )

    op.create_table(
        "portfolio_risk_models",
        sa.Column("model_id", sa.Uuid(), primary_key=True),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dataset_id", sa.Uuid()),
        sa.Column("schema_hash", sa.Text(), nullable=False, server_default=""),
        sa.Column("covariance_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("factor_exposures_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("scenarios_json", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_portfolio_risk_models_asof", "portfolio_risk_models", ["as_of"])

    op.create_table(
        "operator_api_keys",
        sa.Column("key_id", sa.Uuid(), primary_key=True),
        sa.Column("key_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
    )
    op.create_index("uq_operator_api_keys_hash", "operator_api_keys", ["key_hash"], unique=True)
    op.execute(
        """
        DO $$
        BEGIN
            ALTER TABLE operator_api_keys
            ADD CONSTRAINT ck_operator_api_key_role
            CHECK (role IN ('viewer', 'operator', 'admin'));
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE operator_api_keys DROP CONSTRAINT IF EXISTS ck_operator_api_key_role")
    op.drop_index("uq_operator_api_keys_hash", table_name="operator_api_keys")
    op.drop_table("operator_api_keys")
    op.drop_index("ix_portfolio_risk_models_asof", table_name="portfolio_risk_models")
    op.drop_table("portfolio_risk_models")
    op.drop_index("ix_dataset_quorum_kind_asof", table_name="dataset_quorum_evidence")
    op.drop_table("dataset_quorum_evidence")
    op.execute(
        "ALTER TABLE liquidity_profiles_v2 "
        "DROP CONSTRAINT IF EXISTS ck_liquidity_profiles_v2_quality_status"
    )
    op.drop_index("uq_liquidity_profiles_v2_natural", table_name="liquidity_profiles_v2")
    op.drop_index(
        "ix_liquidity_profiles_v2_instrument_available",
        table_name="liquidity_profiles_v2",
    )
    op.drop_table("liquidity_profiles_v2")
    op.execute("DROP INDEX IF EXISTS uq_feature_dataset_natural")
    op.execute("DROP INDEX IF EXISTS uq_bar_dataset_natural")
    op.execute("DROP INDEX IF EXISTS uq_universe_snapshot_natural")
    op.execute("DROP INDEX IF EXISTS uq_symbol_history_natural")
    op.execute("DROP INDEX IF EXISTS uq_security_master_natural")
    for table in (
        "feature_datasets",
        "bar_datasets",
        "universe_snapshots",
        "corporate_action_events",
        "security_master_records",
    ):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS ck_{table}_quality_status")


def _add_quality_check(table: str) -> None:
    constraint = f"ck_{table}_quality_status"
    op.execute(
        f"""
        DO $$
        BEGIN
            ALTER TABLE {table}
            ADD CONSTRAINT {constraint}
            CHECK (quality_status IN ('pending', 'approved', 'quarantined'));
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
        """
    )
