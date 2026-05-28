"""V2 institutional production state.

Revision ID: 018
Revises: 017
Create Date: 2026-04-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "018"
down_revision: str | None = "017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "security_master_records",
        sa.Column("record_id", sa.Uuid(), primary_key=True),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("exchange", sa.Text(), nullable=False),
        sa.Column("asset_class", sa.Text(), nullable=False),
        sa.Column("currency", sa.Text(), nullable=False),
        sa.Column("lot_size", sa.Integer(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("sector", sa.Text()),
        sa.Column("primary_exchange", sa.Text(), nullable=False, server_default=""),
        sa.Column("country", sa.Text(), nullable=False, server_default="US"),
        sa.Column("identifiers_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
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
        "ix_security_master_instrument_asof",
        "security_master_records",
        ["instrument_id", "as_of"],
    )
    op.create_index(
        "ix_security_master_symbol_asof",
        "security_master_records",
        ["symbol", "as_of"],
    )

    op.create_table(
        "symbol_history",
        sa.Column("history_id", sa.Uuid(), primary_key=True),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("valid_from", sa.Date(), nullable=False),
        sa.Column("valid_to", sa.Date()),
        sa.Column("source", sa.Text(), nullable=False),
    )
    op.create_index("ix_symbol_history_symbol", "symbol_history", ["symbol", "valid_from"])

    op.create_table(
        "corporate_action_events",
        sa.Column("event_id", sa.Uuid(), primary_key=True),
        sa.Column("action_id", sa.Uuid(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("action_type", sa.Text(), nullable=False),
        sa.Column("ex_date", sa.Date(), nullable=False),
        sa.Column("record_date", sa.Date(), nullable=False),
        sa.Column("pay_date", sa.Date(), nullable=False),
        sa.Column("ratio", sa.Numeric(20, 8), nullable=False),
        sa.Column("cash_amount", sa.Numeric(20, 8), nullable=False),
        sa.Column("currency", sa.Text(), nullable=False),
        sa.Column("supersedes_id", sa.Uuid()),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quality_status", sa.Text(), nullable=False),
    )
    op.create_index(
        "ix_corporate_action_events_instrument_available",
        "corporate_action_events",
        ["instrument_id", "available_at"],
    )

    op.create_table(
        "universe_snapshots",
        sa.Column("snapshot_id", sa.Uuid(), primary_key=True),
        sa.Column("universe_name", sa.Text(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("instrument_ids_json", postgresql.JSONB(), nullable=False),
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
        "ix_universe_snapshots_name_available",
        "universe_snapshots",
        ["universe_name", "available_at"],
    )

    op.create_table(
        "bar_datasets",
        sa.Column("dataset_id", sa.Uuid(), primary_key=True),
        sa.Column("layer", sa.Text(), nullable=False),
        sa.Column("vendor", sa.Text(), nullable=False),
        sa.Column("bar_seconds", sa.Integer(), nullable=False),
        sa.Column("start_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("schema_hash", sa.Text(), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("quality_status", sa.Text(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_bar_datasets_layer_available", "bar_datasets", ["layer", "available_at"])
    op.create_index("ix_bar_datasets_vendor_available", "bar_datasets", ["vendor", "available_at"])

    op.create_table(
        "feature_datasets",
        sa.Column("dataset_id", sa.Uuid(), primary_key=True),
        sa.Column("feature_set_version", sa.Text(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("schema_hash", sa.Text(), nullable=False),
        sa.Column("source_dataset_ids_json", postgresql.JSONB(), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("quality_status", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_feature_datasets_version_available",
        "feature_datasets",
        ["feature_set_version", "available_at"],
    )

    op.create_table(
        "model_artifacts_v2",
        sa.Column("artifact_id", sa.Uuid(), primary_key=True),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("artifact_hash", sa.Text(), nullable=False),
        sa.Column("feature_schema_hash", sa.Text(), nullable=False),
        sa.Column("training_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("training_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("promotion_state", sa.Text(), nullable=False),
        sa.Column("rollback_artifact_id", sa.Uuid()),
    )
    op.create_index(
        "ix_model_artifacts_v2_name_version",
        "model_artifacts_v2",
        ["model_name", "model_version"],
    )

    op.create_table(
        "model_cards",
        sa.Column("card_id", sa.Uuid(), primary_key=True),
        sa.Column("artifact_id", sa.Uuid(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("owner", sa.Text(), nullable=False),
        sa.Column("intended_use", sa.Text(), nullable=False),
        sa.Column("metrics_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("risk_notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "alpha_readiness_reports",
        sa.Column("report_id", sa.Uuid(), primary_key=True),
        sa.Column("alpha_source", sa.Text(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("promotion_state", sa.Text(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("metrics_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("drift_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("rollback_target", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index(
        "ix_alpha_readiness_source_asof",
        "alpha_readiness_reports",
        ["alpha_source", "as_of"],
    )

    op.create_table(
        "portfolio_risk_snapshots",
        sa.Column("snapshot_id", sa.Uuid(), primary_key=True),
        sa.Column("strategy_run_id", sa.Uuid(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("covariance_dataset_id", sa.Uuid()),
        sa.Column("factor_exposures_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("stress_results_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("cvar", sa.Numeric(20, 8)),
        sa.Column("gross_exposure", sa.Numeric(20, 8), nullable=False),
        sa.Column("net_exposure", sa.Numeric(20, 8), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
    )
    op.create_index(
        "ix_portfolio_risk_snapshots_run_asof",
        "portfolio_risk_snapshots",
        ["strategy_run_id", "as_of"],
    )

    op.create_table(
        "order_state_events",
        sa.Column("event_id", sa.Uuid(), primary_key=True),
        sa.Column("order_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text()),
        sa.Column("broker_order_id", sa.Text()),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False, server_default="{}"),
    )
    op.create_index(
        "ix_order_state_events_order", "order_state_events", ["order_id", "occurred_at"]
    )
    op.create_index(
        "uq_order_state_events_idempotency",
        "order_state_events",
        ["idempotency_key"],
        unique=True,
    )

    op.create_table(
        "execution_quality_reports",
        sa.Column("report_id", sa.Uuid(), primary_key=True),
        sa.Column("order_id", sa.Uuid(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("venue", sa.Text(), nullable=False),
        sa.Column("tactic", sa.Text(), nullable=False),
        sa.Column("arrival_price", sa.Numeric(20, 8)),
        sa.Column("decision_price", sa.Numeric(20, 8)),
        sa.Column("vwap", sa.Numeric(20, 8)),
        sa.Column("fill_price", sa.Numeric(20, 8)),
        sa.Column("slippage_bps", sa.Numeric(20, 8)),
        sa.Column("participation_rate", sa.Numeric(20, 8)),
        sa.Column("passed", sa.Boolean(), nullable=False),
    )
    op.create_index(
        "ix_execution_quality_order_asof",
        "execution_quality_reports",
        ["order_id", "as_of"],
    )

    op.create_table(
        "runbook_evidence",
        sa.Column("evidence_id", sa.Uuid(), primary_key=True),
        sa.Column("runbook_name", sa.Text(), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False, server_default=""),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False, server_default="{}"),
    )

    op.create_table(
        "alert_events",
        sa.Column("alert_id", sa.Uuid(), primary_key=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("severity", sa.Text(), nullable=False),
        sa.Column("component", sa.Text(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False, server_default="{}"),
    )
    op.create_index("ix_alert_events_component_time", "alert_events", ["component", "occurred_at"])

    op.create_table(
        "readiness_snapshots",
        sa.Column("snapshot_id", sa.Uuid(), primary_key=True),
        sa.Column("profile", sa.Text(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("checks_json", postgresql.JSONB(), nullable=False),
    )
    op.create_index(
        "ix_readiness_snapshots_profile_time",
        "readiness_snapshots",
        ["profile", "generated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_readiness_snapshots_profile_time", table_name="readiness_snapshots")
    op.drop_table("readiness_snapshots")
    op.drop_index("ix_alert_events_component_time", table_name="alert_events")
    op.drop_table("alert_events")
    op.drop_table("runbook_evidence")
    op.drop_index("ix_execution_quality_order_asof", table_name="execution_quality_reports")
    op.drop_table("execution_quality_reports")
    op.drop_index("uq_order_state_events_idempotency", table_name="order_state_events")
    op.drop_index("ix_order_state_events_order", table_name="order_state_events")
    op.drop_table("order_state_events")
    op.drop_index("ix_portfolio_risk_snapshots_run_asof", table_name="portfolio_risk_snapshots")
    op.drop_table("portfolio_risk_snapshots")
    op.drop_index("ix_alpha_readiness_source_asof", table_name="alpha_readiness_reports")
    op.drop_table("alpha_readiness_reports")
    op.drop_table("model_cards")
    op.drop_index("ix_model_artifacts_v2_name_version", table_name="model_artifacts_v2")
    op.drop_table("model_artifacts_v2")
    op.drop_index("ix_feature_datasets_version_available", table_name="feature_datasets")
    op.drop_table("feature_datasets")
    op.drop_index("ix_bar_datasets_vendor_available", table_name="bar_datasets")
    op.drop_index("ix_bar_datasets_layer_available", table_name="bar_datasets")
    op.drop_table("bar_datasets")
    op.drop_index("ix_universe_snapshots_name_available", table_name="universe_snapshots")
    op.drop_table("universe_snapshots")
    op.drop_index(
        "ix_corporate_action_events_instrument_available",
        table_name="corporate_action_events",
    )
    op.drop_table("corporate_action_events")
    op.drop_index("ix_symbol_history_symbol", table_name="symbol_history")
    op.drop_table("symbol_history")
    op.drop_index("ix_security_master_symbol_asof", table_name="security_master_records")
    op.drop_index("ix_security_master_instrument_asof", table_name="security_master_records")
    op.drop_table("security_master_records")
