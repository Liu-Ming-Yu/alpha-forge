CREATE TABLE IF NOT EXISTS order_intents (
    order_id        UUID PRIMARY KEY,
    strategy_run_id UUID NOT NULL,
    portfolio_target_id UUID NOT NULL,
    instrument_id   UUID NOT NULL,
    side            TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    order_type      TEXT NOT NULL,
    time_in_force   TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL,
    limit_price     NUMERIC,
    cash_reservation_id UUID,
    is_terminal     BOOLEAN NOT NULL DEFAULT FALSE,
    terminal_reason TEXT,
    CONSTRAINT ck_intent_reservation
        CHECK (side <> 'buy' OR cash_reservation_id IS NOT NULL OR is_terminal = TRUE)
);

CREATE TABLE IF NOT EXISTS fill_events (
    fill_id         UUID PRIMARY KEY,
    order_id        UUID NOT NULL REFERENCES order_intents(order_id),
    broker_order_id TEXT NOT NULL,
    broker_execution_id TEXT,
    instrument_id   UUID NOT NULL,
    side            TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    fill_price      NUMERIC NOT NULL,
    commission      NUMERIC NOT NULL,
    currency        TEXT NOT NULL,
    executed_at     TIMESTAMPTZ NOT NULL,
    received_at     TIMESTAMPTZ NOT NULL,
    supersedes_id   UUID
);

CREATE TABLE IF NOT EXISTS account_snapshots (
    snapshot_id     UUID PRIMARY KEY,
    as_of           TIMESTAMPTZ NOT NULL,
    settled_cash    NUMERIC NOT NULL,
    unsettled_cash  NUMERIC NOT NULL,
    reserved_cash   NUMERIC NOT NULL,
    available_cash  NUMERIC NOT NULL,
    net_asset_value NUMERIC NOT NULL,
    source          TEXT NOT NULL DEFAULT 'broker'
);

CREATE TABLE IF NOT EXISTS position_snapshots (
    snapshot_id     UUID NOT NULL REFERENCES account_snapshots(snapshot_id),
    instrument_id   UUID NOT NULL,
    quantity        INTEGER NOT NULL,
    average_cost    NUMERIC NOT NULL,
    market_price    NUMERIC NOT NULL,
    market_value    NUMERIC NOT NULL,
    unrealised_pnl  NUMERIC NOT NULL,
    as_of           TIMESTAMPTZ NOT NULL,
    source          TEXT NOT NULL DEFAULT 'broker',
    PRIMARY KEY (snapshot_id, instrument_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    entry_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type      TEXT NOT NULL,
    event_payload   JSONB NOT NULL,
    context         JSONB NOT NULL DEFAULT '{}',
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feature_vectors (
    vector_id            UUID PRIMARY KEY,
    instrument_id        UUID NOT NULL,
    as_of                TIMESTAMPTZ NOT NULL,
    feature_set_version  TEXT NOT NULL,
    features             JSONB NOT NULL,
    strategy_run_id      UUID NOT NULL,
    artifact_uri         TEXT NOT NULL DEFAULT '',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_feature_vector_natural_key
        UNIQUE (instrument_id, feature_set_version, as_of)
);

CREATE TABLE IF NOT EXISTS registered_models (
    model_id           UUID PRIMARY KEY,
    strategy_name      TEXT NOT NULL,
    model_version      TEXT NOT NULL,
    feature_set_version TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    active             BOOLEAN NOT NULL DEFAULT TRUE,
    metadata_json      JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS feature_jobs (
    job_id               UUID PRIMARY KEY,
    model_id             UUID NOT NULL REFERENCES registered_models(model_id) ON DELETE CASCADE,
    strategy_name        TEXT NOT NULL,
    feature_set_version  TEXT NOT NULL,
    interval_seconds     FLOAT NOT NULL,
    next_run_at          TIMESTAMPTZ NOT NULL,
    last_run_at          TIMESTAMPTZ,
    enabled              BOOLEAN NOT NULL DEFAULT TRUE,
    consecutive_failures INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS kill_switch_state (
    id           TEXT PRIMARY KEY DEFAULT 'default',
    active       BOOLEAN NOT NULL DEFAULT FALSE,
    reason       TEXT,
    activated_at TIMESTAMPTZ,
    activated_by TEXT,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO kill_switch_state (id, active) VALUES ('default', false) ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS pending_settlement_lots (
    lot_id          TEXT PRIMARY KEY,
    fill_id         TEXT NOT NULL,
    order_id        TEXT NOT NULL,
    instrument_id   TEXT NOT NULL,
    trade_date      DATE NOT NULL,
    settlement_date DATE NOT NULL,
    gross_proceeds  NUMERIC(20, 8) NOT NULL,
    commission      NUMERIC(20, 8) NOT NULL,
    net_proceeds    NUMERIC(20, 8) NOT NULL,
    currency        TEXT NOT NULL,
    run_id          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS completed_order_hints (
    order_id     TEXT PRIMARY KEY,
    run_id       TEXT,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS text_events (
    id            UUID PRIMARY KEY,
    instrument_id UUID,
    event_type    TEXT NOT NULL,
    occurred_at   TIMESTAMPTZ NOT NULL,
    source_uri    TEXT NOT NULL,
    artifact_uri  TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nav_snapshots (
    snapshot_id     UUID PRIMARY KEY,
    strategy_run_id UUID NOT NULL,
    as_of           TIMESTAMPTZ NOT NULL,
    net_asset_value NUMERIC(20, 8) NOT NULL,
    gross_exposure  NUMERIC(20, 8) NOT NULL DEFAULT 0,
    cash            NUMERIC(20, 8) NOT NULL DEFAULT 0,
    source          TEXT NOT NULL DEFAULT 'runtime',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS text_signal_ic_observations (
    strategy_name TEXT NOT NULL,
    as_of         TIMESTAMPTZ NOT NULL,
    daily_ic      FLOAT NOT NULL,
    observations  INTEGER NOT NULL DEFAULT 1,
    metadata_json JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (strategy_name, as_of)
);

CREATE TABLE IF NOT EXISTS signal_gate_observations (
    signal_name   TEXT NOT NULL,
    signal_type   TEXT NOT NULL,
    as_of         TIMESTAMPTZ NOT NULL,
    daily_ic      FLOAT NOT NULL,
    observations  INTEGER NOT NULL DEFAULT 1,
    drawdown      FLOAT NOT NULL DEFAULT 0,
    turnover      FLOAT NOT NULL DEFAULT 0,
    metadata_json JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (signal_name, signal_type, as_of)
);

CREATE TABLE IF NOT EXISTS shadow_paper_parity_observations (
    parity_id                  UUID PRIMARY KEY,
    signal_name                TEXT NOT NULL,
    signal_type                TEXT NOT NULL,
    trading_day                DATE NOT NULL,
    as_of                      TIMESTAMPTZ NOT NULL,
    instruments_compared       INTEGER NOT NULL CHECK (instruments_compared >= 0),
    missing_instruments        INTEGER NOT NULL CHECK (missing_instruments >= 0),
    max_target_weight_diff_bps FLOAT NOT NULL CHECK (max_target_weight_diff_bps >= 0),
    order_side_mismatches      INTEGER NOT NULL CHECK (order_side_mismatches >= 0),
    metadata_json              JSONB NOT NULL DEFAULT '{}',
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feature_audit_results (
    audit_id            UUID PRIMARY KEY,
    feature_name        TEXT NOT NULL,
    feature_version     TEXT NOT NULL,
    feature_set_version TEXT NOT NULL,
    as_of               TIMESTAMPTZ NOT NULL,
    sample_start        TIMESTAMPTZ NOT NULL,
    sample_end          TIMESTAMPTZ NOT NULL,
    status              TEXT NOT NULL,
    passed              BOOLEAN NOT NULL,
    metrics_json        JSONB NOT NULL,
    gate_results_json   JSONB NOT NULL,
    artifact_uri        TEXT NOT NULL,
    schema_hash         TEXT NOT NULL,
    code_commit         TEXT NOT NULL,
    blockers_json       JSONB NOT NULL DEFAULT '[]',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runtime_heartbeats (
    component  TEXT PRIMARY KEY,
    as_of      TIMESTAMPTZ NOT NULL,
    status     TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS broker_health_observations (
    observation_id    UUID PRIMARY KEY,
    observed_at       TIMESTAMPTZ NOT NULL,
    status            TEXT NOT NULL,
    latency_ms        FLOAT NOT NULL,
    last_heartbeat_at TIMESTAMPTZ,
    detail            TEXT NOT NULL DEFAULT '',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS broker_smoke_observations (
    observation_id     UUID PRIMARY KEY,
    observed_at        TIMESTAMPTZ NOT NULL,
    status             TEXT NOT NULL,
    host               TEXT NOT NULL,
    port               INTEGER NOT NULL,
    client_id          INTEGER NOT NULL,
    latency_ms         FLOAT NOT NULL,
    account_status     TEXT NOT NULL,
    positions_status   TEXT NOT NULL,
    open_orders_status TEXT NOT NULL,
    detail             TEXT NOT NULL DEFAULT '',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS paper_lifecycle_observations (
    observation_id          UUID PRIMARY KEY,
    observed_at             TIMESTAMPTZ NOT NULL,
    status                  TEXT NOT NULL,
    host                    TEXT NOT NULL,
    port                    INTEGER NOT NULL,
    client_id               INTEGER NOT NULL,
    instrument_id           UUID NOT NULL,
    broker_order_id         TEXT NOT NULL,
    max_notional_usd        NUMERIC(20, 8) NOT NULL,
    limit_price             NUMERIC(20, 8) NOT NULL,
    quantity                INTEGER NOT NULL,
    ack_status              TEXT NOT NULL,
    cancel_status           TEXT NOT NULL,
    stale_open_order_count  INTEGER NOT NULL,
    detail                  TEXT NOT NULL DEFAULT '',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS strategy_runs (
    run_id           UUID PRIMARY KEY,
    strategy_name    TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    run_type         TEXT NOT NULL,
    status           TEXT NOT NULL,
    config_snapshot  JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL,
    started_at       TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS engine_budgets (
    engine_name    TEXT PRIMARY KEY,
    engine_version TEXT NOT NULL,
    run_mode       TEXT NOT NULL,
    capital_weight NUMERIC(20, 8) NOT NULL,
    max_gross      NUMERIC(20, 8) NOT NULL,
    max_turnover   NUMERIC(20, 8) NOT NULL,
    enabled        BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS combined_portfolio_targets (
    target_id           UUID PRIMARY KEY,
    as_of               TIMESTAMPTZ NOT NULL,
    weights_json        JSONB NOT NULL,
    cash_target_weight  NUMERIC(20, 8) NOT NULL,
    construction_notes  JSONB NOT NULL DEFAULT '[]',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS engine_target_contributions (
    contribution_id  UUID PRIMARY KEY,
    combined_target_id UUID NOT NULL REFERENCES combined_portfolio_targets(target_id),
    engine_name      TEXT NOT NULL,
    strategy_run_id  UUID NOT NULL,
    as_of            TIMESTAMPTZ NOT NULL,
    weights_json     JSONB NOT NULL,
    capital_weight   NUMERIC(20, 8) NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS order_allocations (
    allocation_id    UUID PRIMARY KEY,
    order_id         UUID NOT NULL,
    engine_name      TEXT NOT NULL,
    strategy_run_id  UUID NOT NULL,
    instrument_id    UUID NOT NULL,
    allocated_weight NUMERIC(20, 8) NOT NULL,
    allocated_notional NUMERIC(20, 8),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_fill_events_order_id
    ON fill_events(order_id);
CREATE INDEX IF NOT EXISTS ix_order_intents_strategy_run
    ON order_intents(strategy_run_id);
CREATE INDEX IF NOT EXISTS ix_account_snapshots_as_of
    ON account_snapshots(as_of);
CREATE INDEX IF NOT EXISTS ix_audit_log_event_type
    ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS ix_audit_log_recorded_at
    ON audit_log(recorded_at);
CREATE INDEX IF NOT EXISTS ix_feature_vectors_lookup
    ON feature_vectors(instrument_id, feature_set_version, as_of);
CREATE INDEX IF NOT EXISTS ix_registered_models_strategy
    ON registered_models(strategy_name, active);
CREATE INDEX IF NOT EXISTS ix_feature_jobs_due
    ON feature_jobs(enabled, next_run_at);
CREATE INDEX IF NOT EXISTS ix_pending_settlement_lots_settlement_date
    ON pending_settlement_lots(settlement_date);
CREATE INDEX IF NOT EXISTS ix_pending_settlement_lots_run_id
    ON pending_settlement_lots(run_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_fill_events_broker_execution
    ON fill_events(broker_order_id, broker_execution_id)
    WHERE broker_execution_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_text_events_instrument_id_occurred_at
    ON text_events(instrument_id, occurred_at);
CREATE INDEX IF NOT EXISTS ix_text_events_event_type_occurred_at
    ON text_events(event_type, occurred_at);
CREATE INDEX IF NOT EXISTS ix_nav_snapshots_strategy_run_as_of
    ON nav_snapshots(strategy_run_id, as_of);
CREATE INDEX IF NOT EXISTS ix_text_signal_ic_strategy_as_of
    ON text_signal_ic_observations(strategy_name, as_of);
CREATE INDEX IF NOT EXISTS ix_signal_gate_type_name_as_of
    ON signal_gate_observations(signal_type, signal_name, as_of);
CREATE UNIQUE INDEX IF NOT EXISTS ix_shadow_paper_parity_signal_day
    ON shadow_paper_parity_observations(signal_type, signal_name, trading_day);
CREATE INDEX IF NOT EXISTS ix_shadow_paper_parity_signal_latest
    ON shadow_paper_parity_observations(signal_type, signal_name, as_of);
CREATE INDEX IF NOT EXISTS ix_feature_audit_results_latest
    ON feature_audit_results(feature_name, feature_version, as_of);
CREATE INDEX IF NOT EXISTS ix_feature_audit_results_status
    ON feature_audit_results(status, passed, as_of);
CREATE INDEX IF NOT EXISTS ix_broker_health_observed_at
    ON broker_health_observations(observed_at);
CREATE INDEX IF NOT EXISTS ix_broker_smoke_observed_at
    ON broker_smoke_observations(observed_at);
CREATE INDEX IF NOT EXISTS ix_paper_lifecycle_observed_at
    ON paper_lifecycle_observations(observed_at);
CREATE INDEX IF NOT EXISTS ix_strategy_runs_name_created
    ON strategy_runs(strategy_name, created_at);
CREATE INDEX IF NOT EXISTS ix_combined_targets_as_of
    ON combined_portfolio_targets(as_of);
CREATE INDEX IF NOT EXISTS ix_engine_target_contributions_target
    ON engine_target_contributions(combined_target_id, engine_name);
CREATE INDEX IF NOT EXISTS ix_order_allocations_order
    ON order_allocations(order_id);
