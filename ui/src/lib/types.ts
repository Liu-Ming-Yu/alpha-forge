/**
 * TypeScript mirrors of the operator-API payloads the console renders.
 * Deeply-nested / free-form blobs (research campaigns, audit rows) are kept
 * loose on purpose — the UI renders them defensively.
 */

export interface ConsoleInfo {
  api_version: string;
  requires_auth: boolean;
  product: string;
  run_modes: string[];
  execution_backends: string[];
  profiles: string[];
}

export interface WriteControls {
  kill_switch_clear: boolean;
  trading: boolean;
  model_promotion: boolean;
  alpha_promotion: boolean;
}

export interface Capabilities {
  api_version: string;
  auth: { mode: string; v2_operator_auth: boolean; roles_advertised: string[] };
  features: Record<string, boolean | string>;
  write_controls: WriteControls;
  unsupported_features: string[];
}

export interface BrokerHealth {
  connected: boolean;
  kill_switch_active: boolean;
  kill_switch_reason: string;
  orders_submitted_this_session: number;
  throttle_tokens_available: number;
  status: string;
  detail: string;
  latency_ms: number | null;
  last_heartbeat_at: string | null;
}

export interface CashStatus {
  as_of: string;
  settled_cash: number | string;
  unsettled_cash: number | string;
  reserved_cash: number | string;
  available_cash: number | string;
}

export interface RegimeState {
  as_of: string;
  label: string;
  gross_exposure_scale: number;
  trend_z: number;
  annualized_vol: number;
  breadth_pct: number;
}

export interface CombinedExposure {
  as_of: string;
  enabled_engines: number;
  allocated_capital_weight: number;
  reserved_cash_weight: number;
}

export interface EngineBudget {
  engine_name: string;
  engine_version: string;
  run_mode: string;
  capital_weight: number;
  max_gross: number;
  max_turnover: number;
  enabled: boolean;
}

export interface StrategyRun {
  run_id: string;
  status?: string;
  mode?: string;
  started_at?: string;
  [k: string]: unknown;
}

export interface StrategyLifecycle {
  engine_name: string;
  engine_version: string;
  health: string;
  days_active: number;
  rolling_sharpe_90d: number;
  rolling_ic_60d: number;
  max_drawdown_realized: number;
  max_drawdown_limit: number;
  slippage_ratio: number;
  cycles_completed: number;
  recommendation: string;
}

export interface SignalDecay {
  as_of: string;
  engine_name: string;
  signals_generated: number;
  mean_score: number;
  score_dispersion: number;
  top_quintile_count: number;
  bottom_quintile_count: number;
  turnover_rate: number;
}

export interface SignalContribution {
  score_id: string;
  strategy_run_id: string;
  instrument_id: string;
  as_of: string;
  source: string;
  source_model_version: string;
  raw_score: number;
  normalized_score: number;
  blend_weight: number;
  confidence: number;
  promotion_state: string;
}

export interface ForecastEvidence {
  source: string;
  model_version: string;
  as_of: string;
  horizon: string;
  observations: number;
  mean_confidence: number;
  latest_prediction_at: string | null;
  stale: boolean;
  passed: boolean;
  blockers: string[];
  calibration_buckets: string[];
}

export interface BlotterEntry {
  order_id: string;
  instrument_id: string;
  side: string;
  quantity: number;
  order_type: string;
  fills_count: number;
  total_filled: number;
  avg_fill_price: number | string | null;
  vwap_at_submission: number | string | null;
  commission_paid: number | string | null;
  tif_remaining_seconds: number | null;
  broker_status: string | null;
}

export interface PaperGateMetrics {
  as_of: string;
  orders_considered: number;
  reject_rate: number | string;
  broker_error_rate: number | string;
  reconcile_discrepancies: number;
  cash_drift_incidents: number;
  stale_reservations: number;
  average_fill_slippage_bps: number | string | null;
  fill_quality_summary: string;
}

export interface KillSwitchState {
  active?: boolean;
  reason?: string;
  activated_at?: string | null;
  cleared_at?: string | null;
  error?: string;
  [k: string]: unknown;
}

/** The big aggregator. Per-section values can be an `{error}` envelope. */
export interface DashboardSummary {
  as_of: string;
  capabilities: Capabilities;
  ready: { status: string; checks?: Record<string, unknown> } | ErrEnvelope;
  health: BrokerHealth | ErrEnvelope;
  cash: CashStatus | ErrEnvelope;
  regime: RegimeState | ErrEnvelope;
  strategy_runs: { runs: StrategyRun[]; count: number };
  selected_run: {
    strategy_run_id?: string;
    blotter?: { as_of: string; entries: BlotterEntry[] } | ErrEnvelope;
    metrics?: PaperGateMetrics | ErrEnvelope;
    lifecycle?: StrategyLifecycle | ErrEnvelope;
  };
  engines: { budgets: EngineBudget[]; exposure: CombinedExposure | ErrEnvelope };
  freshness: unknown;
  research_campaigns: unknown;
  feature_audits: unknown;
  forecast_evidence: ForecastEvidence[] | ErrEnvelope;
  audit: { events?: unknown[] } | ErrEnvelope;
  compliance: { violations?: unknown[] } | ErrEnvelope;
  kill_switch: KillSwitchState | null;
  production_candidate: Record<string, unknown> | null;
  readiness_snapshot: Record<string, unknown> | null;
  paper_soak: Record<string, unknown> | null;
}

export interface ErrEnvelope {
  error: string;
}

export interface EffectiveConfig {
  as_of: string;
  deployment: {
    paper_trading: boolean;
    profile_preset: string | null;
    broker_host: string;
    broker_port: number;
    broker_client_id: number;
    primary_broker_path: string | null;
    event_bus_backend: string;
    postgres_configured: boolean;
    redis_configured: boolean;
    object_store_root: string;
  };
  capabilities: Capabilities;
  alpha_source_weights: Record<string, number>;
  sections: Record<string, Record<string, unknown>>;
  enums: { run_modes: string[]; execution_backends: string[]; profiles: string[] };
}

export function isErr(v: unknown): v is ErrEnvelope {
  return !!v && typeof v === "object" && "error" in (v as object);
}

// --- CLI command catalog + jobs -------------------------------------------

export interface CommandArg {
  dest: string;
  option_strings: string[];
  positional: boolean;
  kind: "store" | "flag" | "append" | "count";
  type: "str" | "int" | "float" | "decimal" | "bool";
  required: boolean;
  choices: string[] | null;
  default: string | number | boolean | null;
  nargs: string | null;
  help: string;
  metavar: string | null;
}

export interface CommandNode {
  name: string;
  type: "command" | "group";
  help: string;
  path: string[];
  args?: CommandArg[];
  dangerous?: boolean;
  long_running?: boolean;
  commands?: CommandNode[];
}

export interface CommandGroup {
  name: string;
  commands: CommandNode[];
}

export interface CommandCatalog {
  groups: CommandGroup[];
  execution_enabled: boolean;
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface JobMeta {
  id: string;
  path: string[];
  command: string;
  argv: string[];
  status: JobStatus;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  exit_code: number | null;
  error: string | null;
  log_lines: number;
}

export interface JobDetail extends JobMeta {
  logs: string[];
  log_cursor: number;
}

export interface ValidateResult {
  ok: boolean;
  error: string | null;
  argv: string[];
}

// --- Backtest -------------------------------------------------------------

export interface BacktestRun {
  id: string;
  arm: string;
  category: string | null;
  run_dir: string;
  date_start: string;
  date_end: string;
  n_folds: number;
  total_return: number;
  max_drawdown: number | null;
  ic_60d: number | null;
  saved_at: string | null;
}

export interface BacktestPoint {
  date: string;
  equity: number;
  ret: number;
  drawdown: number;
  ic: number | null;
  sharpe: number | null;
  turnover: number | null;
}

// --- System / hardware ----------------------------------------------------

export interface GpuInfo {
  name: string;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  utilization_pct: number | null;
  temperature_c: number | null;
  power_w: number | null;
}

export interface SystemStatus {
  platform: string;
  python: string;
  hostname: string;
  psutil_available: boolean;
  gpus: GpuInfo[];
  cpu?: { percent: number; per_core: number[]; logical: number | null; physical: number | null };
  memory?: { total: number; used: number; available: number; percent: number };
  disk?: { total: number; used: number; free: number; percent: number } | null;
  process?: { pid: number; rss: number; threads: number; cpu_percent: number; create_time: number } | null;
  boot_time?: number;
}

// --- Factors / alpha / models ---------------------------------------------

export interface FeatureSpecLite {
  name: string;
  direction: string;
  lookback_days: number | null;
  description: string;
}

export interface FeatureFamily {
  name: string;
  version: string;
  feature_count: number;
  key_columns: string[];
  required_inputs: string[];
  features: FeatureSpecLite[];
}

export interface FeatureFamilies {
  families: FeatureFamily[];
  total_families: number;
  total_features: number;
  error?: string;
}

export interface AlphaLite {
  name: string | null;
  description: string;
  expected_direction: string;
  lookback_days: number | null;
  required_inputs: string[];
}

export interface AlphaLibrary {
  ensemble_mode: string | null;
  source_weights: Record<string, number>;
  alphas: AlphaLite[];
  auto_promoted_count: number;
  error?: string;
}

export interface ModelRow {
  model_id: string;
  strategy_name: string | null;
  model_version: string | null;
  feature_set_version: string | null;
  created_at: string | null;
  active: boolean;
  metadata: Record<string, unknown>;
}

export interface ModelRegistry {
  models: ModelRow[];
  count: number;
  error?: string;
}

// --- Broker / TWS sync ----------------------------------------------------

export interface BrokerConnection {
  mode: string;
  host: string;
  port: number;
  use_gateway: boolean;
  broker_kind: string;
  client_path: string;
  paper_trading: boolean;
  sync_client_id: number;
  trading_client_id: number;
  account_id_masked: string;
  ibapi_available: boolean;
  contracts_file: string | null;
  contracts_count: number;
  ports: { paper: number; live: number };
}

export interface BrokerPosition {
  instrument_id: string | null;
  symbol: string | null;
  con_id: number | null;
  quantity: number | null;
  market_value: number | null;
}

export interface BrokerSyncResult extends BrokerConnection {
  synced_at: string;
  connected: boolean;
  error?: string | null;
  health?: {
    status: string;
    latency_ms: number | null;
    last_heartbeat_at: string | null;
    detail: string;
  };
  account?: {
    net_asset_value: number | null;
    settled_cash: number | null;
    unsettled_cash: number | null;
    position_count: number;
  };
  positions?: BrokerPosition[];
  open_orders_count?: number;
}

export interface BacktestResult {
  id: string;
  arm: string | null;
  run_id: string | null;
  date_start: string | null;
  date_end: string | null;
  points: BacktestPoint[];
  metrics: {
    total_return: number;
    sharpe_annualized: number | null;
    max_drawdown: number | null;
    ic_60d: number | null;
    mean_ic: number | null;
    n_folds: number;
    fold_negative_ic_streak: number | null;
  };
  portfolio_config: Record<string, unknown> | null;
  production_candidate: Record<string, unknown> | null;
}
