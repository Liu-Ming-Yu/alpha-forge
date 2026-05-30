import { settingsStore } from "./settings";
import { REQUEST_LIMITS } from "./uiConfig";
import type {
  AlphaLibrary,
  BacktestResult,
  BacktestRun,
  BrokerConnection,
  BrokerSyncResult,
  Capabilities,
  CommandCatalog,
  ConsoleInfo,
  DashboardSummary,
  EffectiveConfig,
  FeatureFamilies,
  JobDetail,
  JobMeta,
  ModelRegistry,
  SystemStatus,
  ValidateResult,
} from "./types";

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail || `HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

interface RequestOpts {
  method?: "GET" | "POST";
  body?: unknown;
  signal?: AbortSignal;
  /** Send the X-API-Key header (default true). Public endpoints pass false. */
  auth?: boolean;
}

async function request<T>(path: string, opts: RequestOpts = {}): Promise<T> {
  const { apiBase, apiKey } = settingsStore.get();
  const headers: Record<string, string> = { Accept: "application/json" };
  if (opts.auth !== false && apiKey) headers["X-API-Key"] = apiKey;
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";

  let res: Response;
  try {
    res = await fetch(`${apiBase}${path}`, {
      method: opts.method ?? "GET",
      headers,
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
      signal: opts.signal,
    });
  } catch (e) {
    throw new ApiError(0, e instanceof Error ? e.message : "network error");
  }

  const text = await res.text();
  let data: unknown = undefined;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }
  if (!res.ok) {
    const detail =
      (data && typeof data === "object" && "detail" in data
        ? String((data as { detail: unknown }).detail)
        : typeof data === "string"
          ? data
          : "") || res.statusText;
    throw new ApiError(res.status, detail);
  }
  return data as T;
}

export const api = {
  // Public bootstrap (no key required).
  consoleInfo: (signal?: AbortSignal) =>
    request<ConsoleInfo>("/console/info", { auth: false, signal }),

  capabilities: (signal?: AbortSignal) =>
    request<Capabilities>("/operator/capabilities", { signal }),

  dashboard: (signal?: AbortSignal) =>
    request<DashboardSummary>("/dashboard/summary", { signal }),

  effectiveConfig: (signal?: AbortSignal) =>
    request<EffectiveConfig>("/v1/config/effective", { signal }),

  healthReady: (signal?: AbortSignal) =>
    request<{ status: string; checks?: Record<string, unknown> }>(
      "/health/ready",
      { signal },
    ),

  strategyRuns: (limit: number = REQUEST_LIMITS.strategyRuns, signal?: AbortSignal) =>
    request<{ runs: Array<Record<string, unknown>>; count: number }>(
      `/strategy/runs?limit=${limit}`,
      { signal },
    ),

  signalDecay: (signal?: AbortSignal) =>
    request<Record<string, unknown>>("/strategy/signal-decay", { signal }),

  signalContributions: (runId: string | null, limit: number = REQUEST_LIMITS.signalContributions, signal?: AbortSignal) =>
    request<{ contributions: Array<Record<string, unknown>> }>(
      `/signals/contributions?limit=${limit}${runId ? `&strategy_run_id=${runId}` : ""}`,
      { signal },
    ),

  blotter: (runId: string, signal?: AbortSignal) =>
    request<Record<string, unknown>>(`/blotter/${runId}`, { signal }),

  unmatchedFills: (limit: number = REQUEST_LIMITS.unmatchedFills, signal?: AbortSignal) =>
    request<Record<string, unknown>>(`/v1/fills/unmatched?limit=${limit}`, { signal }),

  compliance: (sinceHours: number = REQUEST_LIMITS.complianceSinceHours, limit: number = REQUEST_LIMITS.compliance, signal?: AbortSignal) =>
    request<Record<string, unknown>>(
      `/v1/compliance/violations?since_hours=${sinceHours}&limit=${limit}`,
      { signal },
    ),

  cashLedger: (signal?: AbortSignal) =>
    request<Record<string, unknown>>("/v1/cash/ledger", { signal }),

  dataFreshness: (signal?: AbortSignal) =>
    request<Record<string, unknown>>("/v1/data/freshness", { signal }),

  audit: (limit: number = REQUEST_LIMITS.audit, signal?: AbortSignal) =>
    request<Record<string, unknown>>(`/audit?limit=${limit}`, { signal }),

  researchCampaigns: (limit: number = REQUEST_LIMITS.researchCampaigns, signal?: AbortSignal) =>
    request<Record<string, unknown>>(`/research/campaigns?limit=${limit}`, { signal }),

  featureAudits: (limit: number = REQUEST_LIMITS.featureAudits, signal?: AbortSignal) =>
    request<Record<string, unknown>>(`/research/features/audits?limit=${limit}`, {
      signal,
    }),

  paperSoak: (signal?: AbortSignal) =>
    request<Record<string, unknown>>("/v1/paper-soak/latest", { signal }),

  readiness: (profile: string, signal?: AbortSignal) =>
    request<Record<string, unknown>>(`/v1/readiness/latest?profile=${encodeURIComponent(profile)}`, {
      signal,
    }),

  promotionCandidate: (profile: string, signal?: AbortSignal) =>
    request<Record<string, unknown>>(`/v1/promotion/candidate?profile=${encodeURIComponent(profile)}`, {
      signal,
    }),

  // The one write control. Requires a typed confirmation string.
  clearKillSwitch: (reason: string, confirmation: string) =>
    request<Record<string, unknown>>(
      `/v1/kill-switch/clear?reason=${encodeURIComponent(reason)}&confirmation=${encodeURIComponent(confirmation)}`,
      { method: "POST", body: { reason, confirmation } },
    ),

  // --- CLI command catalog + jobs ---
  commands: (signal?: AbortSignal) => request<CommandCatalog>("/v1/commands", { signal }),

  runCommand: (path: string[], values: Record<string, unknown>, confirm = "") =>
    request<JobMeta>("/v1/commands/run", { method: "POST", body: { path, values, confirm } }),

  validateCommand: (path: string[], values: Record<string, unknown>) =>
    request<ValidateResult>("/v1/commands/validate", { method: "POST", body: { path, values } }),

  jobs: (signal?: AbortSignal) => request<{ jobs: JobMeta[] }>("/v1/jobs", { signal }),

  job: (id: string, since = 0, signal?: AbortSignal) =>
    request<JobDetail>(`/v1/jobs/${id}?since=${since}`, { signal }),

  cancelJob: (id: string) =>
    request<{ status: string }>(`/v1/jobs/${id}/cancel`, { method: "POST", body: {} }),

  // --- Backtest ---
  backtestRuns: (signal?: AbortSignal) =>
    request<{ runs: BacktestRun[] }>("/v1/backtest/runs", { signal }),

  backtestResult: (runId: string, signal?: AbortSignal) =>
    request<BacktestResult>(`/v1/backtest/result?run_id=${encodeURIComponent(runId)}`, { signal }),

  // --- Broker / TWS sync ---
  brokerConnection: (mode?: string, signal?: AbortSignal) =>
    request<BrokerConnection>(`/v1/broker/connection${mode ? `?mode=${mode}` : ""}`, { signal }),

  brokerSync: (mode?: string, signal?: AbortSignal) =>
    request<BrokerSyncResult>(`/v1/broker/sync${mode ? `?mode=${mode}` : ""}`, {
      method: "POST",
      body: {},
      signal,
    }),

  // --- System / models / factors / alpha ---
  systemStatus: (signal?: AbortSignal) => request<SystemStatus>("/v1/system/status", { signal }),
  featureFamilies: (signal?: AbortSignal) =>
    request<FeatureFamilies>("/v1/features/families", { signal }),
  alphaLibrary: (signal?: AbortSignal) => request<AlphaLibrary>("/v1/alpha/library", { signal }),
  modelRegistry: (signal?: AbortSignal) => request<ModelRegistry>("/v1/models/registry", { signal }),
};
