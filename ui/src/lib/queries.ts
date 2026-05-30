import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { useEffect } from "react";
import { api } from "./api";
import { record } from "./history";
import { PAUSED_CADENCE, settingsStore } from "./settings";
import {
  isErr,
  type BrokerSyncResult,
  type Capabilities,
  type DashboardSummary,
  type SystemStatus,
} from "./types";
import { QUERY_TIMING } from "./uiConfig";

/** Capabilities double as the connection signal: success === connected. */
export function useCapabilities(): UseQueryResult<Capabilities> {
  return useQuery({
    queryKey: ["capabilities"],
    queryFn: ({ signal }) => api.capabilities(signal),
    retry: false,
    staleTime: QUERY_TIMING.standardStaleMs,
  });
}

/** Live refresh cadence (ms) as a React Query `refetchInterval` value. */
export function useCadence(): number | false {
  const cadence = settingsStore.use((s) => s.cadence);
  return cadence === PAUSED_CADENCE ? false : cadence;
}

export function useRelaxedCadence(): number | false {
  const cadence = settingsStore.use((s) => s.cadence);
  return cadence === PAUSED_CADENCE ? false : QUERY_TIMING.relaxedRefetchMs;
}

const asNum = (v: unknown): number | null => {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
};

/** Pull the scalars worth charting out of one dashboard poll. */
function recordDashboard(d: DashboardSummary): void {
  const out: Record<string, number | null> = {};
  if (!isErr(d.cash)) {
    out.available_cash = asNum(d.cash.available_cash);
    out.settled_cash = asNum(d.cash.settled_cash);
    out.reserved_cash = asNum(d.cash.reserved_cash);
    out.unsettled_cash = asNum(d.cash.unsettled_cash);
  }
  if (!isErr(d.health)) {
    out.throttle_tokens = asNum(d.health.throttle_tokens_available);
    out.latency_ms = asNum(d.health.latency_ms);
    out.orders_submitted = asNum(d.health.orders_submitted_this_session);
  }
  if (!isErr(d.regime)) {
    out.regime_vol = asNum(d.regime.annualized_vol);
    out.regime_trend_z = asNum(d.regime.trend_z);
    out.regime_gross_scale = asNum(d.regime.gross_exposure_scale);
    out.regime_breadth = asNum(d.regime.breadth_pct);
  }
  if (d.engines && !isErr(d.engines.exposure)) {
    out.allocated_weight = asNum(d.engines.exposure.allocated_capital_weight);
  }
  const lc = d.selected_run?.lifecycle;
  if (lc && !isErr(lc)) {
    out.sharpe_90d = asNum(lc.rolling_sharpe_90d);
    out.ic_60d = asNum(lc.rolling_ic_60d);
  }
  const m = d.selected_run?.metrics;
  if (m && !isErr(m)) {
    out.reject_rate = asNum(m.reject_rate);
  }
  const ts = Date.parse(d.as_of);
  record(out, Number.isNaN(ts) ? Date.now() : ts);
}

/** Primary live poll — hydrates the Overview and feeds the history buffers. */
export function useDashboard(): UseQueryResult<DashboardSummary> {
  const refetchInterval = useCadence();
  const q = useQuery({
    queryKey: ["dashboard"],
    queryFn: ({ signal }) => api.dashboard(signal),
    refetchInterval,
    refetchIntervalInBackground: false,
    staleTime: QUERY_TIMING.liveStaleMs,
  });
  useEffect(() => {
    if (q.data) recordDashboard(q.data);
  }, [q.data]);
  return q;
}

/** Live TWS account snapshot for the Overview NAV headline.
 *
 * Each call opens a real (read-only) broker socket, so this polls on a relaxed
 * relaxed cadence (paused with the global live toggle) rather than the live cadence,
 * and records NAV into the history buffer to drive the hero chart. */
export function useBrokerSync(): UseQueryResult<BrokerSyncResult> {
  const cadence = settingsStore.use((s) => s.cadence);
  const q = useQuery({
    queryKey: ["broker-sync"],
    queryFn: ({ signal }) => api.brokerSync(undefined, signal),
    refetchInterval: cadence === PAUSED_CADENCE ? false : QUERY_TIMING.relaxedRefetchMs,
    refetchOnWindowFocus: false,
    retry: false,
    staleTime: QUERY_TIMING.relaxedStaleMs,
  });
  useEffect(() => {
    const nav = q.data?.account?.net_asset_value;
    if (typeof nav !== "number") return;
    const ts = q.data?.synced_at ? Date.parse(q.data.synced_at) : Date.now();
    record({ tws_nav: nav }, Number.isNaN(ts) ? Date.now() : ts);
  }, [q.data]);
  return q;
}

/** Live host hardware status; records CPU/mem/GPU into history for the graphs. */
export function useSystemStatus(): UseQueryResult<SystemStatus> {
  const refetchInterval = useCadence();
  const q = useQuery({
    queryKey: ["system-status"],
    queryFn: ({ signal }) => api.systemStatus(signal),
    refetchInterval,
    staleTime: QUERY_TIMING.liveStaleMs,
  });
  useEffect(() => {
    if (!q.data) return;
    const out: Record<string, number | null> = {
      cpu_pct: q.data.cpu?.percent ?? null,
      mem_pct: q.data.memory?.percent ?? null,
    };
    const gpu = q.data.gpus?.[0];
    if (gpu) {
      out.gpu_util = gpu.utilization_pct;
      if (gpu.memory_used_mb != null && gpu.memory_total_mb)
        out.gpu_mem_pct = (gpu.memory_used_mb / gpu.memory_total_mb) * 100;
    }
    record(out);
  }, [q.data]);
  return q;
}

/** Generic live query helper for per-screen endpoints. */
export function useLive<T>(
  key: readonly unknown[],
  fn: (signal?: AbortSignal) => Promise<T>,
  opts: { enabled?: boolean } = {},
): UseQueryResult<T> {
  const refetchInterval = useCadence();
  return useQuery({
    queryKey: key,
    queryFn: ({ signal }) => fn(signal),
    refetchInterval,
    enabled: opts.enabled ?? true,
    staleTime: QUERY_TIMING.liveStaleMs,
  });
}
