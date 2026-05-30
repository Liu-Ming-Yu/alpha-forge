import { useMutation, useQuery } from "@tanstack/react-query";
import { Banknote, Network, RefreshCw, Wifi, WifiOff } from "lucide-react";
import { useState } from "react";
import { api, ApiError } from "../../lib/api";
import { cn } from "../../lib/cn";
import { fmtAgo, fmtInt, fmtMoney, fmtNum } from "../../lib/format";
import type { BrokerSyncResult } from "../../lib/types";
import { QUERY_TIMING } from "../../lib/uiConfig";
import { Badge, Button, EmptyState, KeyValue, Pill, Skeleton, StatusLamp } from "../ui/atoms";
import { Card, CardHeader } from "../ui/Card";
import { Table } from "../ui/Table";

function Metric({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-wide text-ink-tertiary">{label}</p>
      <p className={cn("mt-0.5 text-[18px] font-semibold tnum", tone ?? "text-ink")}>{value}</p>
    </div>
  );
}

export function BrokerSyncPanel() {
  const [result, setResult] = useState<BrokerSyncResult | null>(null);

  const conn = useQuery({
    queryKey: ["broker-connection"],
    queryFn: ({ signal }) => api.brokerConnection(undefined, signal),
    staleTime: QUERY_TIMING.relaxedRefetchMs,
    retry: false,
  });

  const sync = useMutation({
    mutationFn: () => api.brokerSync(),
    onSuccess: (data) => setResult(data),
  });

  const c = conn.data;
  const activeMode = c?.mode ?? "";

  return (
    <Card index={0}>
      <CardHeader
        title="TWS connection & sync"
        icon={<Network size={15} />}
        hint={c ? `${c.broker_kind} · sync client ${c.sync_client_id}` : undefined}
        right={
          <Button variant="primary" className="py-1.5" onClick={() => sync.mutate()} disabled={sync.isPending || !c?.ibapi_available}>
            <RefreshCw size={13} className={sync.isPending ? "animate-spin" : ""} />
            {sync.isPending ? "Syncing…" : "Sync with TWS"}
          </Button>
        }
      />

      {conn.isLoading ? (
        <Skeleton className="h-24" />
      ) : c ? (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <Pill tone={c.paper_trading ? "accent" : "danger"}>{c.paper_trading ? "Paper" : "Live"} mode</Pill>
            <code className="rounded-md bg-hairline/10 px-2 py-1 font-mono text-xs text-ink">
              {c.host}:{c.port}
            </code>
            {c.account_id_masked && <Badge tone="neutral">acct {c.account_id_masked}</Badge>}
            <Badge tone={c.ibapi_available ? "success" : "danger"}>ibapi {c.ibapi_available ? "ready" : "missing"}</Badge>
            <Badge tone="neutral">{c.contracts_count} contracts</Badge>
          </div>
          <div className="mt-3 flex items-center gap-2 text-xs">
            <span className="text-ink-tertiary">Port by mode:</span>
            {Object.entries(c.ports).map(([mode, port]) => (
              <span
                key={mode}
                className={cn(
                  "rounded-md px-2 py-0.5 font-mono",
                  activeMode === mode ? "bg-accent/15 text-accent" : "text-ink-tertiary",
                )}
              >
                {mode} → :{port}
              </span>
            ))}
          </div>
        </>
      ) : (
        <EmptyState label="Broker connection info unavailable" />
      )}

      {/* Sync result */}
      {sync.isError && (
        <p className="mt-4 rounded-lg border border-danger/20 bg-danger/5 px-3 py-2 text-xs text-danger">
          {sync.error instanceof ApiError ? sync.error.detail : "Sync request failed"}
        </p>
      )}

      {result && (
        <div className="mt-4 border-t border-hairline/[0.07] pt-4">
          <div className="mb-3 flex items-center gap-2">
            {result.connected ? <Wifi size={15} className="text-success" /> : <WifiOff size={15} className="text-danger" />}
            <StatusLamp tone={result.connected && !result.error ? "success" : "danger"} pulse={false} size={8} />
            <span className="text-[14px] font-semibold text-ink">
              {result.error ? "Not synced" : `Synced from TWS :${result.port}`}
            </span>
            <span className="ml-auto text-xs text-ink-tertiary">{fmtAgo(result.synced_at)}</span>
          </div>

          {result.error ? (
            <p className="rounded-lg border border-warn/20 bg-warn/5 px-3 py-2 text-xs text-warn">{result.error}</p>
          ) : (
            <>
              <div className="mb-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Metric label="Net asset value" value={fmtMoney(result.account?.net_asset_value)} tone="text-ink" />
                <Metric label="Settled cash" value={fmtMoney(result.account?.settled_cash)} />
                <Metric label="Positions" value={fmtInt(result.account?.position_count)} />
                <Metric label="Open orders" value={fmtInt(result.open_orders_count)} />
              </div>

              <div className="grid gap-4 sm:grid-cols-3">
                <div className="sm:col-span-2">
                  <p className="mb-1 flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-ink-tertiary">
                    <Banknote size={12} /> Positions
                  </p>
                  {result.positions && result.positions.length ? (
                    <Table
                      dense
                      columns={[
                        { key: "sym", header: "Symbol", cell: (p) => p.symbol ?? <span className="font-mono text-xs text-ink-tertiary">{p.con_id ?? "—"}</span> },
                        { key: "qty", header: "Qty", align: "right", cell: (p) => <span className="tnum">{fmtNum(p.quantity, 0)}</span> },
                        { key: "mv", header: "Market value", align: "right", cell: (p) => <span className="tnum">{fmtMoney(p.market_value)}</span> },
                      ]}
                      rows={result.positions}
                      rowKey={(p, i) => p.instrument_id ?? String(i)}
                    />
                  ) : (
                    <p className="text-xs text-ink-tertiary">No open positions.</p>
                  )}
                </div>
                <div>
                  <p className="mb-1 text-[11px] uppercase tracking-wide text-ink-tertiary">Health</p>
                  <KeyValue k="Status">{result.health?.status ?? "—"}</KeyValue>
                  <KeyValue k="Latency">{result.health?.latency_ms != null ? `${fmtNum(result.health.latency_ms, 0)} ms` : "—"}</KeyValue>
                  <KeyValue k="Heartbeat">{fmtAgo(result.health?.last_heartbeat_at)}</KeyValue>
                  <KeyValue k="Unsettled">{fmtMoney(result.account?.unsettled_cash)}</KeyValue>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </Card>
  );
}
