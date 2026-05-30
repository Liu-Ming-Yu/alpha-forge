import {
  Activity,
  CircleDollarSign,
  Compass,
  Database,
  Gauge,
  RefreshCw,
  ScrollText,
  ShieldCheck,
} from "lucide-react";
import { useSeries } from "../lib/history";
import { useBrokerSync, useDashboard } from "../lib/queries";
import { fmtAgo, fmtInt, fmtMoney, fmtNum, fmtPct, titleCase } from "../lib/format";
import { isObj, listFrom, pick, scalarEntries } from "../lib/objects";
import { isErr } from "../lib/types";
import { DISPLAY_LIMITS } from "../lib/uiConfig";
import { Badge, CheckDot, EmptyState, ErrorCard, Pill, Skeleton, StatusLamp, type Tone } from "../components/ui/atoms";
import { Card, CardHeader } from "../components/ui/Card";
import { Delta, LiveArea, Meter } from "../components/ui/charts";
import { JsonPeek } from "../components/ui/JsonPeek";
import { KeyValue } from "../components/ui/atoms";
import { PageHeader } from "../components/ui/PageHeader";
import { Stat } from "../components/ui/Stat";

function seriesDelta(data: { v: number }[]): number | null {
  if (data.length < 2) return null;
  const a = data[0].v;
  const b = data[data.length - 1].v;
  if (!Number.isFinite(a) || a === 0) return null;
  return (b - a) / Math.abs(a);
}

/** Headline = live Net Asset Value from TWS (broker truth). Falls back to a
 *  clear "TWS unavailable" state with the internal ledger as secondary
 *  context — never a placeholder number. */
function HeroNav({ internalAvailable }: { internalAvailable: number | null }) {
  const sync = useBrokerSync();
  const r = sync.data;
  const series = useSeries("tws_nav");
  const delta = seriesDelta(series);
  const connected = !!r?.connected && !r?.error;
  const nav = connected ? (r?.account?.net_asset_value ?? null) : null;
  return (
    <Card className="lg:col-span-2" index={0}>
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2.5">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-accent/15 text-accent">
            <CircleDollarSign size={18} />
          </span>
          <div>
            <p className="text-[12.5px] font-medium uppercase tracking-wide text-ink-tertiary">
              Net asset value
            </p>
            <p className="flex items-center gap-1.5 text-xs text-ink-tertiary">
              {connected ? (
                <>
                  <StatusLamp tone="success" pulse size={7} /> Live from TWS :{r?.port} ·{" "}
                  {fmtAgo(r?.synced_at)}
                </>
              ) : sync.isFetching ? (
                "Syncing with TWS…"
              ) : (
                "TWS unavailable"
              )}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {delta !== null && <Delta value={delta} />}
          <button
            onClick={() => sync.refetch()}
            disabled={sync.isFetching}
            title="Refresh from TWS"
            className="rounded-lg bg-hairline/10 p-1.5 text-ink-tertiary transition-colors hover:text-ink-secondary disabled:opacity-50"
          >
            <RefreshCw size={13} className={sync.isFetching ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      <div className="mt-4 text-[40px] font-semibold leading-none tracking-tight text-ink tnum">
        {fmtMoney(nav)}
      </div>

      <div className="mt-3">
        <LiveArea data={series} tone="accent" height={150} />
      </div>

      {connected ? (
        <div className="mt-3 grid grid-cols-3 gap-3 border-t border-hairline/[0.06] pt-3">
          <Mini label="Settled cash" value={fmtMoney(r?.account?.settled_cash, { compact: true })} />
          <Mini label="Positions" value={fmtInt(r?.account?.position_count)} />
          <Mini label="Open orders" value={fmtInt(r?.open_orders_count)} />
        </div>
      ) : (
        <div className="mt-3 border-t border-hairline/[0.06] pt-3">
          <p className="text-xs text-warn">
            {r?.error ?? "Connect TWS (Execution → Sync) to show the live account value."}
          </p>
          <p className="mt-1 text-xs text-ink-tertiary">
            Internal ledger available cash: {fmtMoney(internalAvailable)}
          </p>
        </div>
      )}
    </Card>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-wide text-ink-tertiary">{label}</p>
      <p className="mt-0.5 text-[15px] font-semibold text-ink tnum">{value}</p>
    </div>
  );
}

const REGIME_TONE: Record<string, Tone> = {
  risk_on: "success",
  calm: "success",
  neutral: "accent",
  risk_off: "danger",
  stress: "danger",
  high_vol: "warn",
};

export default function Overview() {
  const dash = useDashboard();
  const d = dash.data;

  if (!d && dash.isLoading) {
    return (
      <>
        <PageHeader title="Overview" subtitle="Live command center" />
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <Skeleton className="h-64 lg:col-span-2" />
          <Skeleton className="h-64" />
          {Array.from({ length: DISPLAY_LIMITS.overviewLoadingCards }).map((_, i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
      </>
    );
  }
  if (!d) {
    return (
      <>
        <PageHeader title="Overview" />
        <ErrorCard message={(dash.error as Error)?.message ?? "No data"} onRetry={() => dash.refetch()} />
      </>
    );
  }

  const cash = isErr(d.cash) ? null : d.cash;
  const health = isErr(d.health) ? null : d.health;
  const regime = isErr(d.regime) ? null : d.regime;
  const exposure = d.engines && !isErr(d.engines.exposure) ? d.engines.exposure : null;
  const auditRows = listFrom<Record<string, unknown>>(d.audit, "entries");
  const freshness = listFrom<Record<string, unknown>>(d.freshness, "instruments");
  const soakSections = isObj(d.paper_soak)
    ? (d.paper_soak.passed_sections as Record<string, boolean> | undefined)
    : undefined;

  const num = (v: unknown) => (v == null ? null : Number(v));

  return (
    <>
      <PageHeader
        title="Overview"
        subtitle="Live command center"
        right={<span className="text-xs text-ink-tertiary">Updated {fmtAgo(d.as_of)}</span>}
      />

      {/* Hero + regime */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <HeroNav internalAvailable={cash ? num(cash.available_cash) : null} />
        <Card index={1}>
          <CardHeader title="Market regime" icon={<Compass size={15} />} />
          {regime ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Pill tone={REGIME_TONE[regime.label] ?? "neutral"}>{titleCase(regime.label)}</Pill>
                <span className="text-xs text-ink-tertiary">{fmtAgo(regime.as_of)}</span>
              </div>
              <div>
                <div className="mb-1 flex justify-between text-xs text-ink-tertiary">
                  <span>Gross exposure scale</span>
                  <span className="text-ink tnum">{fmtNum(regime.gross_exposure_scale, 2)}×</span>
                </div>
                <Meter
                  value={regime.gross_exposure_scale}
                  min={0}
                  max={Math.max(1.5, regime.gross_exposure_scale)}
                  threshold={1}
                  tone="accent"
                />
              </div>
              <div>
                <div className="mb-1 flex justify-between text-xs text-ink-tertiary">
                  <span>Breadth</span>
                  <span className="text-ink tnum">{fmtPct(regime.breadth_pct, 0, true)}</span>
                </div>
                <Meter value={regime.breadth_pct} min={0} max={100} tone="success" />
              </div>
              <div className="grid grid-cols-2 gap-3 border-t border-hairline/[0.06] pt-3">
                <Mini label="Trend z" value={fmtNum(regime.trend_z, 2)} />
                <Mini label="Annualized vol" value={fmtPct(regime.annualized_vol)} />
              </div>
            </div>
          ) : (
            <EmptyState label="Regime unavailable" />
          )}
        </Card>
      </div>

      {/* Stat tiles */}
      <div className="mt-4 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat
          label="Throttle tokens"
          value={fmtNum(health?.throttle_tokens_available, 1)}
          seriesKey="throttle_tokens"
          tone="accent"
          index={2}
        />
        <Stat
          label="Orders this session"
          value={fmtInt(health?.orders_submitted_this_session)}
          seriesKey="orders_submitted"
          tone="accent"
          index={3}
        />
        <Stat
          label="Allocated exposure"
          value={exposure ? fmtPct(exposure.allocated_capital_weight) : "—"}
          sub={exposure ? `${exposure.enabled_engines} engine(s)` : undefined}
          seriesKey="allocated_weight"
          tone="success"
          index={4}
        />
        <Stat
          label="Regime vol"
          value={fmtPct(regime?.annualized_vol)}
          seriesKey="regime_vol"
          tone="warn"
          index={5}
        />
      </div>

      {/* Monitors */}
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Broker */}
        <Card index={6}>
          <CardHeader title="Broker" icon={<Activity size={15} />} />
          {health ? (
            <div>
              <div className="flex items-center gap-2">
                <StatusLamp
                  tone={health.kill_switch_active ? "danger" : health.connected ? "success" : "warn"}
                  pulse={health.connected && !health.kill_switch_active}
                />
                <span className="text-[15px] font-semibold text-ink">
                  {health.kill_switch_active ? "Halted" : health.connected ? "Connected" : "Disconnected"}
                </span>
                <span className="ml-auto text-xs text-ink-tertiary">{titleCase(health.status)}</span>
              </div>
              <div className="mt-3">
                <KeyValue k="Latency">{health.latency_ms != null ? `${fmtNum(health.latency_ms, 0)} ms` : "—"}</KeyValue>
                <KeyValue k="Last heartbeat">{fmtAgo(health.last_heartbeat_at)}</KeyValue>
                <KeyValue k="Throttle tokens">{fmtNum(health.throttle_tokens_available, 1)}</KeyValue>
              </div>
            </div>
          ) : (
            <EmptyState label="Broker health unavailable" />
          )}
        </Card>

        {/* Readiness & promotion */}
        <Card index={7}>
          <CardHeader title="Readiness" icon={<ShieldCheck size={15} />} />
          {soakSections && Object.keys(soakSections).length > 0 ? (
            <div className="space-y-1.5">
              {Object.entries(soakSections).map(([k, ok]) => (
                <div key={k} className="flex items-center justify-between">
                  <span className="text-[13px] text-ink-secondary">{titleCase(k)}</span>
                  <CheckDot ok={!!ok} />
                </div>
              ))}
              <PromotionVerdict candidate={d.production_candidate} />
            </div>
          ) : (
            <div>
              <EmptyState label="No paper-soak report yet" />
              <PromotionVerdict candidate={d.production_candidate} />
            </div>
          )}
        </Card>

        {/* Data freshness */}
        <Card index={8}>
          <CardHeader
            title="Data freshness"
            icon={<Database size={15} />}
            right={<Badge tone="neutral">{fmtInt(freshness.length)}</Badge>}
          />
          {freshness.length > 0 ? (
            <div className="space-y-1.5">
              {freshness.slice(0, DISPLAY_LIMITS.overviewFreshnessRows).map((row, i) => (
                <div key={i} className="flex items-center justify-between gap-2">
                  <span className="truncate font-mono text-xs text-ink-secondary">
                    {String(pick(row, "instrument_id") ?? "—").slice(0, DISPLAY_LIMITS.overviewInstrumentIdChars)}
                  </span>
                  <span className="shrink-0 text-xs text-ink-tertiary">
                    {fmtAgo(String(pick(row, "last_bar_at") ?? ""))}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState icon={<Gauge size={20} />} label="No recent bar ingests" />
          )}
        </Card>
      </div>

      {/* Activity feed */}
      <div className="mt-4">
        <Card index={9}>
          <CardHeader title="Recent activity" icon={<ScrollText size={15} />} hint="Audit event stream" />
          {auditRows.length > 0 ? (
            <ul className="divide-y divide-hairline/[0.05]">
              {auditRows.slice(0, DISPLAY_LIMITS.overviewAuditRows).map((row, i) => {
                const type = String(pick(row, "event_type", "type", "kind", "name") ?? "event");
                const when = String(pick(row, "occurred_at", "timestamp", "created_at", "as_of") ?? "");
                const detail = pick<string>(row, "detail", "summary", "message", "reason");
                return (
                  <li key={i} className="flex items-center gap-3 py-2.5">
                    <StatusLamp tone="neutral" size={7} />
                    <span className="text-[13px] font-medium text-ink">{titleCase(type)}</span>
                    {detail && <span className="truncate text-[13px] text-ink-tertiary">{String(detail)}</span>}
                    <span className="ml-auto shrink-0 text-xs text-ink-tertiary">{fmtAgo(when)}</span>
                  </li>
                );
              })}
            </ul>
          ) : (
            <EmptyState label="No recent audit events" />
          )}
        </Card>
      </div>
    </>
  );
}

function PromotionVerdict({ candidate }: { candidate: unknown }) {
  if (!isObj(candidate)) return null;
  if ("error" in candidate) return null;
  const eligible = pick<boolean>(candidate, "eligible", "passed", "promote", "ready");
  const blockers = listFrom(candidate, "blockers");
  const scalars = scalarEntries(candidate).filter(([k]) =>
    ["profile", "eligible", "passed", "decision", "sharpe", "recommendation"].includes(k),
  );
  return (
    <div className="mt-3 border-t border-hairline/[0.06] pt-3">
      <div className="flex items-center justify-between">
        <span className="text-[13px] font-medium text-ink-secondary">Promotion candidate</span>
        {eligible !== undefined ? (
          <Pill tone={eligible ? "success" : "warn"}>{eligible ? "Eligible" : "Not yet"}</Pill>
        ) : (
          <span className="text-xs text-ink-tertiary">{blockers.length} blocker(s)</span>
        )}
      </div>
      {scalars.length > 0 && (
        <div className="mt-1">
          {scalars.map(([k, v]) => (
            <KeyValue key={k} k={titleCase(k)}>
              {String(v)}
            </KeyValue>
          ))}
        </div>
      )}
      <JsonPeek data={candidate} label="Details" />
    </div>
  );
}
