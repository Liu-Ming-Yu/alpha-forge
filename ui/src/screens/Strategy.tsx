import { Brain, GitBranch, Layers, Sparkles, Waves } from "lucide-react";
import { api } from "../lib/api";
import { useSeries } from "../lib/history";
import { useDashboard, useLive } from "../lib/queries";
import { fmtInt, fmtNum, fmtPct, shortId, titleCase } from "../lib/format";
import { listFrom, pick } from "../lib/objects";
import { isErr, type ForecastEvidence } from "../lib/types";
import { DISPLAY_LIMITS, REQUEST_LIMITS } from "../lib/uiConfig";
import { Badge, EmptyState, ErrorCard, KeyValue, Pill, Skeleton, type Tone } from "../components/ui/atoms";
import { Card, CardHeader } from "../components/ui/Card";
import { Meter, Sparkline } from "../components/ui/charts";
import { Table } from "../components/ui/Table";
import { PageHeader } from "../components/ui/PageHeader";

const HEALTH_TONE: Record<string, Tone> = {
  stable: "success",
  scaling_up: "accent",
  launching: "accent",
  degraded: "warn",
  retiring: "warn",
  retired: "danger",
};

/** A lifecycle metric: live value + real sparkline. The meter (with a real
 *  threshold) is only drawn when one is supplied — we never assert a hardcoded
 *  gate the backend didn't provide. */
function GaugeRow({
  label,
  value,
  display,
  tone,
  seriesKey,
  meter,
}: {
  label: string;
  value: number | null;
  display: string;
  tone: Tone;
  seriesKey?: string;
  meter?: { min: number; max: number; threshold?: number };
}) {
  const series = useSeries(seriesKey ?? "");
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-[13px] text-ink-secondary">{label}</span>
        <div className="flex items-center gap-2">
          {seriesKey && series.length > 1 && <Sparkline data={series} tone={tone} width={60} height={18} />}
          <span className="text-[15px] font-semibold text-ink tnum">{display}</span>
        </div>
      </div>
      {meter && <Meter value={value} min={meter.min} max={meter.max} threshold={meter.threshold} tone={tone} />}
    </div>
  );
}

export default function Strategy() {
  const dash = useDashboard();
  const d = dash.data;
  const runId = d?.selected_run?.strategy_run_id ?? null;

  const decayQ = useLive(["signal-decay"], api.signalDecay);
  const contribQ = useLive(["signal-contributions", runId], (s) => api.signalContributions(runId, REQUEST_LIMITS.signalContributionAttribution, s));

  if (!d && dash.isLoading) {
    return (
      <>
        <PageHeader title="Strategy" subtitle="Engine lifecycle & alpha health" />
        <div className="grid gap-4 lg:grid-cols-3">
          <Skeleton className="h-72 lg:col-span-2" />
          <Skeleton className="h-72" />
        </div>
      </>
    );
  }
  if (!d) {
    return (
      <>
        <PageHeader title="Strategy" />
        <ErrorCard message={(dash.error as Error)?.message ?? "No data"} onRetry={() => dash.refetch()} />
      </>
    );
  }

  const lc = d.selected_run?.lifecycle && !isErr(d.selected_run.lifecycle) ? d.selected_run.lifecycle : null;
  const runs = d.strategy_runs?.runs ?? [];
  const budgets = d.engines?.budgets ?? [];
  const forecast: ForecastEvidence[] = Array.isArray(d.forecast_evidence) ? d.forecast_evidence : [];

  // Aggregate signal contributions by source for a clean attribution view.
  const contribRows = listFrom<Record<string, unknown>>(contribQ.data, "contributions");
  const bySource = new Map<string, { w: number; c: number; n: number; state: string }>();
  for (const r of contribRows) {
    const src = String(pick(r, "source") ?? "?");
    const w = Number(pick(r, "blend_weight") ?? 0);
    const c = Number(pick(r, "confidence") ?? 0);
    const prev = bySource.get(src) ?? { w: 0, c: 0, n: 0, state: String(pick(r, "promotion_state") ?? "") };
    prev.w += w;
    prev.c += c;
    prev.n += 1;
    bySource.set(src, prev);
  }
  const sources = [...bySource.entries()].map(([source, a]) => ({
    source,
    weight: a.n ? a.w / a.n : 0,
    confidence: a.n ? a.c / a.n : 0,
    n: a.n,
    state: a.state,
  }));
  const maxW = Math.max(0.0001, ...sources.map((s) => s.weight));

  return (
    <>
      <PageHeader
        title="Strategy"
        subtitle="Engine lifecycle & alpha health"
        right={lc ? <Pill tone={HEALTH_TONE[lc.health] ?? "neutral"}>{titleCase(lc.health)}</Pill> : null}
      />

      <div className="grid gap-4 lg:grid-cols-3">
        {/* Lifecycle */}
        <Card className="lg:col-span-2" index={0}>
          <CardHeader
            title="Engine lifecycle"
            icon={<Brain size={15} />}
            hint={lc ? `${lc.engine_name} · v${lc.engine_version}` : undefined}
            right={lc ? <Badge tone="neutral">{fmtInt(lc.cycles_completed)} cycles</Badge> : null}
          />
          {lc ? (
            <div className="space-y-5">
              <div className="grid gap-5 sm:grid-cols-2">
                <GaugeRow
                  label="Rolling Sharpe (90d)"
                  value={lc.rolling_sharpe_90d}
                  display={fmtNum(lc.rolling_sharpe_90d, 2)}
                  tone={lc.rolling_sharpe_90d < 0 ? "danger" : "accent"}
                  seriesKey="sharpe_90d"
                />
                <GaugeRow
                  label="Rolling IC (60d)"
                  value={lc.rolling_ic_60d}
                  display={fmtNum(lc.rolling_ic_60d, 3)}
                  tone={lc.rolling_ic_60d < 0 ? "danger" : "accent"}
                  seriesKey="ic_60d"
                />
                <GaugeRow
                  label="Max drawdown"
                  value={lc.max_drawdown_realized}
                  display={fmtPct(lc.max_drawdown_realized)}
                  tone={lc.max_drawdown_realized > lc.max_drawdown_limit ? "success" : "danger"}
                  meter={{
                    // range derives from the real realized/limit values; the
                    // reference line is the backend's max_drawdown_limit.
                    min: Math.min(lc.max_drawdown_realized, lc.max_drawdown_limit) * 1.2 - 1e-6,
                    max: 0,
                    threshold: lc.max_drawdown_limit,
                  }}
                />
                <GaugeRow
                  label="Slippage ratio"
                  value={lc.slippage_ratio}
                  display={fmtNum(lc.slippage_ratio, 2)}
                  tone="accent"
                />
              </div>
              <div className="flex items-start gap-2 rounded-xl border border-hairline/[0.07] bg-base/40 p-3">
                <Sparkles size={15} className="mt-0.5 shrink-0 text-accent" />
                <div>
                  <p className="text-[13px] font-medium text-ink">Recommendation</p>
                  <p className="mt-0.5 text-[13px] text-ink-secondary">{lc.recommendation || "—"}</p>
                </div>
                <span className="ml-auto shrink-0 text-xs text-ink-tertiary">{fmtInt(lc.days_active)}d active</span>
              </div>
            </div>
          ) : (
            <EmptyState label="No active strategy run" />
          )}
        </Card>

        {/* Signal decay */}
        <Card index={1}>
          <CardHeader title="Signal quality" icon={<Waves size={15} />} hint="Decay & dispersion" />
          {decayQ.data ? (
            <div>
              <KeyValue k="Signals generated">{fmtInt(pick(decayQ.data, "signals_generated"))}</KeyValue>
              <KeyValue k="Mean score">{fmtNum(pick(decayQ.data, "mean_score"), 3)}</KeyValue>
              <KeyValue k="Dispersion">{fmtNum(pick(decayQ.data, "score_dispersion"), 3)}</KeyValue>
              <KeyValue k="Top quintile">{fmtInt(pick(decayQ.data, "top_quintile_count"))}</KeyValue>
              <KeyValue k="Bottom quintile">{fmtInt(pick(decayQ.data, "bottom_quintile_count"))}</KeyValue>
              <KeyValue k="Turnover">{fmtPct(pick(decayQ.data, "turnover_rate"))}</KeyValue>
            </div>
          ) : decayQ.isError ? (
            <EmptyState label="Signal decay unavailable" />
          ) : (
            <Skeleton className="h-40" />
          )}
        </Card>
      </div>

      {/* Source attribution + forecast evidence */}
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card index={2}>
          <CardHeader title="Source contributions" icon={<Layers size={15} />} hint="Ensemble blend weights" />
          {sources.length > 0 ? (
            <div className="space-y-3">
              {sources
                .sort((a, b) => b.weight - a.weight)
                .map((s) => (
                  <div key={s.source}>
                    <div className="mb-1 flex items-center justify-between text-[13px]">
                      <span className="font-medium text-ink">{titleCase(s.source)}</span>
                      <span className="text-ink-tertiary tnum">
                        w {fmtNum(s.weight, 3)} · conf {fmtPct(s.confidence, 0)}
                      </span>
                    </div>
                    <Meter value={s.weight} min={0} max={maxW} tone="accent" height={6} />
                  </div>
                ))}
            </div>
          ) : (
            <EmptyState label="No signal contributions for this run" />
          )}
        </Card>

        <Card index={3}>
          <CardHeader title="Forecast evidence" icon={<GitBranch size={15} />} hint="Per-source prediction gates" />
          {forecast.length > 0 ? (
            <Table
              dense
              columns={[
                { key: "source", header: "Source", cell: (r) => titleCase(r.source) },
                {
                  key: "obs",
                  header: "Obs",
                  align: "right",
                  cell: (r) => <span className="tnum">{fmtInt(r.observations)}</span>,
                },
                {
                  key: "conf",
                  header: "Conf",
                  align: "right",
                  cell: (r) => <span className="tnum">{fmtPct(r.mean_confidence, 0)}</span>,
                },
                {
                  key: "status",
                  header: "Status",
                  align: "right",
                  cell: (r) => (
                    <Badge tone={r.passed ? "success" : r.stale ? "warn" : "danger"}>
                      {r.passed ? "Pass" : r.stale ? "Stale" : "Fail"}
                    </Badge>
                  ),
                },
              ]}
              rows={forecast}
              rowKey={(r, i) => `${r.source}-${i}`}
            />
          ) : (
            <EmptyState label="No forecast sources active" />
          )}
        </Card>
      </div>

      {/* Engine budgets + runs */}
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card index={4}>
          <CardHeader title="Engine budgets" icon={<Layers size={15} />} />
          {budgets.length > 0 ? (
            <Table
              dense
              columns={[
                { key: "engine", header: "Engine", cell: (r) => titleCase(r.engine_name) },
                { key: "mode", header: "Mode", cell: (r) => <Badge tone="neutral">{r.run_mode}</Badge> },
                { key: "cap", header: "Capital", align: "right", cell: (r) => <span className="tnum">{fmtPct(r.capital_weight)}</span> },
                { key: "gross", header: "Max gross", align: "right", cell: (r) => <span className="tnum">{fmtPct(r.max_gross)}</span> },
                {
                  key: "on",
                  header: "",
                  align: "right",
                  cell: (r) => <Badge tone={r.enabled ? "success" : "neutral"}>{r.enabled ? "On" : "Off"}</Badge>,
                },
              ]}
              rows={budgets}
              rowKey={(r) => r.engine_name + r.engine_version}
            />
          ) : (
            <EmptyState label="No engine budgets" />
          )}
        </Card>

        <Card index={5}>
          <CardHeader title="Strategy runs" icon={<GitBranch size={15} />} right={<Badge tone="neutral">{fmtInt(runs.length)}</Badge>} />
          {runs.length > 0 ? (
            <Table
              dense
              columns={[
                {
                  key: "run",
                  header: "Run",
                  cell: (r) => (
                    <span className={`font-mono text-xs ${r.run_id === runId ? "text-accent" : "text-ink-secondary"}`}>
                      {shortId(String(r.run_id), DISPLAY_LIMITS.shortIdChars)}
                    </span>
                  ),
                },
                { key: "status", header: "Status", cell: (r) => <Badge tone="neutral">{String(r.status ?? "—")}</Badge> },
                {
                  key: "mode",
                  header: "Mode",
                  align: "right",
                  cell: (r) => <span className="text-ink-tertiary">{String(pick(r, "mode", "run_mode") ?? "—")}</span>,
                },
              ]}
              rows={runs}
              rowKey={(r) => String(r.run_id)}
            />
          ) : (
            <EmptyState label="No strategy runs recorded" />
          )}
        </Card>
      </div>
    </>
  );
}
