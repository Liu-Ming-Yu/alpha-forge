import { useQuery } from "@tanstack/react-query";
import { Boxes, Brain, GitBranch, Layers, LineChart, Sparkles } from "lucide-react";
import { api } from "../lib/api";
import { useDashboard, useLive } from "../lib/queries";
import { fmtInt, fmtNum, fmtPct, titleCase } from "../lib/format";
import { pick } from "../lib/objects";
import type { ForecastEvidence } from "../lib/types";
import { DISPLAY_LIMITS, QUERY_TIMING } from "../lib/uiConfig";
import { Badge, EmptyState, KeyValue, Pill, Skeleton, type Tone } from "../components/ui/atoms";
import { Card, CardHeader } from "../components/ui/Card";
import { Meter } from "../components/ui/charts";
import { ReplayArea } from "../components/backtest/ReplayArea";
import { Table } from "../components/ui/Table";
import { PageHeader } from "../components/ui/PageHeader";

function useSnapshot<T>(key: string, fn: (s?: AbortSignal) => Promise<T>) {
  return useQuery({ queryKey: [key], queryFn: ({ signal }) => fn(signal), staleTime: QUERY_TIMING.standardStaleMs, retry: false });
}

const DIR_TONE: Record<string, Tone> = { "+": "success", "-": "danger", unknown: "neutral" };

export default function Alpha() {
  const dash = useDashboard();
  const alphaQ = useSnapshot("alpha-library", api.alphaLibrary);
  const familiesQ = useSnapshot("feature-families", api.featureFamilies);
  const modelsQ = useSnapshot("model-registry", api.modelRegistry);
  const decayQ = useLive(["signal-decay"], api.signalDecay);
  const runsQ = useQuery({ queryKey: ["backtest-runs"], queryFn: ({ signal }) => api.backtestRuns(signal), staleTime: QUERY_TIMING.standardStaleMs });
  const topRun = runsQ.data?.runs?.[0]?.id ?? null;
  const icQ = useQuery({
    queryKey: ["backtest-result", topRun],
    queryFn: ({ signal }) => api.backtestResult(topRun!, signal),
    enabled: !!topRun,
    staleTime: QUERY_TIMING.standardStaleMs,
  });

  const alpha = alphaQ.data;
  const families = familiesQ.data?.families ?? [];
  const maxFeat = Math.max(1, ...families.map((f) => f.feature_count));
  const weights = alpha?.source_weights ?? {};
  const maxW = Math.max(0.0001, ...Object.values(weights));
  const models = modelsQ.data?.models ?? [];
  const forecast: ForecastEvidence[] = dash.data && Array.isArray(dash.data.forecast_evidence) ? dash.data.forecast_evidence : [];
  const icPoints = (icQ.data?.points ?? []).map((p) => ({ date: p.date, value: Number(p.ic ?? 0) }));

  return (
    <>
      <PageHeader
        title="Alpha & Models"
        subtitle="Factors, alphas, models & information coefficient"
        right={
          alpha?.ensemble_mode ? <Pill tone="accent">ensemble: {alpha.ensemble_mode}</Pill> : null
        }
      />

      <div className="grid gap-4 lg:grid-cols-3">
        {/* Alpha blend */}
        <Card index={0}>
          <CardHeader title="Alpha blend" icon={<Layers size={15} />} hint="Source weights" />
          {alphaQ.isLoading ? (
            <Skeleton className="h-40" />
          ) : Object.keys(weights).length ? (
            <div className="space-y-3">
              {Object.entries(weights)
                .sort((a, b) => b[1] - a[1])
                .map(([src, w]) => (
                  <div key={src}>
                    <div className="mb-1 flex justify-between text-[13px]">
                      <span className="text-ink">{titleCase(src)}</span>
                      <span className="text-ink-tertiary tnum">{fmtPct(w)}</span>
                    </div>
                    <Meter value={w} min={0} max={maxW} tone="accent" height={6} />
                  </div>
                ))}
              <p className="pt-1 text-xs text-ink-tertiary">
                {alpha?.alphas.length ?? 0} formulaic alphas · {alpha?.auto_promoted_count ?? 0} auto-promoted
              </p>
            </div>
          ) : (
            <EmptyState label="No alpha sources" />
          )}
        </Card>

        {/* Forecast evidence */}
        <Card index={1} className="lg:col-span-2">
          <CardHeader title="Forecast evidence" icon={<GitBranch size={15} />} hint="Per-source prediction gates" />
          {forecast.length ? (
            <Table
              dense
              columns={[
                { key: "src", header: "Source", cell: (r) => titleCase(r.source) },
                { key: "ver", header: "Model", cell: (r) => <span className="font-mono text-xs text-ink-tertiary">{r.model_version}</span> },
                { key: "obs", header: "Obs", align: "right", cell: (r) => <span className="tnum">{fmtInt(r.observations)}</span> },
                { key: "conf", header: "Conf", align: "right", cell: (r) => <span className="tnum">{fmtPct(r.mean_confidence, 0)}</span> },
                {
                  key: "status",
                  header: "Status",
                  align: "right",
                  cell: (r) => <Badge tone={r.passed ? "success" : r.stale ? "warn" : "danger"}>{r.passed ? "Pass" : r.stale ? "Stale" : "Fail"}</Badge>,
                },
              ]}
              rows={forecast}
              rowKey={(r, i) => `${r.source}-${i}`}
            />
          ) : (
            <EmptyState label="No active forecast sources (classical-only blend)" />
          )}
        </Card>
      </div>

      {/* Factor families */}
      <div className="mt-4">
        <Card index={2}>
          <CardHeader
            title="Factor families"
            icon={<Boxes size={15} />}
            hint="Feature factory"
            right={
              <Badge tone="neutral">
                {familiesQ.data?.total_families ?? 0} families · {familiesQ.data?.total_features ?? 0} features
              </Badge>
            }
          />
          {familiesQ.isLoading ? (
            <Skeleton className="h-40" />
          ) : families.length ? (
            <div className="grid gap-x-8 gap-y-3 sm:grid-cols-2">
              {families.map((f) => (
                <div key={f.name}>
                  <div className="mb-1 flex items-baseline justify-between gap-2">
                    <span className="text-[13px] font-medium text-ink">{titleCase(f.name)}</span>
                    <span className="shrink-0 text-xs text-ink-tertiary">
                      <span className="font-mono">{f.version}</span> · {f.feature_count}
                    </span>
                  </div>
                  <Meter value={f.feature_count} min={0} max={maxFeat} tone="accent" height={5} />
                  {f.features.length > 0 && (
                    <p className="mt-1 truncate text-[11px] text-ink-tertiary">
                      {f.features.slice(0, DISPLAY_LIMITS.alphaFamilyFeatureNames).map((s) => s.name).join(", ")}
                      {f.feature_count > DISPLAY_LIMITS.alphaFamilyFeatureNames ? ` +${f.feature_count - DISPLAY_LIMITS.alphaFamilyFeatureNames}` : ""}
                    </p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <EmptyState label={familiesQ.data?.error ?? "No feature families registered"} />
          )}
        </Card>
      </div>

      {/* Alpha library + models */}
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card index={3}>
          <CardHeader title="Alpha library" icon={<Sparkles size={15} />} right={<Badge tone="neutral">{alpha?.alphas.length ?? 0}</Badge>} />
          {alpha?.alphas.length ? (
            <ul className="max-h-[22rem] space-y-2 overflow-y-auto pr-1">
              {alpha.alphas.map((a, i) => (
                <li key={a.name ?? i} className="rounded-lg border border-hairline/[0.06] bg-base/40 p-2.5">
                  <div className="flex items-center gap-2">
                    <span className="min-w-0 flex-1 truncate font-mono text-xs text-ink">{a.name ?? "alpha"}</span>
                    <Badge tone={DIR_TONE[a.expected_direction] ?? "neutral"}>{a.expected_direction}</Badge>
                    {a.lookback_days != null && <span className="shrink-0 text-[11px] text-ink-tertiary">{a.lookback_days}d</span>}
                  </div>
                  {a.description && <p className="mt-1 line-clamp-2 text-[11px] leading-snug text-ink-tertiary">{a.description}</p>}
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState label="No alphas in the library" />
          )}
        </Card>

        <Card index={4}>
          <CardHeader title="Model registry" icon={<Brain size={15} />} right={<Badge tone="neutral">{modelsQ.data?.count ?? 0}</Badge>} />
          {models.length ? (
            <Table
              dense
              columns={[
                { key: "strat", header: "Strategy", cell: (m) => titleCase(m.strategy_name ?? "—") },
                { key: "ver", header: "Version", cell: (m) => <span className="font-mono text-xs">{m.model_version}</span> },
                { key: "feat", header: "Feature set", cell: (m) => <span className="font-mono text-xs text-ink-tertiary">{m.feature_set_version}</span> },
                { key: "act", header: "", align: "right", cell: (m) => <Badge tone={m.active ? "success" : "neutral"}>{m.active ? "Active" : "Retired"}</Badge> },
              ]}
              rows={models}
              rowKey={(m) => m.model_id}
            />
          ) : (
            <EmptyState label={modelsQ.data?.error ?? "No promoted models"} />
          )}
        </Card>
      </div>

      {/* IC */}
      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card index={5} className="lg:col-span-2">
          <CardHeader
            title="Information coefficient"
            icon={<LineChart size={15} />}
            hint={icQ.data ? `${icQ.data.arm} · per-fold mean IC` : "latest walk-forward"}
            right={icQ.data?.metrics ? <Badge tone="neutral">IC60 {fmtNum(icQ.data.metrics.ic_60d, 3)}</Badge> : null}
          />
          {icQ.isLoading ? (
            <Skeleton className="h-44" />
          ) : icPoints.length > 1 ? (
            <>
              <ReplayArea data={icPoints} tone="accent" height={180} baseline={0} />
              <div className="mt-1 flex justify-between text-[11px] text-ink-tertiary">
                <span>{icQ.data?.date_start}</span>
                <span>mean {fmtNum(icQ.data?.metrics.mean_ic, 3)}</span>
                <span>{icQ.data?.date_end}</span>
              </div>
            </>
          ) : (
            <EmptyState label="No IC series (run a backtest first)" />
          )}
        </Card>

        <Card index={6}>
          <CardHeader title="Signal quality" icon={<Sparkles size={15} />} hint="Live decay & dispersion" />
          {decayQ.data ? (
            <div>
              <KeyValue k="Signals">{fmtInt(pick(decayQ.data, "signals_generated"))}</KeyValue>
              <KeyValue k="Mean score">{fmtNum(pick(decayQ.data, "mean_score"), 3)}</KeyValue>
              <KeyValue k="Dispersion">{fmtNum(pick(decayQ.data, "score_dispersion"), 3)}</KeyValue>
              <KeyValue k="Turnover">{fmtPct(pick(decayQ.data, "turnover_rate"))}</KeyValue>
            </div>
          ) : (
            <Skeleton className="h-32" />
          )}
        </Card>
      </div>
    </>
  );
}
