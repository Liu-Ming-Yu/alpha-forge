import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Cpu,
  Gauge,
  KeyRound,
  Layers,
  Moon,
  Plug,
  ShieldCheck,
  Sun,
} from "lucide-react";
import { useState } from "react";
import { api } from "../lib/api";
import { useCapabilities } from "../lib/queries";
import { fmtNum, fmtPct, titleCase } from "../lib/format";
import { scalarEntries } from "../lib/objects";
import { CADENCE_OPTIONS, type Cadence, settingsStore, updateSettings } from "../lib/settings";
import { DISPLAY_LIMITS, QUERY_TIMING } from "../lib/uiConfig";
import { Badge, Button, CheckDot, EmptyState, KeyValue, Pill, Segmented, Skeleton } from "../components/ui/atoms";
import { Card, CardHeader } from "../components/ui/Card";
import { Meter } from "../components/ui/charts";
import { JsonPeek } from "../components/ui/JsonPeek";
import { PageHeader } from "../components/ui/PageHeader";

function ConnectionCard() {
  const qc = useQueryClient();
  const settings = settingsStore.use((s) => s);
  const caps = useCapabilities();
  const [base, setBase] = useState(settings.apiBase);
  const [key, setKey] = useState(settings.apiKey);

  const save = () => {
    updateSettings({ apiBase: base.trim(), apiKey: key.trim() });
    qc.invalidateQueries();
  };

  return (
    <Card>
      <CardHeader title="Connection" icon={<Plug size={15} />} />
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Pill tone={caps.isSuccess ? "success" : "danger"}>{caps.isSuccess ? "Connected" : "Disconnected"}</Pill>
          {caps.data && <span className="text-xs text-ink-tertiary">auth: {caps.data.auth.mode}</span>}
        </div>
        <label className="block">
          <span className="mb-1 block text-[13px] text-ink-secondary">API base URL</span>
          <input
            value={base}
            onChange={(e) => setBase(e.target.value)}
            placeholder="(same origin)"
            className="w-full rounded-xl border border-hairline/10 bg-base/60 px-3 py-2 font-mono text-xs text-ink outline-none focus:border-accent/50"
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-[13px] text-ink-secondary">Operator API key</span>
          <div className="relative">
            <KeyRound size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-tertiary" />
            <input
              type="password"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="X-API-Key"
              className="w-full rounded-xl border border-hairline/10 bg-base/60 py-2 pl-9 pr-3 text-sm text-ink outline-none focus:border-accent/50"
            />
          </div>
        </label>
        <Button variant="primary" onClick={save} className="w-full">
          Save & test
        </Button>
      </div>
    </Card>
  );
}

function AppearanceCard() {
  const theme = settingsStore.use((s) => s.theme);
  const density = settingsStore.use((s) => s.density);
  const cadence = settingsStore.use((s) => s.cadence);
  return (
    <Card>
      <CardHeader title="Appearance & live" icon={<Gauge size={15} />} />
      <div className="space-y-4">
        <Row label="Theme">
          <Segmented
            size="sm"
            value={theme}
            onChange={(v) => updateSettings({ theme: v })}
            options={[
              { value: "system", label: "Auto" },
              { value: "light", label: <Sun size={14} /> },
              { value: "dark", label: <Moon size={14} /> },
            ]}
          />
        </Row>
        <Row label="Density">
          <Segmented
            size="sm"
            value={density}
            onChange={(v) => updateSettings({ density: v })}
            options={[
              { value: "comfortable", label: "Comfortable" },
              { value: "compact", label: "Compact" },
            ]}
          />
        </Row>
        <Row label="Live cadence">
          <Segmented
            size="sm"
            value={String(cadence)}
            onChange={(v) => updateSettings({ cadence: Number(v) as Cadence })}
            options={CADENCE_OPTIONS.map((option) => ({
              value: String(option.value),
              label: option.label,
            }))}
          />
        </Row>
      </div>
    </Card>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-[13px] text-ink-secondary">{label}</span>
      {children}
    </div>
  );
}

export default function Settings() {
  const cfg = useQuery({
    queryKey: ["effective-config"],
    queryFn: ({ signal }) => api.effectiveConfig(signal),
    staleTime: QUERY_TIMING.relaxedRefetchMs,
    retry: false,
  });
  const caps = useCapabilities();
  const dep = cfg.data?.deployment;
  const weights = cfg.data?.alpha_source_weights ?? {};
  const maxW = Math.max(0.0001, ...Object.values(weights));
  const sections = cfg.data?.sections ?? {};

  return (
    <>
      <PageHeader title="Settings" subtitle="Connection, modes & configuration" />

      <div className="grid gap-4 lg:grid-cols-3">
        <ConnectionCard />
        <AppearanceCard />

        {/* Capabilities */}
        <Card>
          <CardHeader title="Capabilities" icon={<ShieldCheck size={15} />} />
          {caps.data ? (
            <div>
              <p className="mb-1.5 text-[11px] uppercase tracking-wide text-ink-tertiary">Write controls</p>
              {Object.entries(caps.data.write_controls).map(([k, v]) => (
                <div key={k} className="flex items-center justify-between py-1">
                  <span className="text-[13px] text-ink-secondary">{titleCase(k)}</span>
                  <Badge tone={v ? "success" : "neutral"}>{v ? "Enabled" : "Off"}</Badge>
                </div>
              ))}
              <p className="mb-1 mt-3 text-[11px] uppercase tracking-wide text-ink-tertiary">Roles</p>
              <div className="flex flex-wrap gap-1.5">
                {caps.data.auth.roles_advertised.length ? (
                  caps.data.auth.roles_advertised.map((r) => (
                    <Badge key={r} tone="accent">
                      {r}
                    </Badge>
                  ))
                ) : (
                  <span className="text-xs text-ink-tertiary">single-key auth</span>
                )}
              </div>
            </div>
          ) : (
            <Skeleton className="h-40" />
          )}
        </Card>
      </div>

      {/* Modes & deployment */}
      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Modes & deployment"
            icon={<Cpu size={15} />}
            hint="Run mode is set at launch (CLI / env) — shown read-only"
          />
          {dep ? (
            <div className="grid gap-x-8 gap-y-0 sm:grid-cols-2">
              <div>
                <KeyValue k="Trading">
                  <Pill tone={dep.paper_trading ? "accent" : "danger"}>{dep.paper_trading ? "Paper" : "Live"}</Pill>
                </KeyValue>
                <KeyValue k="Profile preset">{dep.profile_preset ?? "—"}</KeyValue>
                <KeyValue k="Broker">
                  <span className="font-mono text-xs">
                    {dep.broker_host}:{dep.broker_port}
                  </span>
                </KeyValue>
                <KeyValue k="Broker path">{dep.primary_broker_path ?? "—"}</KeyValue>
              </div>
              <div>
                <KeyValue k="Event bus">{dep.event_bus_backend}</KeyValue>
                <KeyValue k="Postgres">
                  <CheckDot ok={dep.postgres_configured} />
                </KeyValue>
                <KeyValue k="Redis">
                  <CheckDot ok={dep.redis_configured} />
                </KeyValue>
                <KeyValue k="Object store" mono>
                  {dep.object_store_root}
                </KeyValue>
              </div>
            </div>
          ) : cfg.isLoading ? (
            <Skeleton className="h-28" />
          ) : (
            <EmptyState label="Configuration unavailable" />
          )}
        </Card>

        {/* Alpha source weights */}
        <Card>
          <CardHeader title="Alpha blend" icon={<Layers size={15} />} hint="Source weights" />
          {Object.keys(weights).length ? (
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
            </div>
          ) : (
            <EmptyState label="No source weights" />
          )}
        </Card>
      </div>

      {/* Config sections */}
      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {Object.entries(sections).map(([name, body]) => {
          const entries = scalarEntries(body);
          if (!entries.length) return null;
          return (
            <Card key={name}>
              <CardHeader title={titleCase(name)} />
              <div>
                {entries.slice(0, DISPLAY_LIMITS.settingsConfigFields).map(([k, v]) => (
                  <KeyValue key={k} k={titleCase(k)}>
                    {typeof v === "boolean" ? <CheckDot ok={v} /> : typeof v === "number" ? fmtNum(v, 4).replace(/\.?0+$/, "") : String(v)}
                  </KeyValue>
                ))}
                {entries.length > DISPLAY_LIMITS.settingsConfigFields && <JsonPeek data={body} label={`All ${entries.length} fields`} />}
              </div>
            </Card>
          );
        })}
      </div>

      <p className="mt-6 text-center text-xs text-ink-tertiary">
        {cfg.data && caps.data
          ? `Console · API v${caps.data.api_version}`
          : "Quant Operator Console"}
      </p>
    </>
  );
}
