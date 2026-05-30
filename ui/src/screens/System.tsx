import { Activity, Cpu, Database, HardDrive, MemoryStick, Microchip, Server } from "lucide-react";
import type { ReactNode } from "react";
import { useSeries } from "../lib/history";
import { useLive, useSystemStatus } from "../lib/queries";
import { cn } from "../lib/cn";
import { fmtBytes, fmtInt, fmtNum, fmtPct, fmtUptimeSince, titleCase } from "../lib/format";
import { isObj } from "../lib/objects";
import { Badge, EmptyState, ErrorCard, KeyValue, Skeleton, StatusLamp, type Tone } from "../components/ui/atoms";
import { Card, CardHeader } from "../components/ui/Card";
import { LiveArea, Meter } from "../components/ui/charts";
import { PageHeader } from "../components/ui/PageHeader";
import { api } from "../lib/api";
import { DISPLAY_LIMITS } from "../lib/uiConfig";

function pctTone(pct: number | null | undefined): Tone {
  if (pct == null) return "neutral";
  if (pct >= 90) return "danger";
  if (pct >= 75) return "warn";
  return "accent";
}

function LiveMetric({
  label,
  icon,
  value,
  seriesKey,
  tone,
}: {
  label: string;
  icon: ReactNode;
  value: string;
  seriesKey: string;
  tone: Tone;
}) {
  const series = useSeries(seriesKey);
  return (
    <Card>
      <CardHeader title={label} icon={icon} />
      <div className="text-metric font-semibold leading-none tracking-tight text-ink tnum">{value}</div>
      <div className="mt-2">
        <LiveArea data={series} tone={tone} height={90} />
      </div>
    </Card>
  );
}

export default function System() {
  const sys = useSystemStatus();
  const health = useLive(["health-ready"], api.healthReady);
  const s = sys.data;
  const gpu = s?.gpus?.[0];

  if (!s && sys.isLoading) {
    return (
      <>
        <PageHeader title="System" subtitle="Host hardware & services" />
        <div className="grid gap-4 sm:grid-cols-3">
          {Array.from({ length: DISPLAY_LIMITS.systemLoadingCards }).map((_, i) => (
            <Skeleton key={i} className="h-40" />
          ))}
        </div>
      </>
    );
  }
  if (!s) {
    return (
      <>
        <PageHeader title="System" />
        <ErrorCard message={(sys.error as Error)?.message ?? "System status unavailable"} onRetry={() => sys.refetch()} />
      </>
    );
  }

  const checks = isObj(health.data?.checks) ? (health.data!.checks as Record<string, unknown>) : {};

  return (
    <>
      <PageHeader
        title="System"
        subtitle="Host hardware & services"
        right={
          <span className="hidden text-xs text-ink-tertiary sm:inline">
            {s.hostname} · {s.platform.replace(/-SP\d+$/, "")} · py{s.python}
          </span>
        }
      />

      {/* Live graphs */}
      <div className="grid gap-4 sm:grid-cols-3">
        <LiveMetric
          label="CPU"
          icon={<Cpu size={15} />}
          value={s.cpu ? fmtPct(s.cpu.percent, 0, true) : "—"}
          seriesKey="cpu_pct"
          tone={pctTone(s.cpu?.percent)}
        />
        <LiveMetric
          label="Memory"
          icon={<MemoryStick size={15} />}
          value={s.memory ? fmtPct(s.memory.percent, 0, true) : "—"}
          seriesKey="mem_pct"
          tone={pctTone(s.memory?.percent)}
        />
        <LiveMetric
          label="GPU"
          icon={<Microchip size={15} />}
          value={gpu?.utilization_pct != null ? `${gpu.utilization_pct}%` : "—"}
          seriesKey="gpu_util"
          tone={pctTone(gpu?.utilization_pct)}
        />
      </div>

      {/* Detail cards */}
      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        {/* CPU detail */}
        <Card>
          <CardHeader
            title="Processor"
            icon={<Cpu size={15} />}
            right={<Badge tone="neutral">{fmtInt(s.cpu?.logical)} threads</Badge>}
          />
          {s.cpu ? (
            <>
              <KeyValue k="Physical cores">{fmtInt(s.cpu.physical)}</KeyValue>
              <KeyValue k="Logical cores">{fmtInt(s.cpu.logical)}</KeyValue>
              <KeyValue k="Load">{fmtPct(s.cpu.percent, 1, true)}</KeyValue>
              <div className="mt-2 flex flex-wrap gap-1">
                {s.cpu.per_core.map((c, i) => (
                  <div
                    key={i}
                    title={`core ${i}: ${c.toFixed(0)}%`}
                    className="h-8 w-2.5 overflow-hidden rounded-sm bg-hairline/10"
                  >
                    <div
                      className={cn(
                        "w-full",
                        c >= 90 ? "bg-danger" : c >= 75 ? "bg-warn" : "bg-accent",
                      )}
                      style={{ height: `${Math.max(4, c)}%`, marginTop: `${100 - Math.max(4, c)}%` }}
                    />
                  </div>
                ))}
              </div>
            </>
          ) : (
            <EmptyState label="CPU metrics unavailable" />
          )}
        </Card>

        {/* Memory + disk */}
        <Card>
          <CardHeader title="Memory & disk" icon={<HardDrive size={15} />} />
          {s.memory && (
            <div className="mb-3">
              <div className="mb-1 flex justify-between text-[13px]">
                <span className="text-ink-secondary">RAM</span>
                <span className="text-ink-tertiary tnum">
                  {fmtBytes(s.memory.used)} / {fmtBytes(s.memory.total)}
                </span>
              </div>
              <Meter value={s.memory.percent} min={0} max={100} tone={pctTone(s.memory.percent)} />
            </div>
          )}
          {s.disk && (
            <div>
              <div className="mb-1 flex justify-between text-[13px]">
                <span className="text-ink-secondary">Disk</span>
                <span className="text-ink-tertiary tnum">
                  {fmtBytes(s.disk.used)} / {fmtBytes(s.disk.total)}
                </span>
              </div>
              <Meter value={s.disk.percent} min={0} max={100} tone={pctTone(s.disk.percent)} />
            </div>
          )}
        </Card>

        {/* GPU */}
        <Card>
          <CardHeader title="GPU" icon={<Microchip size={15} />} hint={gpu?.name} />
          {gpu ? (
            <>
              {gpu.memory_used_mb != null && gpu.memory_total_mb != null && (
                <div className="mb-3">
                  <div className="mb-1 flex justify-between text-[13px]">
                    <span className="text-ink-secondary">VRAM</span>
                    <span className="text-ink-tertiary tnum">
                      {fmtNum(gpu.memory_used_mb, 0)} / {fmtNum(gpu.memory_total_mb, 0)} MB
                    </span>
                  </div>
                  <Meter
                    value={gpu.memory_used_mb}
                    min={0}
                    max={gpu.memory_total_mb}
                    tone="accent"
                  />
                </div>
              )}
              <KeyValue k="Utilization">{gpu.utilization_pct != null ? `${gpu.utilization_pct}%` : "—"}</KeyValue>
              <KeyValue k="Temperature">{gpu.temperature_c != null ? `${gpu.temperature_c} °C` : "—"}</KeyValue>
              <KeyValue k="Power">{gpu.power_w != null ? `${gpu.power_w} W` : "—"}</KeyValue>
            </>
          ) : (
            <EmptyState label="No GPU detected" />
          )}
        </Card>

        {/* Process */}
        <Card>
          <CardHeader title="API process" icon={<Activity size={15} />} />
          {s.process ? (
            <>
              <KeyValue k="PID">{fmtInt(s.process.pid)}</KeyValue>
              <KeyValue k="Resident memory">{fmtBytes(s.process.rss)}</KeyValue>
              <KeyValue k="Threads">{fmtInt(s.process.threads)}</KeyValue>
              <KeyValue k="Uptime">{fmtUptimeSince(s.process.create_time)}</KeyValue>
            </>
          ) : (
            <EmptyState label="Process metrics unavailable" />
          )}
        </Card>

        {/* Services */}
        <Card>
          <CardHeader
            title="Services"
            icon={<Database size={15} />}
            right={
              <Badge tone={health.data?.status === "ready" ? "success" : "warn"}>
                {health.data?.status ?? "—"}
              </Badge>
            }
          />
          {Object.keys(checks).length ? (
            Object.entries(checks).map(([k, v]) => {
              const val = String(v);
              const tone: Tone = val === "ok" ? "success" : val === "error" ? "danger" : "neutral";
              return (
                <div key={k} className="flex items-center justify-between py-1.5">
                  <span className="text-[13px] text-ink-secondary">{titleCase(k)}</span>
                  <div className="flex items-center gap-2">
                    <StatusLamp tone={tone} size={7} />
                    <span className="text-[13px] text-ink tnum">{val}</span>
                  </div>
                </div>
              );
            })
          ) : (
            <EmptyState label="No service checks" />
          )}
        </Card>

        {/* Host */}
        <Card>
          <CardHeader title="Host" icon={<Server size={15} />} />
          <KeyValue k="Hostname">{s.hostname}</KeyValue>
          <KeyValue k="Platform" mono>
            {s.platform}
          </KeyValue>
          <KeyValue k="Python">{s.python}</KeyValue>
          <KeyValue k="psutil">{s.psutil_available ? "available" : "missing"}</KeyValue>
        </Card>
      </div>
    </>
  );
}
