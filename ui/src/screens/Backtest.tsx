import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CalendarRange, FlaskConical, Play, RotateCcw, TrendingDown, TrendingUp } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../lib/api";
import { cn } from "../lib/cn";
import {
  buildCommandPayload,
  findCommandArg,
  findCommandByArgSignature,
  initCommandValues,
  isMissingRequiredCommandArg,
  previewCommandArgv,
  type CommandValues,
} from "../lib/commands";
import { fmtNum, fmtPct, fmtSignedPct, titleCase } from "../lib/format";
import type { BacktestPoint, BacktestRun, CommandArg } from "../lib/types";
import { BACKTEST_COMMAND_SIGNATURE, DISPLAY_LIMITS, FORMATS, QUERY_TIMING, REPLAY_TIMING } from "../lib/uiConfig";
import { Badge, Button, EmptyState, Pill, Skeleton, StatusLamp } from "../components/ui/atoms";
import { Card, CardHeader } from "../components/ui/Card";
import { ReplayArea } from "../components/backtest/ReplayArea";
import { PageHeader } from "../components/ui/PageHeader";

const BACKTEST_DESTS = {
  contractsFile: BACKTEST_COMMAND_SIGNATURE.requiredDests[0],
  start: BACKTEST_COMMAND_SIGNATURE.requiredDests[1],
  end: BACKTEST_COMMAND_SIGNATURE.requiredDests[2],
  modelVersion: BACKTEST_COMMAND_SIGNATURE.requiredDests[3],
  topN: BACKTEST_COMMAND_SIGNATURE.preferredDests[0],
} as const;

function todayIso(): string {
  return new Date().toISOString().slice(0, FORMATS.isoDateLength);
}

/** Reveal a growing slice of `total` points to draw in the curve. */
function useReplay(total: number, restartKey: string | null): { revealed: number; replay: () => void } {
  const [revealed, setRevealed] = useState(total);
  const rafRef = useRef<number>();
  const play = () => {
    cancelAnimationFrame(rafRef.current ?? QUERY_TIMING.immediateMs);
    if (!total) {
      setRevealed(0);
      return;
    }
    const t0 = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / REPLAY_TIMING.backtestCurveMs);
      const eased = 1 - Math.pow(1 - p, 3);
      setRevealed(Math.max(1, Math.round(eased * total)));
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  };
  useEffect(() => {
    play();
    return () => cancelAnimationFrame(rafRef.current ?? QUERY_TIMING.immediateMs);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [restartKey, total]);
  return { revealed, replay: play };
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-wide text-ink-tertiary">{label}</p>
      <p className={cn("mt-0.5 text-[19px] font-semibold tnum", tone ?? "text-ink")}>{value}</p>
    </div>
  );
}

function valuesEqual(a: CommandValues, b: CommandValues): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const key of keys) if (a[key] !== b[key]) return false;
  return true;
}

function CommandArgInput({
  arg,
  value,
  onChange,
  type = "text",
  min,
  max,
}: {
  arg: CommandArg | null;
  value: string | boolean | undefined;
  onChange: (value: string) => void;
  type?: "date" | "text";
  min?: string;
  max?: string;
}) {
  if (!arg) return null;
  const numeric = arg.type === "int" || arg.type === "float" || arg.type === "decimal";
  return (
    <label className="block">
      <span className="mb-1 flex items-center gap-1.5 text-[13px] text-ink-secondary">
        {titleCase(arg.dest)}
        {arg.required && <span className="rounded bg-danger/12 px-1.5 py-0.5 text-[10px] font-semibold text-danger">Required</span>}
      </span>
      <input
        type={type}
        value={String(value ?? "")}
        min={min}
        max={max}
        inputMode={type === "date" ? undefined : numeric ? "decimal" : "text"}
        placeholder={arg.metavar ?? arg.type}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-hairline/10 bg-base/60 px-2.5 py-1.5 text-[13px] text-ink outline-none focus:border-accent/50"
      />
      {arg.help && <span className="mt-1 block text-[11px] leading-snug text-ink-tertiary">{arg.help}</span>}
    </label>
  );
}

function rangeSuggestions(runs: BacktestRun[]) {
  const seen = new Set<string>();
  const out: Array<{ key: string; label: string; start: string; end: string }> = [];
  for (const run of runs) {
    const key = `${run.date_start}:${run.date_end}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({
      key,
      label: run.arm || `${run.date_start} to ${run.date_end}`,
      start: run.date_start,
      end: run.date_end,
    });
    if (out.length >= DISPLAY_LIMITS.backtestRangeSuggestions) break;
  }
  return out;
}

export default function Backtest() {
  const qc = useQueryClient();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [launchedJob, setLaunchedJob] = useState<string | null>(null);
  const [values, setValues] = useState<CommandValues>({});
  const touchedRef = useRef<Set<string>>(new Set());
  const maxDate = todayIso();

  const catalogQ = useQuery({
    queryKey: ["commands"],
    queryFn: ({ signal }) => api.commands(signal),
    staleTime: QUERY_TIMING.commandCatalogStaleMs,
    retry: false,
  });
  const brokerQ = useQuery({
    queryKey: ["broker-connection"],
    queryFn: ({ signal }) => api.brokerConnection(undefined, signal),
    staleTime: QUERY_TIMING.relaxedRefetchMs,
    retry: false,
  });
  const modelsQ = useQuery({
    queryKey: ["model-registry"],
    queryFn: ({ signal }) => api.modelRegistry(signal),
    staleTime: QUERY_TIMING.standardStaleMs,
    retry: false,
  });
  const runsQ = useQuery({
    queryKey: ["backtest-runs"],
    queryFn: ({ signal }) => api.backtestRuns(signal),
    staleTime: QUERY_TIMING.standardStaleMs,
  });

  const allRuns = runsQ.data?.runs ?? [];
  const backtestCommand = useMemo(
    () => findCommandByArgSignature(catalogQ.data, BACKTEST_COMMAND_SIGNATURE),
    [catalogQ.data],
  );
  const commandKey = backtestCommand?.path.join(" ") ?? "";
  const executionEnabled = catalogQ.data?.execution_enabled ?? false;
  const activeModelVersion = useMemo(
    () =>
      modelsQ.data?.models.find((model) => model.active && model.model_version)?.model_version
      ?? modelsQ.data?.models.find((model) => model.model_version)?.model_version
      ?? null,
    [modelsQ.data],
  );

  useEffect(() => {
    touchedRef.current.clear();
    setValues(backtestCommand ? initCommandValues(backtestCommand) : {});
  }, [commandKey]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!backtestCommand) return;
    setValues((prev) => {
      const next = Object.keys(prev).length ? { ...initCommandValues(backtestCommand), ...prev } : initCommandValues(backtestCommand);
      const fillIfEmpty = (dest: string, suggested: string | null | undefined) => {
        if (!suggested || touchedRef.current.has(dest) || !(dest in next)) return;
        if (next[dest] === "" || next[dest] == null) next[dest] = suggested;
      };
      fillIfEmpty(BACKTEST_DESTS.contractsFile, brokerQ.data?.contracts_file ?? null);
      fillIfEmpty(BACKTEST_DESTS.modelVersion, activeModelVersion);
      fillIfEmpty(BACKTEST_DESTS.start, allRuns[0]?.date_start);
      fillIfEmpty(BACKTEST_DESTS.end, allRuns[0]?.date_end);
      return valuesEqual(prev, next) ? prev : next;
    });
  }, [
    activeModelVersion,
    allRuns,
    backtestCommand,
    brokerQ.data?.contracts_file,
  ]);

  const setValue = useCallback((dest: string, value: string | boolean) => {
    touchedRef.current.add(dest);
    setValues((prev) => ({ ...prev, [dest]: value }));
  }, []);

  const start = String(values[BACKTEST_DESTS.start] ?? "");
  const end = String(values[BACKTEST_DESTS.end] ?? "");
  const runs = useMemo(
    () => allRuns.filter((run) => (!end || run.date_start <= end) && (!start || run.date_end >= start)),
    [allRuns, start, end],
  );
  const suggestions = useMemo(() => rangeSuggestions(allRuns), [allRuns]);

  useEffect(() => {
    if (runs.length && (!selectedId || !runs.some((r) => r.id === selectedId))) {
      setSelectedId(runs[0].id);
    } else if (!runs.length && selectedId) {
      setSelectedId(null);
    }
  }, [runs, selectedId]);

  const resultQ = useQuery({
    queryKey: ["backtest-result", selectedId],
    queryFn: ({ signal }) => api.backtestResult(selectedId!, signal),
    enabled: !!selectedId,
    staleTime: QUERY_TIMING.standardStaleMs,
  });
  const result = resultQ.data;
  const points = result?.points ?? [];
  const { revealed, replay } = useReplay(points.length, selectedId);
  const shown = points.slice(0, revealed);

  const equityData = shown.map((p: BacktestPoint) => ({ date: p.date, value: p.equity }));
  const ddData = shown.map((p: BacktestPoint) => ({ date: p.date, value: p.drawdown }));
  const icData = shown.map((p: BacktestPoint) => ({ date: p.date, value: Number(p.ic ?? 0) }));

  const curEquity = shown.length ? shown[shown.length - 1].equity : 1;
  const curReturn = curEquity - 1;
  const curDD = shown.length ? Math.min(...shown.map((p) => p.drawdown)) : 0;
  const m = result?.metrics;

  const payload = useMemo(
    () => (backtestCommand ? buildCommandPayload(backtestCommand, values) : {}),
    [backtestCommand, values],
  );
  const preview = backtestCommand ? previewCommandArgv(backtestCommand, values) : "";
  const requiredArgs = backtestCommand?.args?.filter((arg) => arg.required && arg.kind !== "flag") ?? [];
  const missing = requiredArgs.filter((arg) => isMissingRequiredCommandArg(arg, values[arg.dest] ?? ""));
  const canRun = executionEnabled && !!backtestCommand && missing.length === 0 && !backtestCommand.dangerous;
  const runDisabledReason = !backtestCommand
    ? "Backtest command is not advertised by the command catalog"
    : backtestCommand.dangerous
      ? "Use the Commands dashboard for catalog commands that require typed confirmation"
      : !executionEnabled
        ? "Enable command execution to launch a fresh run"
        : missing.length
          ? `Missing ${missing.map((arg) => titleCase(arg.dest)).join(", ")}`
          : null;

  const runMut = useMutation({
    mutationFn: () => {
      if (!backtestCommand) throw new Error("Backtest command unavailable");
      return api.runCommand(backtestCommand.path, payload);
    },
    onSuccess: (job) => {
      setLaunchedJob(job.id);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const launchedQ = useQuery({
    queryKey: ["job", launchedJob],
    queryFn: ({ signal }) => api.job(launchedJob!, QUERY_TIMING.immediateMs, signal),
    enabled: !!launchedJob,
    refetchInterval: QUERY_TIMING.focusedJobPollMs,
  });
  useEffect(() => {
    if (launchedQ.data && ["succeeded", "failed", "cancelled"].includes(launchedQ.data.status)) {
      qc.invalidateQueries({ queryKey: ["backtest-runs"] });
    }
  }, [launchedQ.data?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const applyRange = (range: { start: string; end: string }) => {
    touchedRef.current.add(BACKTEST_DESTS.start);
    touchedRef.current.add(BACKTEST_DESTS.end);
    setValues((prev) => ({
      ...prev,
      [BACKTEST_DESTS.start]: range.start,
      [BACKTEST_DESTS.end]: range.end,
    }));
  };

  return (
    <>
      <PageHeader
        title="Backtest"
        subtitle="Walk-forward equity, replayed live"
        right={
          <Pill tone={executionEnabled ? "success" : "warn"}>
            {executionEnabled ? "Execution enabled" : "View-only"}
          </Pill>
        }
      />

      <div className="grid gap-4 lg:grid-cols-12">
        <div className="space-y-4 lg:col-span-4">
          <Card index={0}>
            <CardHeader
              title="Run a backtest"
              icon={<CalendarRange size={15} />}
              hint={backtestCommand ? backtestCommand.path.join(" ") : undefined}
            />
            {catalogQ.isLoading ? (
              <Skeleton className="h-64" />
            ) : backtestCommand ? (
              <>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <CommandArgInput
                    arg={findCommandArg(backtestCommand, BACKTEST_DESTS.contractsFile)}
                    value={values[BACKTEST_DESTS.contractsFile]}
                    onChange={(value) => setValue(BACKTEST_DESTS.contractsFile, value)}
                  />
                  <CommandArgInput
                    arg={findCommandArg(backtestCommand, BACKTEST_DESTS.modelVersion)}
                    value={values[BACKTEST_DESTS.modelVersion]}
                    onChange={(value) => setValue(BACKTEST_DESTS.modelVersion, value)}
                  />
                  <CommandArgInput
                    arg={findCommandArg(backtestCommand, BACKTEST_DESTS.start)}
                    type="date"
                    value={values[BACKTEST_DESTS.start]}
                    max={end || maxDate}
                    onChange={(value) => setValue(BACKTEST_DESTS.start, value)}
                  />
                  <CommandArgInput
                    arg={findCommandArg(backtestCommand, BACKTEST_DESTS.end)}
                    type="date"
                    value={values[BACKTEST_DESTS.end]}
                    min={start || undefined}
                    max={maxDate}
                    onChange={(value) => setValue(BACKTEST_DESTS.end, value)}
                  />
                  <CommandArgInput
                    arg={findCommandArg(backtestCommand, BACKTEST_DESTS.topN)}
                    value={values[BACKTEST_DESTS.topN]}
                    onChange={(value) => setValue(BACKTEST_DESTS.topN, value)}
                  />
                </div>

                {suggestions.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {suggestions.map((range) => (
                      <button
                        key={range.key}
                        onClick={() => applyRange(range)}
                        className="rounded-md bg-hairline/10 px-2 py-1 text-[11px] text-ink-secondary hover:bg-hairline/[0.16]"
                        title={`${range.start} to ${range.end}`}
                      >
                        {range.label}
                      </button>
                    ))}
                  </div>
                )}

                <div className="mt-4 rounded-lg border border-hairline/[0.07] bg-base/40 p-3">
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <span className="text-[11px] font-semibold uppercase text-ink-tertiary">Command preview</span>
                    <Badge tone={missing.length ? "warn" : "success"}>
                      {missing.length ? `${missing.length} missing` : "ready"}
                    </Badge>
                  </div>
                  <code className="block break-words font-mono text-[11px] leading-relaxed text-ink-secondary">{preview}</code>
                </div>

                <div className="mt-4">
                  <Button
                    variant="primary"
                    onClick={() => runMut.mutate()}
                    disabled={!canRun || runMut.isPending}
                    className="w-full"
                  >
                    <Play size={14} /> {runMut.isPending ? "Starting..." : "Run backtest"}
                  </Button>
                  {runDisabledReason && (
                    <p className="mt-2 text-[11px] text-ink-tertiary">{runDisabledReason}. Saved runs below replay instantly.</p>
                  )}
                  {runMut.isError && (
                    <p className="mt-2 text-xs text-danger">
                      {runMut.error instanceof ApiError ? runMut.error.detail : "Failed to start"}
                    </p>
                  )}
                  {launchedQ.data && (
                    <div className="mt-2 flex items-center gap-2 text-xs">
                      <StatusLamp
                        tone={launchedQ.data.status === "succeeded" ? "success" : launchedQ.data.status === "failed" ? "danger" : "accent"}
                        pulse={!["succeeded", "failed", "cancelled"].includes(launchedQ.data.status)}
                        size={7}
                      />
                      <span className="text-ink-secondary">campaign job {launchedQ.data.status}</span>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <EmptyState label="Backtest command unavailable from catalog" />
            )}
          </Card>

          <Card index={1}>
            <CardHeader title="Saved runs" icon={<FlaskConical size={15} />} hint={`${runs.length} in range`} />
            {runsQ.isLoading ? (
              <Skeleton className="h-64" />
            ) : runs.length ? (
              <ul className="max-h-[44vh] space-y-1 overflow-y-auto pr-1">
                {runs.map((r: BacktestRun) => {
                  const active = r.id === selectedId;
                  return (
                    <li key={r.id}>
                      <button
                        onClick={() => setSelectedId(r.id)}
                        className={cn(
                          "w-full rounded-lg px-2.5 py-2 text-left transition-colors",
                          active ? "bg-accent/12" : "hover:bg-hairline/[0.06]",
                        )}
                      >
                        <div className="flex items-center gap-2">
                          <span className={cn("min-w-0 flex-1 truncate text-[12.5px] font-medium", active ? "text-accent" : "text-ink")}>
                            {r.arm}
                          </span>
                          <span className={cn("shrink-0 text-xs tnum", r.total_return >= 0 ? "text-success" : "text-danger")}>
                            {fmtSignedPct(r.total_return)}
                          </span>
                        </div>
                        <div className="mt-0.5 flex items-center gap-2 text-[11px] text-ink-tertiary">
                          <span>{r.date_start} to {r.date_end}</span>
                          <span>{r.n_folds} folds</span>
                        </div>
                      </button>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <EmptyState label="No saved runs overlap this range" />
            )}
          </Card>
        </div>

        <div className="space-y-4 lg:col-span-8">
          <Card index={2}>
            <CardHeader
              title="Equity curve"
              icon={<TrendingUp size={15} />}
              hint={result ? `${result.arm} · ${result.date_start} to ${result.date_end}` : undefined}
              right={
                <button
                  onClick={replay}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-hairline/10 px-2.5 py-1 text-xs text-ink-secondary hover:bg-hairline/[0.16]"
                >
                  <RotateCcw size={12} /> Replay
                </button>
              }
            />
            {resultQ.isLoading ? (
              <Skeleton className="h-56" />
            ) : result ? (
              <>
                <div className="mb-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
                  <Metric label="Total return" value={fmtSignedPct(curReturn)} tone={curReturn >= 0 ? "text-success" : "text-danger"} />
                  <Metric label="Sharpe (ann)" value={m?.sharpe_annualized != null ? fmtNum(m.sharpe_annualized, 2) : "—"} />
                  <Metric label="Max drawdown" value={fmtPct(curDD)} tone="text-danger" />
                  <Metric label="IC (60d)" value={m?.ic_60d != null ? fmtNum(m.ic_60d, 3) : "—"} />
                </div>
                <ReplayArea data={equityData} tone="accent" height={210} baseline={1} />
                <div className="mt-1 flex justify-between text-[11px] text-ink-tertiary">
                  <span>{result.date_start}</span>
                  <span>
                    {shown.length}/{points.length} folds · {shown.length ? shown[shown.length - 1].date : ""}
                  </span>
                  <span>{result.date_end}</span>
                </div>
              </>
            ) : (
              <EmptyState label="Select a saved run to replay" />
            )}
          </Card>

          {result && (
            <div className="grid gap-4 sm:grid-cols-2">
              <Card index={3}>
                <CardHeader title="Drawdown" icon={<TrendingDown size={15} />} />
                <ReplayArea data={ddData} tone="danger" height={150} baseline={0} />
              </Card>
              <Card index={4}>
                <CardHeader title="Per-fold IC" icon={<FlaskConical size={15} />} right={<Badge tone="neutral">mean {m?.mean_ic != null ? fmtNum(m.mean_ic, 3) : "—"}</Badge>} />
                <ReplayArea data={icData} tone="accent" height={150} baseline={0} />
              </Card>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
