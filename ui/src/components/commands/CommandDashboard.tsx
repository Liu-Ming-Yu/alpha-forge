import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  Clipboard,
  Clock,
  Copy,
  FileTerminal,
  History,
  Info,
  ListChecks,
  Loader2,
  Play,
  RotateCcw,
  Shield,
  Terminal,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../../lib/api";
import { cn } from "../../lib/cn";
import {
  buildCommandPayload,
  initCommandValues,
  isMissingRequiredCommandArg,
  previewCommandArgv,
  type CommandValues,
} from "../../lib/commands";
import { fmtAgo, fmtDuration, titleCase } from "../../lib/format";
import type { CommandNode, JobStatus } from "../../lib/types";
import { DISPLAY_LIMITS, QUERY_TIMING } from "../../lib/uiConfig";
import { Badge, Button, EmptyState, Pill, StatusLamp, type Tone } from "../ui/atoms";
import { Card, CardHeader } from "../ui/Card";
import { Modal } from "../ui/Dialog";
import { Field } from "./fields";
import { JobLogs } from "./JobsPanel";

const STORAGE_PREFIX = "qp.cmd.";
// Bump when the persisted shape changes. Persisted blobs from an older schema
// are discarded on load rather than reconciled: a full-value blob cannot tell an
// operator edit from a stale default, so honoring it would let a removed/renamed
// arg shadow the live catalog. Only the diff format (v2+) is trusted.
const STORAGE_SCHEMA = 2;

interface StoredValues {
  v: number;
  d: CommandValues; // only fields the operator changed from the live catalog default
}

/** Last-used values, reconciled against the live catalog: fresh defaults for
 *  every current arg, overlaid only with persisted operator edits for args that
 *  still exist. Renamed/removed args and stale defaults cannot leak through. */
function loadSaved(command: CommandNode): CommandValues {
  const base = initCommandValues(command);
  try {
    const raw = localStorage.getItem(STORAGE_PREFIX + command.path.join("/"));
    if (!raw) return base;
    const parsed = JSON.parse(raw) as Partial<StoredValues>;
    if (!parsed || parsed.v !== STORAGE_SCHEMA || typeof parsed.d !== "object" || parsed.d === null) {
      return base;
    }
    const diff = parsed.d as CommandValues;
    const reconciled: CommandValues = { ...base };
    for (const dest of Object.keys(base)) {
      if (Object.prototype.hasOwnProperty.call(diff, dest)) reconciled[dest] = diff[dest];
    }
    return reconciled;
  } catch {
    return base;
  }
}

/** Persist only the diff from the live default (and drop the entry entirely when
 *  nothing changed), so future loads reconcile cleanly against the catalog. */
function saveDiff(command: CommandNode, values: CommandValues): void {
  try {
    const base = initCommandValues(command);
    const diff: CommandValues = {};
    for (const dest of Object.keys(values)) {
      if (dest in base && values[dest] !== base[dest]) diff[dest] = values[dest];
    }
    const key = STORAGE_PREFIX + command.path.join("/");
    if (Object.keys(diff).length === 0) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify({ v: STORAGE_SCHEMA, d: diff } satisfies StoredValues));
  } catch {
    /* storage unavailable */
  }
}

const TERMINAL: JobStatus[] = ["succeeded", "failed", "cancelled"];
const STATUS_TONE: Record<JobStatus, Tone> = {
  queued: "warn",
  running: "accent",
  succeeded: "success",
  failed: "danger",
  cancelled: "neutral",
};
const STATUS_COPY: Record<JobStatus, string> = {
  queued: "Queued",
  running: "Running",
  succeeded: "Succeeded",
  failed: "Failed",
  cancelled: "Cancelled",
};

const TONE_BORDER: Record<Tone, string> = {
  neutral: "border-hairline/[0.08] bg-base/25 text-ink-secondary",
  accent: "border-accent/25 bg-accent/10 text-accent",
  success: "border-success/25 bg-success/10 text-success",
  warn: "border-warn/25 bg-warn/10 text-warn",
  danger: "border-danger/25 bg-danger/10 text-danger",
};

type StageState = "done" | "active" | "blocked" | "idle";

function commandTitle(command: CommandNode): string {
  return command.path.map(titleCase).join(" / ");
}

function changedValueCount(values: CommandValues, defaults: CommandValues): number {
  return Object.keys(values).filter((dest) => values[dest] !== defaults[dest]).length;
}

function stageClass(state: StageState): string {
  if (state === "done") return "border-success/35 bg-success/10 text-success";
  if (state === "active") return "border-accent/40 bg-accent/12 text-accent";
  if (state === "blocked") return "border-danger/35 bg-danger/10 text-danger";
  return "border-hairline/[0.08] bg-base/20 text-ink-tertiary";
}

function StageRail({
  missingCount,
  valid,
  validating,
  running,
}: {
  missingCount: number;
  valid: boolean;
  validating: boolean;
  running: boolean;
}) {
  const stages: Array<{ label: string; detail: string; state: StageState }> = [
    { label: "Choose", detail: "Command selected", state: "done" },
    {
      label: "Configure",
      detail: missingCount ? `${missingCount} required missing` : "Inputs complete",
      state: missingCount ? "active" : "done",
    },
    {
      label: "Review",
      detail: validating ? "Validating parser" : valid ? "Ready to run" : "Needs fixes",
      state: validating ? "active" : valid ? "done" : "blocked",
    },
    {
      label: "Run",
      detail: running ? "Job in progress" : "Start when ready",
      state: running ? "active" : "idle",
    },
  ];

  return (
    <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-4">
      {stages.map((stage, index) => (
        <div key={stage.label} className={cn("rounded-lg border px-3 py-2", stageClass(stage.state))}>
          <div className="flex items-center gap-2">
            <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-current/10 text-[11px] font-semibold">
              {index + 1}
            </span>
            <span className="text-[12px] font-semibold">{stage.label}</span>
          </div>
          <p className="mt-1 truncate text-[11px] opacity-80">{stage.detail}</p>
        </div>
      ))}
    </div>
  );
}

function readinessIcon(tone: Tone, pending?: boolean) {
  if (pending) return <Loader2 size={15} className="animate-spin" />;
  if (tone === "success") return <CheckCircle2 size={15} />;
  if (tone === "danger") return <XCircle size={15} />;
  if (tone === "warn") return <AlertTriangle size={15} />;
  return <Info size={15} />;
}

function ReadinessRow({
  label,
  detail,
  tone,
  pending,
}: {
  label: string;
  detail: string;
  tone: Tone;
  pending?: boolean;
}) {
  return (
    <div className={cn("flex items-start gap-3 rounded-xl border p-3", TONE_BORDER[tone])}>
      <span className="mt-0.5 shrink-0">{readinessIcon(tone, pending)}</span>
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-semibold text-ink">{label}</p>
        <p className="mt-0.5 break-words text-[12px] leading-snug text-ink-tertiary">{detail}</p>
      </div>
    </div>
  );
}

export function CommandDashboard({ command }: { command: CommandNode }) {
  const qc = useQueryClient();
  const pathKey = command.path.join(" ");
  const args = command.args ?? [];
  const required = args.filter((a) => a.required && a.kind !== "flag");
  const optional = args.filter((a) => !(a.required && a.kind !== "flag"));

  const catalogQ = useQuery({
    queryKey: ["commands"],
    queryFn: ({ signal }) => api.commands(signal),
    staleTime: QUERY_TIMING.commandCatalogStaleMs,
    retry: false,
  });
  const executionEnabled = catalogQ.data?.execution_enabled ?? false;

  const [values, setValues] = useState<CommandValues>(() => loadSaved(command));
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmText, setConfirmText] = useState("");
  const [activeJob, setActiveJob] = useState<string | null>(null);
  const [serverError, setServerError] = useState<string | null>(null);
  const [validating, setValidating] = useState(false);
  const [copied, setCopied] = useState(false);

  // Restore last-used args when switching commands (recognition over recall).
  useEffect(() => {
    setValues(loadSaved(command));
    setConfirmOpen(false);
    setConfirmText("");
    setActiveJob(null);
    setServerError(null);
    setCopied(false);
  }, [pathKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist edits per command (diff-only, reconciled against the live catalog).
  useEffect(() => {
    saveDiff(command, values);
  }, [values, pathKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const defaults = useMemo(() => initCommandValues(command), [command]);
  const preview = useMemo(() => previewCommandArgv(command, values), [command, values]);
  const payload = useMemo(() => buildCommandPayload(command, values), [command, values]);
  const missing = required.filter((a) => isMissingRequiredCommandArg(a, values[a.dest] ?? ""));
  const changedCount = changedValueCount(values, defaults);
  const valuesKey = JSON.stringify(values);

  // Server-side argparse validation, debounced (error prevention).
  useEffect(() => {
    let cancelled = false;
    setValidating(true);
    const timer = setTimeout(() => {
      api
        .validateCommand(command.path, payload)
        .then((r) => {
          if (!cancelled) setServerError(r.ok ? null : r.error);
        })
        .catch(() => {
          if (!cancelled) setServerError(null);
        })
        .finally(() => {
          if (!cancelled) setValidating(false);
        });
    }, QUERY_TIMING.commandValidationDebounceMs);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [pathKey, valuesKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const valid = missing.length === 0 && !serverError;

  const mutation = useMutation({
    mutationFn: (confirm: string) => api.runCommand(command.path, payload, confirm),
    onSuccess: (job) => {
      setConfirmOpen(false);
      setConfirmText("");
      setActiveJob(job.id);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const jobsQ = useQuery({
    queryKey: ["jobs"],
    queryFn: ({ signal }) => api.jobs(signal),
    refetchInterval: QUERY_TIMING.jobPollMs,
  });
  const history = useMemo(
    () => (jobsQ.data?.jobs ?? []).filter((j) => j.command === pathKey),
    [jobsQ.data, pathKey],
  );
  const runningJob = history.find((j) => !TERMINAL.includes(j.status));
  const shownJob = activeJob ?? history[0]?.id ?? null;
  const latestJob = history[0] ?? null;

  const runError =
    mutation.isError && mutation.error instanceof ApiError
      ? mutation.error.detail
      : mutation.isError
        ? "Failed to start command"
        : null;

  const doRun = useCallback(() => {
    if (!executionEnabled || !valid || mutation.isPending) return;
    if (command.dangerous) setConfirmOpen(true);
    else mutation.mutate("");
  }, [executionEnabled, valid, mutation, command.dangerous]);

  const cancelRunning = useCallback(() => {
    if (!runningJob) return;
    void api.cancelJob(runningJob.id).finally(() => qc.invalidateQueries({ queryKey: ["jobs"] }));
  }, [qc, runningJob]);

  const copyPreview = useCallback(() => {
    if (!navigator.clipboard) return;
    void navigator.clipboard.writeText(preview).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), QUERY_TIMING.clipboardResetMs);
    });
  }, [preview]);

  // Keyboard: Ctrl/Cmd+Enter to run, Esc to cancel a running job.
  const runningRef = useRef(runningJob);
  runningRef.current = runningJob;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const editing = ["INPUT", "TEXTAREA", "SELECT"].includes(t?.tagName ?? "");
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        doRun();
      } else if (e.key === "Escape" && runningRef.current && !editing) {
        void api.cancelJob(runningRef.current.id).finally(() => qc.invalidateQueries({ queryKey: ["jobs"] }));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [doRun, qc]);

  const requiredDetail = required.length
    ? missing.length
      ? `${missing.length} of ${required.length} required inputs still need values`
      : `${required.length} required input${required.length === 1 ? "" : "s"} complete`
    : "This command has no required inputs";
  const validationDetail = validating
    ? "Checking the current values against the real argparse parser"
    : serverError
      ? serverError
      : "Current values parse cleanly";
  const runDisabledReason = !executionEnabled
    ? "Execution is disabled by server configuration"
    : !valid
      ? "Resolve readiness checks before running"
      : mutation.isPending
        ? "Starting job"
        : command.dangerous
          ? "Ready, with typed confirmation required"
          : "Ready to run";

  return (
    <>
      <section className="mb-4 rounded-xl border border-hairline/[0.07] bg-surface/75 p-5 shadow-card">
        <div className="flex flex-wrap items-start gap-3">
          <div className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-accent/14 text-accent">
            <FileTerminal size={20} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <h2 className="truncate text-title font-semibold text-ink">{commandTitle(command)}</h2>
              {command.dangerous && (
                <Badge tone="danger">
                  <AlertTriangle size={11} /> confirmation
                </Badge>
              )}
              {command.long_running && (
                <Badge tone="warn">
                  <Clock size={11} /> cancellable
                </Badge>
              )}
            </div>
            <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-ink-secondary">
              {command.help || "Configure inputs, review the reconstructed argv, and track the job output here."}
            </p>
            <code className="mt-3 inline-flex max-w-full rounded-lg bg-base/55 px-2.5 py-1.5 font-mono text-[12px] text-ink-secondary">
              <span className="truncate">{pathKey}</span>
            </code>
          </div>
          <Pill tone={valid && executionEnabled ? "success" : valid ? "warn" : "danger"} className="shrink-0">
            {valid && executionEnabled ? "Runnable" : valid ? "Catalog only" : "Needs input"}
          </Pill>
        </div>
        <StageRail missingCount={missing.length} valid={valid} validating={validating} running={!!runningJob} />
      </section>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        <div className="space-y-4 xl:col-span-7">
          <Card className="overflow-hidden p-0">
            <div className="flex flex-wrap items-center gap-3 border-b border-hairline/[0.07] px-5 py-4">
              <div className="min-w-0 flex-1">
                <h3 className="text-[15px] font-semibold text-ink">Configure inputs</h3>
                <p className="mt-0.5 text-xs text-ink-tertiary">
                  Defaults are prefilled. Edits are saved per command and reconciled with the live catalog.
                </p>
              </div>
              {changedCount > 0 && <Badge tone="accent">{changedCount} changed</Badge>}
              <Button variant="ghost" onClick={() => setValues(initCommandValues(command))} title="Reset to defaults">
                <RotateCcw size={13} /> Reset
              </Button>
            </div>
            <div className="p-5">
              {args.length === 0 ? (
                <EmptyState icon={<ListChecks size={20} />} label="This command takes no arguments." />
              ) : (
                <div className="space-y-5">
                  {required.length > 0 && (
                    <section>
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <div>
                          <p className="text-[12px] font-semibold uppercase text-ink-secondary">Required inputs</p>
                          <p className="mt-0.5 text-[11px] text-ink-tertiary">{requiredDetail}</p>
                        </div>
                      </div>
                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                        {required.map((a) => (
                          <Field
                            key={a.dest}
                            arg={a}
                            value={values[a.dest] ?? ""}
                            invalid={isMissingRequiredCommandArg(a, values[a.dest] ?? "")}
                            onChange={(v) => setValues((s) => ({ ...s, [a.dest]: v }))}
                          />
                        ))}
                      </div>
                    </section>
                  )}

                  {optional.length > 0 && (
                    <details className="group" open={required.length === 0 || optional.length <= DISPLAY_LIMITS.commandOptionalExpandedArgs}>
                      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 rounded-xl border border-hairline/[0.06] bg-base/20 px-3 py-2.5 text-[12px] font-semibold uppercase text-ink-secondary transition-colors hover:bg-hairline/[0.04]">
                        <span>Optional settings</span>
                        <span className="text-[11px] font-medium normal-case text-ink-tertiary">
                          {optional.length} available
                        </span>
                      </summary>
                      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                        {optional.map((a) => (
                          <Field
                            key={a.dest}
                            arg={a}
                            value={values[a.dest] ?? ""}
                            onChange={(v) => setValues((s) => ({ ...s, [a.dest]: v }))}
                          />
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              )}
            </div>
          </Card>

          <Card>
            <CardHeader
              title="Review command"
              icon={<Terminal size={15} />}
              hint={runDisabledReason}
              right={<Badge tone={valid ? "success" : "danger"}>{valid ? "Parser ready" : "Blocked"}</Badge>}
            />
            <div className="rounded-xl border border-hairline/[0.07] bg-base/45 p-3">
              <div className="mb-2 flex items-center justify-between gap-2 text-[11px] uppercase text-ink-tertiary">
                <span className="flex items-center gap-1.5">
                  <Clipboard size={12} /> Reconstructed argv
                </span>
                <button
                  type="button"
                  onClick={copyPreview}
                  className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-[11px] font-medium text-ink-tertiary transition-colors hover:bg-hairline/[0.08] hover:text-ink-secondary"
                >
                  <Copy size={12} /> {copied ? "Copied" : "Copy"}
                </button>
              </div>
              <code className="block max-h-28 overflow-auto break-all font-mono text-xs leading-relaxed text-ink-secondary">
                {preview}
              </code>
            </div>

            {(serverError || runError) && (
              <p className="mt-3 rounded-lg border border-danger/20 bg-danger/5 px-3 py-2 text-xs text-danger">
                {runError ?? serverError}
              </p>
            )}

            <div className="mt-4 flex flex-wrap items-center gap-2">
              <Button
                variant={command.dangerous ? "danger" : "primary"}
                onClick={doRun}
                disabled={!executionEnabled || !valid || mutation.isPending}
              >
                <Play size={14} /> {mutation.isPending ? "Starting..." : command.dangerous ? "Review and run" : "Run"}
              </Button>
              {runningJob && (
                <Button variant="ghost" onClick={cancelRunning}>
                  <Ban size={13} /> Cancel running job
                </Button>
              )}
              <span className="ml-auto hidden text-[11px] text-ink-tertiary sm:inline">
                Ctrl/Cmd+Enter run - Esc cancel
              </span>
            </div>
            {!executionEnabled && (
              <Pill tone="warn" className="mt-3">
                Set QP__API__ENABLE_COMMAND_EXECUTION=true to launch jobs
              </Pill>
            )}
          </Card>
        </div>

        <aside className="space-y-4 xl:col-span-5">
          <Card>
            <CardHeader
              title="Run readiness"
              icon={<Shield size={15} />}
              hint="Visible checks before anything launches"
            />
            <div className="space-y-2">
              <ReadinessRow
                label="Execution gate"
                detail={executionEnabled ? "The API is accepting command jobs." : "This instance is browsing-only until command execution is enabled."}
                tone={executionEnabled ? "success" : "warn"}
              />
              <ReadinessRow
                label="Required inputs"
                detail={requiredDetail}
                tone={missing.length ? "danger" : "success"}
              />
              <ReadinessRow
                label="Parser validation"
                detail={validationDetail}
                tone={validating ? "accent" : serverError ? "danger" : "success"}
                pending={validating}
              />
              <ReadinessRow
                label="Safety model"
                detail={
                  command.dangerous
                    ? "This command can mutate durable state or broker-facing systems. A typed RUN confirmation is required."
                    : command.long_running
                      ? "This command can keep running; the job can be cancelled from the output panel."
                      : "No extra confirmation is required for this command."
                }
                tone={command.dangerous ? "danger" : command.long_running ? "warn" : "success"}
              />
            </div>
            {latestJob && (
              <div className="mt-4 rounded-xl border border-hairline/[0.07] bg-base/30 p-3">
                <div className="flex items-center gap-2">
                  <StatusLamp tone={STATUS_TONE[latestJob.status]} pulse={!TERMINAL.includes(latestJob.status)} size={8} />
                  <p className="text-[13px] font-semibold text-ink">{STATUS_COPY[latestJob.status]}</p>
                  <span className="ml-auto text-[11px] text-ink-tertiary">
                    {fmtAgo(new Date(latestJob.created_at * 1000).toISOString())}
                  </span>
                </div>
                {latestJob.error && <p className="mt-2 text-xs text-danger">{latestJob.error}</p>}
              </div>
            )}
          </Card>

          <Card className="min-h-[29rem]">
            <CardHeader
              title="Live output"
              icon={<History size={15} />}
              hint={`${history.length} run${history.length === 1 ? "" : "s"} for this command`}
              right={runningJob ? <Badge tone="accent">Streaming</Badge> : null}
            />
            {history.length > 0 && (
              <ul className="mb-3 max-h-36 space-y-1 overflow-y-auto pr-1">
                {history.map((j) => {
                  const dur =
                    j.finished_at && j.started_at ? fmtDuration(j.finished_at - j.started_at) : null;
                  return (
                    <li key={j.id}>
                      <button
                        onClick={() => setActiveJob(j.id)}
                        className={cn(
                          "flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left transition-colors",
                          j.id === shownJob ? "bg-accent/12" : "hover:bg-hairline/[0.06]",
                        )}
                      >
                        <StatusLamp tone={STATUS_TONE[j.status]} pulse={!TERMINAL.includes(j.status)} size={7} />
                        <span className="min-w-0 flex-1 truncate text-xs font-medium text-ink-secondary">
                          {STATUS_COPY[j.status]}
                        </span>
                        {j.exit_code != null && <span className="text-[11px] text-ink-tertiary">exit {j.exit_code}</span>}
                        {dur && <span className="text-[11px] text-ink-tertiary">{dur}</span>}
                        <span className="text-[11px] text-ink-tertiary">
                          {fmtAgo(new Date(j.created_at * 1000).toISOString())}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
            {shownJob ? (
              <JobLogs jobId={shownJob} />
            ) : (
              <EmptyState icon={<History size={20} />} label="Run the command to stream output and status here." />
            )}
          </Card>
        </aside>
      </div>

      <Modal open={confirmOpen} onClose={() => setConfirmOpen(false)} labelledBy="cmd-confirm">
        <h3 id="cmd-confirm" className="text-title font-semibold text-ink">
          Confirm command launch
        </h3>
        <p className="mt-2 text-sm text-ink-secondary">
          <code className="rounded bg-hairline/15 px-1.5 py-0.5 font-mono text-xs">{pathKey}</code> is flagged as
          dangerous because it can mutate durable state or broker-facing systems. Type{" "}
          <code className="rounded bg-hairline/15 px-1.5 py-0.5 font-mono text-xs">RUN</code> to continue.
        </p>
        <div className="mt-4 rounded-xl border border-danger/20 bg-danger/5 p-3">
          <code className="block max-h-24 overflow-auto break-all font-mono text-xs leading-relaxed text-danger">
            {preview}
          </code>
        </div>
        <input
          autoFocus
          value={confirmText}
          onChange={(e) => setConfirmText(e.target.value)}
          placeholder="RUN"
          className="mt-4 w-full rounded-xl border border-hairline/10 bg-base/60 px-3 py-2.5 font-mono text-sm text-ink outline-none focus:border-danger/60"
        />
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setConfirmOpen(false)}>
            Cancel
          </Button>
          <Button variant="danger" disabled={confirmText !== "RUN" || mutation.isPending} onClick={() => mutation.mutate("RUN")}>
            {mutation.isPending ? "Starting..." : "Run"}
          </Button>
        </div>
      </Modal>
    </>
  );
}
