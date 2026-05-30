import { useQuery } from "@tanstack/react-query";
import { Ban, ListChecks } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../../lib/api";
import { cn } from "../../lib/cn";
import { fmtAgo } from "../../lib/format";
import { useCadence } from "../../lib/queries";
import type { JobMeta, JobStatus } from "../../lib/types";
import { QUERY_TIMING } from "../../lib/uiConfig";
import { Button, EmptyState, StatusLamp, type Tone } from "../ui/atoms";

const STATUS_TONE: Record<JobStatus, Tone> = {
  queued: "warn",
  running: "accent",
  succeeded: "success",
  failed: "danger",
  cancelled: "neutral",
};
const TERMINAL: JobStatus[] = ["succeeded", "failed", "cancelled"];

export function JobLogs({ jobId }: { jobId: string }) {
  const [lines, setLines] = useState<string[]>([]);
  const [meta, setMeta] = useState<JobMeta | null>(null);
  const cursorRef = useRef(0);
  const boxRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    setLines([]);
    setMeta(null);
    cursorRef.current = 0;
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const d = await api.job(jobId, cursorRef.current);
        if (stop) return;
        if (d.logs.length) setLines((prev) => [...prev, ...d.logs]);
        cursorRef.current = d.log_cursor;
        setMeta(d);
        if (TERMINAL.includes(d.status)) return;
      } catch {
        /* transient — retry */
      }
      if (!stop) timer = setTimeout(tick, QUERY_TIMING.clockTickMs);
    };
    timer = setTimeout(tick, QUERY_TIMING.immediateMs);
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [jobId]);

  useEffect(() => {
    boxRef.current?.scrollTo({ top: boxRef.current.scrollHeight });
  }, [lines]);

  const running = meta && !TERMINAL.includes(meta.status);

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          {meta && <StatusLamp tone={STATUS_TONE[meta.status]} pulse={!!running} size={8} />}
          <code className="font-mono text-xs text-ink-secondary">{meta?.command ?? jobId}</code>
          {meta && meta.exit_code != null && (
            <span className="text-xs text-ink-tertiary">exit {meta.exit_code}</span>
          )}
        </div>
        {running && (
          <Button variant="ghost" className="py-1" onClick={() => api.cancelJob(jobId)}>
            <Ban size={13} /> Cancel
          </Button>
        )}
      </div>
      <pre
        ref={boxRef}
        className="max-h-72 overflow-auto rounded-lg border border-hairline/10 bg-black/40 p-3 font-mono text-[11px] leading-relaxed text-ink-secondary"
      >
        {lines.length ? lines.join("\n") : running ? "Waiting for output…" : "No output."}
      </pre>
    </div>
  );
}

export function JobsPanel({ focusJobId }: { focusJobId: string | null }) {
  const cadence = useCadence();
  const jobsQ = useQuery({
    queryKey: ["jobs"],
    queryFn: ({ signal }) => api.jobs(signal),
    refetchInterval: cadence || QUERY_TIMING.jobPollMs,
  });
  const jobs = jobsQ.data?.jobs ?? [];
  const [selected, setSelected] = useState<string | null>(null);
  useEffect(() => {
    if (focusJobId) setSelected(focusJobId);
  }, [focusJobId]);
  const activeId = selected ?? jobs[0]?.id ?? null;

  return (
    <div className="grid gap-4 lg:grid-cols-5">
      <div className="lg:col-span-2">
        {jobs.length ? (
          <ul className="space-y-1.5">
            {jobs.map((j) => (
              <li key={j.id}>
                <button
                  onClick={() => setSelected(j.id)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left transition-colors",
                    j.id === activeId ? "bg-accent/12" : "hover:bg-hairline/[0.06]",
                  )}
                >
                  <StatusLamp tone={STATUS_TONE[j.status]} pulse={!TERMINAL.includes(j.status)} size={7} />
                  <code className="min-w-0 flex-1 truncate font-mono text-xs text-ink">{j.command}</code>
                  <span className="shrink-0 text-[11px] text-ink-tertiary">{fmtAgo(new Date(j.created_at * 1000).toISOString())}</span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <EmptyState icon={<ListChecks size={20} />} label="No jobs yet" />
        )}
      </div>
      <div className="lg:col-span-3">
        {activeId ? <JobLogs jobId={activeId} /> : <EmptyState label="Select a job to view logs" />}
      </div>
    </div>
  );
}
