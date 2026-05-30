import { useQuery } from "@tanstack/react-query";
import { Pause } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../../lib/api";
import { cn } from "../../lib/cn";
import { fmtAgo } from "../../lib/format";
import { CADENCE_OPTIONS, PAUSED_CADENCE, settingsStore, updateSettings } from "../../lib/settings";
import type { BrokerHealth } from "../../lib/types";
import { QUERY_TIMING } from "../../lib/uiConfig";
import { Pill, StatusLamp, type Tone } from "../ui/atoms";

function useNow(intervalMs = QUERY_TIMING.clockTickMs): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}

function brokerTone(h: BrokerHealth | null): Tone {
  if (!h) return "neutral";
  if (h.kill_switch_active) return "danger";
  if (!h.connected) return "warn";
  return "success";
}

export function TopBar({
  asOf,
  health,
  stale,
}: {
  asOf: string | null;
  health: BrokerHealth | null;
  stale: boolean;
}) {
  const cadence = settingsStore.use((s) => s.cadence);
  const now = useNow();
  const live = cadence !== PAUSED_CADENCE;

  const cfg = useQuery({
    queryKey: ["effective-config"],
    queryFn: ({ signal }) => api.effectiveConfig(signal),
    staleTime: QUERY_TIMING.relaxedRefetchMs,
    retry: false,
  });
  const deployment = cfg.data?.deployment;

  return (
    <header className="glass sticky top-0 z-20 flex h-14 items-center gap-3 border-b border-hairline/[0.07] px-5 sm:px-8">
      {/* Live / freshness */}
      <div className="flex items-center gap-2">
        <StatusLamp tone={live && !stale ? "success" : "neutral"} pulse={live && !stale} size={8} />
        <span className="text-[13px] font-medium text-ink">
          {stale ? "Reconnecting" : live ? "Live" : "Paused"}
        </span>
        <span className="hidden text-xs text-ink-tertiary sm:inline">· {fmtAgo(asOf, now)}</span>
      </div>

      <div className="ml-auto flex items-center gap-2.5 sm:gap-4">
        {/* Broker status */}
        <div className="hidden items-center gap-2 sm:flex">
          <StatusLamp tone={brokerTone(health)} size={8} />
          <span className="text-[13px] text-ink-secondary">
            {health?.kill_switch_active
              ? "Halted"
              : health?.connected
                ? "Broker live"
                : "Broker down"}
          </span>
        </div>

        {/* Mode pill — only once the backend reports the real deployment mode */}
        {deployment ? (
          <Pill tone={deployment.paper_trading ? "accent" : "danger"}>
            {deployment.paper_trading ? "Paper" : "Live"}
            {deployment.profile_preset && (
              <span className="opacity-60">· {deployment.profile_preset}</span>
            )}
          </Pill>
        ) : (
          <Pill tone="neutral">…</Pill>
        )}

        {/* Cadence control */}
        <div className="flex items-center gap-0.5 rounded-xl bg-hairline/[0.08] p-0.5">
          {CADENCE_OPTIONS.map((c) => (
            <button
              key={c.value}
              onClick={() => updateSettings({ cadence: c.value })}
              aria-label={c.ariaLabel}
              className={cn(
                "min-w-[30px] rounded-[9px] px-2 py-1 text-xs font-medium tabular-nums transition-colors",
                cadence === c.value
                  ? "bg-surface text-ink shadow-sm"
                  : "text-ink-tertiary hover:text-ink-secondary",
              )}
            >
              {c.value === PAUSED_CADENCE ? <Pause size={12} className="mx-auto" /> : c.shortLabel}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
