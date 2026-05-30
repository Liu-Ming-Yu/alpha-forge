import { useQuery } from "@tanstack/react-query";
import { Beaker, ClipboardCheck, FlaskConical, Microscope, Rocket } from "lucide-react";
import { useMemo } from "react";
import { api } from "../lib/api";
import { useRelaxedCadence } from "../lib/queries";
import { fmtAgo, titleCase } from "../lib/format";
import { isObj, listFrom, pick, scalarEntries } from "../lib/objects";
import type { EffectiveConfig } from "../lib/types";
import { DISPLAY_LIMITS, intervalLabel, QUERY_TIMING, REQUEST_LIMITS } from "../lib/uiConfig";
import { Badge, CheckDot, EmptyState, KeyValue, Pill, Skeleton, type Tone } from "../components/ui/atoms";
import { Card, CardHeader } from "../components/ui/Card";
import { JsonPeek } from "../components/ui/JsonPeek";
import { PageHeader } from "../components/ui/PageHeader";

// Research evidence is slow-moving; use the shared relaxed cadence so the
// global live toggle still pauses these reads.
function useSnapshot<T>(
  key: readonly unknown[],
  fn: (signal?: AbortSignal) => Promise<T>,
  enabled = true,
) {
  const refetchInterval = useRelaxedCadence();
  return useQuery({
    queryKey: key,
    queryFn: ({ signal }) => fn(signal),
    enabled,
    staleTime: QUERY_TIMING.relaxedStaleMs,
    refetchInterval,
    retry: false,
  });
}

function activeProfile(config: EffectiveConfig | undefined): string | null {
  if (!config) return null;
  const profiles = config.enums.profiles;
  if (!profiles.length) return null;
  const desired = config.deployment.paper_trading ? "paper" : "live";
  return profiles.find((profile) => profile.toLowerCase() === desired)
    ?? profiles.find((profile) => profile.toLowerCase().includes(desired))
    ?? profiles[0];
}

function firstList(o: unknown, keys: string[]): Record<string, unknown>[] {
  for (const k of keys) {
    const l = listFrom<Record<string, unknown>>(o, k);
    if (l.length) return l;
  }
  return Array.isArray(o) ? (o as Record<string, unknown>[]) : [];
}

export default function Research() {
  const cadence = useRelaxedCadence();
  const cfg = useQuery({
    queryKey: ["effective-config"],
    queryFn: ({ signal }) => api.effectiveConfig(signal),
    staleTime: QUERY_TIMING.relaxedRefetchMs,
    retry: false,
  });
  const profile = useMemo(() => activeProfile(cfg.data), [cfg.data]);
  const campaigns = useSnapshot(["research-campaigns"], (signal) => api.researchCampaigns(REQUEST_LIMITS.researchCampaigns, signal));
  const audits = useSnapshot(["feature-audits"], (signal) => api.featureAudits(REQUEST_LIMITS.featureAudits, signal));
  const soak = useSnapshot(["paper-soak"], api.paperSoak);
  const readiness = useSnapshot(["readiness", profile], (signal) => api.readiness(profile!, signal), !!profile);
  const promotion = useSnapshot(["promotion", profile], (signal) => api.promotionCandidate(profile!, signal), !!profile);

  const campaignRows = firstList(campaigns.data, ["campaigns", "runs", "items", "results"]);
  const auditRows = firstList(audits.data, ["audits", "feature_audits", "entries", "items"]);
  const soakSections = isObj(soak.data) ? (soak.data.passed_sections as Record<string, boolean> | undefined) : undefined;
  const promoEligible = pick<boolean>(promotion.data, "eligible", "passed", "promote", "ready");

  return (
    <>
      <PageHeader
        title="Research"
        subtitle="Evidence, gates & promotion"
        right={<span className="text-xs text-ink-tertiary">{cadence ? `Auto-refresh ${intervalLabel(cadence)}` : "Paused"}</span>}
      />

      {/* Promotion + readiness + paper-soak */}
      <div className="grid gap-4 lg:grid-cols-3">
        <Card index={0}>
          <CardHeader title="Promotion candidate" icon={<Rocket size={15} />} hint={profile ? `profile: ${profile}` : undefined} />
          {promotion.isLoading || cfg.isLoading ? (
            <Skeleton className="h-32" />
          ) : isObj(promotion.data) && !("error" in promotion.data) ? (
            <div>
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[13px] text-ink-secondary">Verdict</span>
                {promoEligible !== undefined ? (
                  <Pill tone={promoEligible ? "success" : "warn"}>{promoEligible ? "Eligible" : "Not yet"}</Pill>
                ) : (
                  <Badge tone="neutral">unknown</Badge>
                )}
              </div>
              {scalarEntries(promotion.data)
                .filter(([k]) => !["error"].includes(k))
                .slice(0, DISPLAY_LIMITS.researchPromotionFields)
                .map(([k, v]) => (
                  <KeyValue key={k} k={titleCase(k)}>
                    {String(v)}
                  </KeyValue>
                ))}
              <JsonPeek data={promotion.data} label="Full evidence" />
            </div>
          ) : (
            <EmptyState label="No promotion candidate" />
          )}
        </Card>

        <Card index={1}>
          <CardHeader title="Readiness snapshot" icon={<ClipboardCheck size={15} />} />
          {readiness.isLoading || cfg.isLoading ? (
            <Skeleton className="h-32" />
          ) : isObj(readiness.data) && readiness.data.snapshot ? (
            <div>
              {scalarEntries(readiness.data.snapshot).slice(0, DISPLAY_LIMITS.researchReadinessFields).map(([k, v]) => (
                <KeyValue key={k} k={titleCase(k)}>
                  {typeof v === "boolean" ? <CheckDot ok={v} /> : String(v)}
                </KeyValue>
              ))}
              <JsonPeek data={readiness.data.snapshot} label="Full snapshot" />
            </div>
          ) : (
            <EmptyState label="No readiness snapshot" />
          )}
        </Card>

        <Card index={2}>
          <CardHeader title="Paper-soak" icon={<Beaker size={15} />} hint={isObj(soak.data) ? String(pick(soak.data, "version") ?? "") : undefined} />
          {soakSections && Object.keys(soakSections).length > 0 ? (
            <div className="space-y-1.5">
              {Object.entries(soakSections).map(([k, ok]) => (
                <div key={k} className="flex items-center justify-between">
                  <span className="text-[13px] text-ink-secondary">{titleCase(k)}</span>
                  <CheckDot ok={!!ok} />
                </div>
              ))}
              <p className="mt-2 text-xs text-ink-tertiary">
                Generated {fmtAgo(isObj(soak.data) ? String(pick(soak.data, "generated_at") ?? "") : "")}
              </p>
            </div>
          ) : (
            <EmptyState label="No paper-soak report" />
          )}
        </Card>
      </div>

      {/* Campaigns */}
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card index={3}>
          <CardHeader title="Research campaigns" icon={<FlaskConical size={15} />} right={<Badge tone="neutral">{campaignRows.length}</Badge>} />
          {campaigns.isLoading ? (
            <Skeleton className="h-40" />
          ) : campaignRows.length > 0 ? (
            <ul className="space-y-2.5">
              {campaignRows.slice(0, DISPLAY_LIMITS.researchCampaignRows).map((row, i) => {
                const title = String(pick(row, "name", "campaign", "title", "run_id", "id") ?? `campaign ${i}`);
                const status = pick<string>(row, "status", "state", "decision");
                const when = String(pick(row, "created_at", "started_at", "as_of", "timestamp") ?? "");
                const tone: Tone = status === "passed" || status === "promoted" ? "success" : status === "failed" ? "danger" : "neutral";
                return (
                  <li key={i} className="rounded-xl border border-hairline/[0.06] bg-base/40 p-3">
                    <div className="flex items-center gap-2">
                      <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">{title}</span>
                      {status && <Badge tone={tone}>{status}</Badge>}
                      {when && <span className="shrink-0 text-xs text-ink-tertiary">{fmtAgo(when)}</span>}
                    </div>
                    <JsonPeek data={row} label="Details" />
                  </li>
                );
              })}
            </ul>
          ) : (
            <EmptyState icon={<Microscope size={20} />} label="No research campaigns recorded" />
          )}
        </Card>

        <Card index={4}>
          <CardHeader title="Feature audits" icon={<Microscope size={15} />} right={<Badge tone="neutral">{auditRows.length}</Badge>} />
          {audits.isLoading ? (
            <Skeleton className="h-40" />
          ) : auditRows.length > 0 ? (
            <ul className="space-y-2.5">
              {auditRows.slice(0, DISPLAY_LIMITS.researchAuditRows).map((row, i) => {
                const name = String(pick(row, "feature_name", "feature", "name", "id") ?? `feature ${i}`);
                const decision = pick<string>(row, "decision", "status", "mode", "verdict");
                const when = String(pick(row, "created_at", "as_of", "timestamp") ?? "");
                return (
                  <li key={i} className="rounded-xl border border-hairline/[0.06] bg-base/40 p-3">
                    <div className="flex items-center gap-2">
                      <span className="min-w-0 flex-1 truncate font-mono text-xs text-ink-secondary">{name}</span>
                      {decision && <Badge tone="neutral">{decision}</Badge>}
                      {when && <span className="shrink-0 text-xs text-ink-tertiary">{fmtAgo(when)}</span>}
                    </div>
                    <JsonPeek data={row} label="Details" />
                  </li>
                );
              })}
            </ul>
          ) : (
            <EmptyState icon={<Microscope size={20} />} label="No feature audits recorded" />
          )}
        </Card>
      </div>
    </>
  );
}
