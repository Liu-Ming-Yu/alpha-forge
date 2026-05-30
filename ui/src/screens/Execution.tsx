import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  AlertOctagon,
  ArrowLeftRight,
  Ban,
  Receipt,
  ShieldCheck,
  Wallet,
} from "lucide-react";
import { useState } from "react";
import { api, ApiError } from "../lib/api";
import { useCapabilities, useDashboard, useLive } from "../lib/queries";
import { fmtAgo, fmtInt, fmtMoney, fmtNum, fmtPct, shortId } from "../lib/format";
import { listFrom, pick } from "../lib/objects";
import { isErr } from "../lib/types";
import { DISPLAY_LIMITS, REQUEST_LIMITS } from "../lib/uiConfig";
import { Badge, Button, EmptyState, ErrorCard, KeyValue, Pill, Skeleton, StatusLamp } from "../components/ui/atoms";
import { BrokerSyncPanel } from "../components/broker/BrokerSyncPanel";
import { Card, CardHeader } from "../components/ui/Card";
import { Modal } from "../components/ui/Dialog";
import { Table } from "../components/ui/Table";
import { PageHeader } from "../components/ui/PageHeader";

const CONFIRM_PHRASE = "CLEAR KILL SWITCH";

function sideTone(side: string) {
  return side?.toUpperCase() === "BUY" ? "success" : side?.toUpperCase() === "SELL" ? "danger" : "neutral";
}

function KillSwitchControl({ active, reason }: { active: boolean; reason: string }) {
  const caps = useCapabilities();
  const qc = useQueryClient();
  const allowed = caps.data?.write_controls?.kill_switch_clear ?? false;
  const [open, setOpen] = useState(false);
  const [phrase, setPhrase] = useState("");
  const [why, setWhy] = useState("");

  const mutation = useMutation({
    mutationFn: () => api.clearKillSwitch(why.trim() || "operator-cleared via console", CONFIRM_PHRASE),
    onSuccess: () => {
      setOpen(false);
      setPhrase("");
      setWhy("");
      qc.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });

  return (
    <Card index={0} className={active ? "border-danger/30 bg-danger/[0.04]" : undefined}>
      <CardHeader
        title="Kill switch"
        icon={active ? <AlertOctagon size={15} className="text-danger" /> : <ShieldCheck size={15} />}
      />
      <div className="flex items-center gap-3">
        <StatusLamp tone={active ? "danger" : "success"} pulse={active} />
        <div className="min-w-0 flex-1">
          <p className="text-[15px] font-semibold text-ink">
            {active ? "Active — order submission halted" : "Clear — trading armed"}
          </p>
          {active && reason && <p className="mt-0.5 truncate text-[13px] text-ink-tertiary">{reason}</p>}
        </div>
        {active &&
          (allowed ? (
            <Button variant="danger" onClick={() => setOpen(true)}>
              Clear…
            </Button>
          ) : (
            <Pill tone="neutral">Clear not enabled</Pill>
          ))}
      </div>

      <Modal open={open} onClose={() => setOpen(false)} labelledBy="ks-title">
        <h3 id="ks-title" className="text-title font-semibold text-ink">
          Clear kill switch?
        </h3>
        <p className="mt-2 text-sm text-ink-secondary">
          This re-arms order submission. Type{" "}
          <code className="rounded bg-hairline/15 px-1.5 py-0.5 font-mono text-xs text-ink">{CONFIRM_PHRASE}</code>{" "}
          to confirm.
        </p>
        <div className="mt-4 space-y-3">
          <input
            value={why}
            onChange={(e) => setWhy(e.target.value)}
            placeholder="Reason (audit trail)"
            className="w-full rounded-xl border border-hairline/10 bg-base/60 px-3 py-2.5 text-sm text-ink outline-none focus:border-accent/50"
          />
          <input
            value={phrase}
            onChange={(e) => setPhrase(e.target.value)}
            placeholder={CONFIRM_PHRASE}
            className="w-full rounded-xl border border-hairline/10 bg-base/60 px-3 py-2.5 font-mono text-sm text-ink outline-none focus:border-danger/60"
          />
          {mutation.isError && (
            <p className="text-xs text-danger">
              {mutation.error instanceof ApiError ? mutation.error.detail : "Failed to clear"}
            </p>
          )}
        </div>
        <div className="mt-6 flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button
            variant="danger"
            disabled={phrase !== CONFIRM_PHRASE || mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            {mutation.isPending ? "Clearing…" : "Clear kill switch"}
          </Button>
        </div>
      </Modal>
    </Card>
  );
}

export default function Execution() {
  const dash = useDashboard();
  const d = dash.data;
  const ledgerQ = useLive(["cash-ledger"], api.cashLedger);
  const unmatchedQ = useLive(["unmatched-fills"], () => api.unmatchedFills(REQUEST_LIMITS.unmatchedFills));

  if (!d && dash.isLoading) {
    return (
      <>
        <PageHeader title="Execution" subtitle="Broker, orders & controls" />
        <div className="grid gap-4 lg:grid-cols-3">
          {Array.from({ length: DISPLAY_LIMITS.executionLoadingCards }).map((_, i) => (
            <Skeleton key={i} className="h-40" />
          ))}
        </div>
      </>
    );
  }
  if (!d) {
    return (
      <>
        <PageHeader title="Execution" />
        <ErrorCard message={(dash.error as Error)?.message ?? "No data"} onRetry={() => dash.refetch()} />
      </>
    );
  }

  const health = isErr(d.health) ? null : d.health;
  const metrics = d.selected_run?.metrics && !isErr(d.selected_run.metrics) ? d.selected_run.metrics : null;
  const blotter = d.selected_run?.blotter && !isErr(d.selected_run.blotter) ? d.selected_run.blotter : null;
  const entries = blotter?.entries ?? [];
  const violations = listFrom<Record<string, unknown>>(d.compliance, "violations");
  const unmatched = listFrom<Record<string, unknown>>(unmatchedQ.data, "unmatched_fills");
  const ledger = ledgerQ.data;
  const killActive = (health?.kill_switch_active ?? false) || (d.kill_switch?.active ?? false);
  const killReason = health?.kill_switch_reason || d.kill_switch?.reason || "";

  return (
    <>
      <PageHeader title="Execution" subtitle="Broker, orders & controls" />

      <div className="mb-4">
        <BrokerSyncPanel />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <KillSwitchControl active={killActive} reason={killReason} />

        {/* Paper-gate metrics */}
        <Card index={1}>
          <CardHeader title="Order quality" icon={<ArrowLeftRight size={15} />} hint="Paper-gate metrics" />
          {metrics ? (
            <div>
              <KeyValue k="Orders considered">{fmtInt(metrics.orders_considered)}</KeyValue>
              <KeyValue k="Reject rate">{fmtPct(metrics.reject_rate)}</KeyValue>
              <KeyValue k="Broker error rate">{fmtPct(metrics.broker_error_rate)}</KeyValue>
              <KeyValue k="Avg slippage">
                {metrics.average_fill_slippage_bps != null ? `${fmtNum(metrics.average_fill_slippage_bps, 1)} bps` : "—"}
              </KeyValue>
              <KeyValue k="Reconcile gaps">{fmtInt(metrics.reconcile_discrepancies)}</KeyValue>
            </div>
          ) : (
            <EmptyState label="No metrics for selected run" />
          )}
        </Card>

        {/* Cash ledger */}
        <Card index={2}>
          <CardHeader title="Cash ledger" icon={<Wallet size={15} />} />
          {ledger ? (
            <div>
              <KeyValue k="Settled">{fmtMoney(pick(ledger, "settled_cash"))}</KeyValue>
              <KeyValue k="Unsettled">{fmtMoney(pick(ledger, "unsettled_cash"))}</KeyValue>
              <KeyValue k="Reserved">{fmtMoney(pick(ledger, "reserved_cash"))}</KeyValue>
              <KeyValue k="Available">{fmtMoney(pick(ledger, "available_cash"))}</KeyValue>
              <KeyValue k="Pending lots">{fmtInt(pick(ledger, "pending_lots_count"))}</KeyValue>
            </div>
          ) : ledgerQ.isError ? (
            <EmptyState label="Ledger unavailable" />
          ) : (
            <Skeleton className="h-40" />
          )}
        </Card>
      </div>

      {/* Blotter */}
      <div className="mt-4">
        <Card index={3}>
          <CardHeader
            title="Order blotter"
            icon={<Receipt size={15} />}
            hint={blotter ? `as of ${fmtAgo(blotter.as_of)}` : undefined}
            right={<Badge tone="neutral">{fmtInt(entries.length)} orders</Badge>}
          />
          <Table
            columns={[
              { key: "id", header: "Order", cell: (r) => <span className="font-mono text-xs text-ink-secondary">{shortId(r.order_id)}</span> },
              { key: "side", header: "Side", cell: (r) => <Badge tone={sideTone(r.side)}>{r.side}</Badge> },
              { key: "qty", header: "Qty", align: "right", cell: (r) => <span className="tnum">{fmtInt(r.quantity)}</span> },
              { key: "type", header: "Type", cell: (r) => <span className="text-ink-tertiary">{r.order_type}</span> },
              {
                key: "fill",
                header: "Filled",
                align: "right",
                cell: (r) => (
                  <span className="tnum">
                    {fmtInt(r.total_filled)}/{fmtInt(r.quantity)}
                  </span>
                ),
              },
              { key: "px", header: "Avg px", align: "right", cell: (r) => <span className="tnum">{r.avg_fill_price != null ? fmtMoney(r.avg_fill_price) : "—"}</span> },
              {
                key: "status",
                header: "Status",
                align: "right",
                cell: (r) => <span className="text-ink-tertiary">{r.broker_status ?? "—"}</span>,
              },
            ]}
            rows={entries}
            rowKey={(r) => r.order_id}
            empty={<EmptyState icon={<Receipt size={20} />} label="No orders for the selected run" />}
          />
        </Card>
      </div>

      {/* Compliance + unmatched */}
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card index={4}>
          <CardHeader title="Compliance violations" icon={<Ban size={15} />} right={<Badge tone={violations.length ? "warn" : "neutral"}>{fmtInt(violations.length)}</Badge>} />
          {violations.length > 0 ? (
            <ul className="divide-y divide-hairline/[0.05]">
              {violations.slice(0, DISPLAY_LIMITS.executionComplianceRows).map((v, i) => (
                <li key={i} className="flex items-center gap-3 py-2.5">
                  <Badge tone="warn">{String(pick(v, "rule") ?? "rule")}</Badge>
                  <span className="truncate text-[13px] text-ink-secondary">{String(pick(v, "detail") ?? "")}</span>
                  <span className="ml-auto shrink-0 text-xs text-ink-tertiary">{fmtAgo(String(pick(v, "occurred_at") ?? ""))}</span>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState icon={<ShieldCheck size={20} />} label="No violations in window" />
          )}
        </Card>

        <Card index={5}>
          <CardHeader title="Unmatched fills" icon={<AlertOctagon size={15} />} right={<Badge tone={unmatched.length ? "warn" : "neutral"}>{fmtInt(unmatched.length)}</Badge>} />
          {unmatched.length > 0 ? (
            <Table
              dense
              columns={[
                { key: "ib", header: "IB order", cell: (r) => <span className="font-mono text-xs">{String(pick(r, "ib_order_id") ?? "—")}</span> },
                { key: "exec", header: "Exec", cell: (r) => <span className="font-mono text-xs text-ink-tertiary">{shortId(String(pick(r, "exec_id") ?? ""), DISPLAY_LIMITS.execIdChars)}</span> },
                { key: "when", header: "When", align: "right", cell: (r) => fmtAgo(String(pick(r, "occurred_at") ?? "")) },
              ]}
              rows={unmatched}
              rowKey={(_, i) => String(i)}
            />
          ) : (
            <EmptyState icon={<ShieldCheck size={20} />} label="All fills reconciled" />
          )}
        </Card>
      </div>
    </>
  );
}
