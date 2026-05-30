import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, ChevronRight, Clock, Search, ShieldCheck, SlidersHorizontal, TerminalSquare } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { flattenCommandCatalog, type CommandLeaf } from "../lib/commands";
import type { CommandNode } from "../lib/types";
import { QUERY_TIMING } from "../lib/uiConfig";
import { Badge, EmptyState, ErrorCard, Pill, Skeleton } from "../components/ui/atoms";
import { Card, CardHeader } from "../components/ui/Card";
import { CommandDashboard } from "../components/commands/CommandDashboard";
import { PageHeader } from "../components/ui/PageHeader";

export default function Commands() {
  const navigate = useNavigate();
  const splat = useParams()["*"] ?? "";
  const [search, setSearch] = useState("");

  const catalogQ = useQuery({
    queryKey: ["commands"],
    queryFn: ({ signal }) => api.commands(signal),
    staleTime: QUERY_TIMING.commandCatalogStaleMs,
    retry: false,
  });
  const leaves = useMemo(() => (catalogQ.data ? flattenCommandCatalog(catalogQ.data) : []), [catalogQ.data]);
  const executionEnabled = catalogQ.data?.execution_enabled ?? false;
  const selectedNode = leaves.find((l) => l.node.path.join("/") === splat)?.node ?? null;
  const dangerousCount = useMemo(() => leaves.filter((l) => l.node.dangerous).length, [leaves]);
  const longRunningCount = useMemo(() => leaves.filter((l) => l.node.long_running).length, [leaves]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return leaves;
    return leaves.filter(
      (l) =>
        l.group.toLowerCase().includes(q) ||
        l.node.path.join(" ").toLowerCase().includes(q) ||
        l.node.help.toLowerCase().includes(q),
    );
  }, [leaves, search]);

  const grouped = useMemo(() => {
    const map = new Map<string, CommandLeaf[]>();
    for (const l of filtered) {
      const arr = map.get(l.group) ?? [];
      arr.push(l);
      map.set(l.group, arr);
    }
    return [...map.entries()];
  }, [filtered]);

  const go = (node: CommandNode) => navigate(`/commands/${node.path.join("/")}`);

  return (
    <>
      <PageHeader
        title="Commands"
        subtitle={
          selectedNode ? (
            <span className="flex items-center gap-1 font-mono text-[12px]">
              <button onClick={() => navigate("/commands")} className="text-ink-tertiary hover:text-ink-secondary">
                commands
              </button>
              {selectedNode.path.map((p, i) => (
                <span key={i} className="flex items-center gap-1">
                  <ChevronRight size={11} className="text-ink-tertiary" />
                  <span className={i === selectedNode.path.length - 1 ? "text-ink" : "text-ink-tertiary"}>{p}</span>
                </span>
              ))}
            </span>
          ) : (
            "Each command is a live, validated dashboard"
          )
        }
        right={<Pill tone={executionEnabled ? "success" : "warn"}>{executionEnabled ? "Execution enabled" : "Execution disabled"}</Pill>}
      />

      {catalogQ.isError ? (
        <ErrorCard message={(catalogQ.error as Error)?.message ?? "Catalog unavailable"} onRetry={() => catalogQ.refetch()} />
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          {/* Palette */}
          <Card className="lg:col-span-4 xl:col-span-3" index={0}>
            <CardHeader
              title="Command library"
              icon={<TerminalSquare size={15} />}
              hint={`${filtered.length} visible of ${leaves.length}`}
            />
            <div className="mb-3 grid grid-cols-1 gap-2 sm:grid-cols-3">
              <div className="rounded-lg border border-hairline/[0.06] bg-base/25 px-2.5 py-2">
                <p className="text-[10px] uppercase text-ink-tertiary">Commands</p>
                <p className="mt-0.5 text-sm font-semibold text-ink">{leaves.length}</p>
              </div>
              <div className="rounded-lg border border-danger/15 bg-danger/5 px-2.5 py-2">
                <p className="text-[10px] uppercase text-danger/80">Confirm</p>
                <p className="mt-0.5 text-sm font-semibold text-danger">{dangerousCount}</p>
              </div>
              <div className="rounded-lg border border-warn/15 bg-warn/5 px-2.5 py-2">
                <p className="text-[10px] uppercase text-warn/80">Cancellable</p>
                <p className="mt-0.5 text-sm font-semibold text-warn">{longRunningCount}</p>
              </div>
            </div>
            <div className="relative mb-3">
              <Search size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-tertiary" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder={`Search ${leaves.length} commands…`}
                className="w-full rounded-xl border border-hairline/10 bg-base/60 py-2 pl-9 pr-3 text-sm text-ink outline-none focus:border-accent/50"
              />
            </div>
            {catalogQ.isLoading ? (
              <Skeleton className="h-72" />
            ) : (
              <div className="max-h-[60vh] space-y-4 overflow-y-auto pr-1">
                {grouped.map(([group, items]) => (
                  <div key={group}>
                    <div className="mb-1 flex items-center justify-between px-1">
                      <p className="text-[11px] font-semibold uppercase text-ink-tertiary">{group}</p>
                      <span className="text-[10.5px] text-ink-tertiary">{items.length}</span>
                    </div>
                    <ul className="space-y-0.5">
                      {items.map(({ node }) => {
                        const key = node.path.join("/");
                        const active = key === splat;
                        return (
                          <li key={key}>
                            <button
                              onClick={() => go(node)}
                              className={cn(
                                "group flex w-full items-start gap-2 rounded-lg px-2.5 py-2 text-left transition-colors",
                                active ? "bg-accent/12 shadow-[inset_3px_0_0_rgb(var(--accent))]" : "hover:bg-hairline/[0.06]",
                              )}
                            >
                              <div className="min-w-0 flex-1">
                                <code className={cn("block truncate font-mono text-[12.5px]", active ? "text-accent" : "text-ink")}>
                                  {node.path.join(" ")}
                                </code>
                                {node.help && <p className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-ink-tertiary">{node.help}</p>}
                                {(node.dangerous || node.long_running) && (
                                  <div className="mt-1.5 flex flex-wrap gap-1">
                                    {node.dangerous && (
                                      <Badge tone="danger">
                                        <AlertTriangle size={10} /> confirm
                                      </Badge>
                                    )}
                                    {node.long_running && (
                                      <Badge tone="warn">
                                        <Clock size={10} /> cancel
                                      </Badge>
                                    )}
                                  </div>
                                )}
                              </div>
                              <ChevronRight size={13} className="mt-0.5 shrink-0 text-ink-tertiary opacity-0 group-hover:opacity-100" />
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                ))}
                {grouped.length === 0 && <EmptyState label="No commands match" />}
              </div>
            )}
          </Card>

          {/* Dashboard */}
          <div className="lg:col-span-8 xl:col-span-9">
            {selectedNode ? (
              <CommandDashboard command={selectedNode} />
            ) : (
              <Card className="min-h-[24rem]" index={1}>
                <div className="mx-auto flex max-w-2xl flex-col items-center py-10 text-center">
                  <div className="grid h-12 w-12 place-items-center rounded-xl bg-accent/14 text-accent">
                    <SlidersHorizontal size={22} />
                  </div>
                  <h2 className="mt-4 text-title font-semibold text-ink">Choose a command to build a run</h2>
                  <p className="mt-2 text-sm leading-relaxed text-ink-secondary">
                    The dashboard turns each CLI command into guided inputs, parser validation, a reviewable argv, safety checks, and live job output.
                  </p>
                  <div className="mt-5 flex flex-wrap justify-center gap-2">
                    <Pill tone="accent">
                      <Search size={12} /> Search by path or purpose
                    </Pill>
                    <Pill tone="success">
                      <ShieldCheck size={12} /> Validate before run
                    </Pill>
                    <Pill tone="warn">
                      <Clock size={12} /> Cancel long jobs
                    </Pill>
                  </div>
                </div>
              </Card>
            )}
          </div>
        </div>
      )}
    </>
  );
}
