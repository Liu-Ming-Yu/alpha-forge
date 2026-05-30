import { useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Activity, ChevronDown, Eye, EyeOff, KeyRound } from "lucide-react";
import { useState } from "react";
import { api, ApiError } from "../lib/api";
import { settingsStore, updateSettings } from "../lib/settings";
import { Button } from "./ui/atoms";

export function Connect({ error, loading }: { error: unknown; loading: boolean }) {
  const qc = useQueryClient();
  const current = settingsStore.get();
  const [apiKey, setApiKey] = useState(current.apiKey);
  const [apiBase, setApiBase] = useState(current.apiBase);
  const [showKey, setShowKey] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(!!current.apiBase);

  const info = useQuery({
    queryKey: ["console-info"],
    queryFn: ({ signal }) => api.consoleInfo(signal),
    retry: false,
    staleTime: Infinity,
  });

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    updateSettings({ apiKey: apiKey.trim(), apiBase: apiBase.trim() });
    qc.invalidateQueries({ queryKey: ["capabilities"] });
  };

  const reachable = info.isSuccess || info.isLoading;
  let hint: string | null = null;
  if (error instanceof ApiError) {
    if (error.status === 401) hint = "That key was rejected. Check QP__API__OPERATOR_API_KEY.";
    else if (error.status === 0)
      hint = `Couldn't reach the API${apiBase ? ` at ${apiBase}` : ""}. Is serve-api running?`;
    else hint = error.detail;
  }

  return (
    <div className="relative grid min-h-screen place-items-center overflow-hidden bg-base p-6">
      {/* ambient depth */}
      <div
        className="pointer-events-none absolute -top-40 left-1/2 h-[36rem] w-[36rem] -translate-x-1/2 rounded-full opacity-25 blur-3xl"
        style={{ background: "radial-gradient(closest-side, rgb(var(--ambient-1)), transparent)" }}
      />
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="card relative z-10 w-full max-w-[26rem] p-8 shadow-float"
      >
        <div className="mb-6 flex flex-col items-center text-center">
          <div className="mb-4 grid h-12 w-12 place-items-center rounded-2xl bg-accent/15 text-accent">
            <Activity size={22} />
          </div>
          <h1 className="text-title font-semibold tracking-tight text-ink">
            {info.data?.product ?? "Quant Operator Console"}
          </h1>
          <p className="mt-1 text-sm text-ink-tertiary">
            {reachable
              ? `Connect to your operator API${info.data ? ` · v${info.data.api_version}` : ""}`
              : "Operator API not reachable"}
          </p>
        </div>

        <form onSubmit={submit} className="space-y-4">
          <label className="block">
            <span className="mb-1.5 block text-[13px] font-medium text-ink-secondary">
              Operator API key
            </span>
            <div className="relative">
              <KeyRound
                size={15}
                className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-tertiary"
              />
              <input
                autoFocus
                type={showKey ? "text" : "password"}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="X-API-Key"
                className="w-full rounded-xl border border-hairline/10 bg-base/60 py-2.5 pl-9 pr-10 text-sm text-ink outline-none transition focus:border-accent/50"
              />
              <button
                type="button"
                onClick={() => setShowKey((s) => !s)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-ink-tertiary hover:text-ink-secondary"
                aria-label={showKey ? "Hide key" : "Show key"}
              >
                {showKey ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </div>
          </label>

          <button
            type="button"
            onClick={() => setShowAdvanced((s) => !s)}
            className="flex items-center gap-1 text-xs text-ink-tertiary hover:text-ink-secondary"
          >
            <ChevronDown
              size={13}
              className={`transition-transform ${showAdvanced ? "rotate-180" : ""}`}
            />
            Advanced
          </button>
          {showAdvanced && (
            <label className="block">
              <span className="mb-1.5 block text-[13px] font-medium text-ink-secondary">
                API base URL
              </span>
              <input
                type="text"
                value={apiBase}
                onChange={(e) => setApiBase(e.target.value)}
                placeholder="(same origin)"
                className="w-full rounded-xl border border-hairline/10 bg-base/60 px-3 py-2.5 font-mono text-xs text-ink outline-none transition focus:border-accent/50"
              />
              <span className="mt-1 block text-xs text-ink-tertiary">
                Leave blank when the console is served by the API.
              </span>
            </label>
          )}

          {hint && (
            <p className="rounded-lg border border-danger/20 bg-danger/5 px-3 py-2 text-xs text-danger">
              {hint}
            </p>
          )}

          <Button type="submit" variant="primary" className="w-full" disabled={loading}>
            {loading ? "Connecting…" : "Connect"}
          </Button>
        </form>
      </motion.div>
    </div>
  );
}
