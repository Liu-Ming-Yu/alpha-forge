import { ChevronRight } from "lucide-react";
import { useState } from "react";
import { cn } from "../../lib/cn";

/** Collapsible raw-JSON viewer — a graceful fallback for free-form payloads. */
export function JsonPeek({ data, label = "Raw payload" }: { data: unknown; label?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-1">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 text-xs text-ink-tertiary transition-colors hover:text-ink-secondary"
      >
        <ChevronRight size={13} className={cn("transition-transform", open && "rotate-90")} />
        {label}
      </button>
      {open && (
        <pre className="mt-2 max-h-72 overflow-auto rounded-lg border border-hairline/10 bg-base/60 p-3 font-mono text-[11px] leading-relaxed text-ink-secondary">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}
