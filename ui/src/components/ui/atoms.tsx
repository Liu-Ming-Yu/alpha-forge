import { motion } from "framer-motion";
import { AlertTriangle, Check, Info } from "lucide-react";
import {
  type ButtonHTMLAttributes,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";
import { cn } from "../../lib/cn";
import { REPLAY_TIMING } from "../../lib/uiConfig";

export type Tone = "neutral" | "accent" | "success" | "warn" | "danger";

const TONE_BG: Record<Tone, string> = {
  neutral: "bg-hairline/10 text-ink-secondary",
  accent: "bg-accent/15 text-accent",
  success: "bg-success/15 text-success",
  warn: "bg-warn/15 text-warn",
  danger: "bg-danger/15 text-danger",
};
const TONE_DOT: Record<Tone, string> = {
  neutral: "bg-ink-tertiary",
  accent: "bg-accent",
  success: "bg-success",
  warn: "bg-warn",
  danger: "bg-danger",
};

/** Soft-glow status indicator. Conveys state by icon-free dot + adjacent text. */
export function StatusLamp({
  tone,
  pulse,
  size = 9,
}: {
  tone: Tone;
  pulse?: boolean;
  size?: number;
}) {
  return (
    <span className="relative inline-flex" style={{ width: size, height: size }}>
      {pulse && (
        <span
          className={cn("absolute inset-0 rounded-full opacity-60 animate-ping", TONE_DOT[tone])}
        />
      )}
      <span
        className={cn("relative rounded-full", TONE_DOT[tone])}
        style={{ width: size, height: size, boxShadow: "0 0 10px 0 currentColor" }}
      />
    </span>
  );
}

export function Pill({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
        TONE_BG[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-semibold tracking-tight",
        TONE_BG[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

interface BtnProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  children: ReactNode;
}

export function Button({ variant = "secondary", className, children, ...rest }: BtnProps) {
  const styles: Record<string, string> = {
    primary: "bg-accent text-white hover:brightness-110 shadow-sm",
    secondary: "bg-hairline/10 text-ink hover:bg-hairline/[0.16]",
    ghost: "text-ink-secondary hover:bg-hairline/10",
    danger: "bg-danger text-white hover:brightness-110",
  };
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium",
        "transition-all active:scale-[0.98] disabled:opacity-40 disabled:pointer-events-none",
        styles[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

export interface SegOption<T extends string> {
  value: T;
  label: ReactNode;
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  size = "md",
}: {
  options: SegOption<T>[];
  value: T;
  onChange: (v: T) => void;
  size?: "sm" | "md";
}) {
  return (
    <div
      role="tablist"
      className={cn(
        "inline-flex items-center gap-0.5 rounded-xl bg-hairline/[0.08] p-0.5",
        size === "sm" ? "text-xs" : "text-[13px]",
      )}
    >
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(o.value)}
            className={cn(
              "relative rounded-[10px] px-3 py-1.5 font-medium transition-colors",
              active ? "text-ink" : "text-ink-tertiary hover:text-ink-secondary",
            )}
          >
            {active && (
              <motion.span
                layoutId="seg-active"
                className="absolute inset-0 rounded-[10px] bg-surface shadow-sm"
                transition={{ type: "spring", stiffness: 400, damping: 32 }}
              />
            )}
            <span className="relative z-10 whitespace-nowrap">{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label?: string;
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative h-[26px] w-[44px] rounded-full transition-colors",
        checked ? "bg-success" : "bg-hairline/20",
      )}
    >
      <motion.span
        layout
        transition={{ type: "spring", stiffness: 500, damping: 35 }}
        className="absolute top-[3px] h-5 w-5 rounded-full bg-white shadow"
        style={{ left: checked ? 21 : 3 }}
      />
    </button>
  );
}

/** rAF number tween — smooth metric transitions without re-render storms. */
export function AnimatedNumber({
  value,
  format,
  className,
}: {
  value: number | null;
  format: (v: number) => string;
  className?: string;
}) {
  const [display, setDisplay] = useState<number | null>(value);
  const fromRef = useRef<number | null>(value);
  const rafRef = useRef<number>();

  useEffect(() => {
    if (value === null) {
      setDisplay(null);
      return;
    }
    const from = fromRef.current ?? value;
    const to = value;
    if (from === to) {
      setDisplay(to);
      return;
    }
    const start = performance.now();
    const dur = REPLAY_TIMING.shimmerMs;
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay(from + (to - from) * eased);
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
      else fromRef.current = to;
    };
    cancelAnimationFrame(rafRef.current ?? 0);
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current ?? 0);
  }, [value]);

  return <span className={cn("tnum", className)}>{display === null ? "—" : format(display)}</span>;
}

export function Skeleton({ className }: { className?: string }) {
  return (
    <div className={cn("relative overflow-hidden rounded-md bg-hairline/10", className)}>
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-hairline/10 to-transparent" />
    </div>
  );
}

export function ErrorCard({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-danger/20 bg-danger/5 p-4 text-sm">
      <AlertTriangle size={16} className="mt-0.5 shrink-0 text-danger" />
      <div className="min-w-0 flex-1">
        <p className="font-medium text-ink">Couldn't load</p>
        <p className="mt-0.5 break-words text-ink-tertiary">{message}</p>
      </div>
      {onRetry && (
        <Button variant="ghost" onClick={onRetry} className="shrink-0 py-1">
          Retry
        </Button>
      )}
    </div>
  );
}

export function EmptyState({ icon, label }: { icon?: ReactNode; label: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-10 text-center">
      <span className="text-ink-tertiary opacity-60">{icon ?? <Info size={20} />}</span>
      <p className="text-sm text-ink-tertiary">{label}</p>
    </div>
  );
}

export function KeyValue({
  k,
  children,
  mono,
}: {
  k: string;
  children: ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <span className="text-[13px] text-ink-tertiary">{k}</span>
      <span className={cn("text-[13px] text-ink text-right tnum", mono && "font-mono text-xs")}>
        {children}
      </span>
    </div>
  );
}

export function CheckDot({ ok }: { ok: boolean }) {
  return ok ? (
    <Check size={14} className="text-success" />
  ) : (
    <AlertTriangle size={14} className="text-warn" />
  );
}
