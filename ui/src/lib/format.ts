/** Formatting helpers — consistent, locale-aware, null-safe. */

import { DISPLAY_LIMITS } from "./uiConfig";

const num = (v: unknown): number | null => {
  if (v === null || v === undefined || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
};

export function fmtMoney(v: unknown, opts: { compact?: boolean } = {}): string {
  const n = num(v);
  if (n === null) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    notation: opts.compact ? "compact" : "standard",
    maximumFractionDigits: opts.compact ? 1 : 2,
    minimumFractionDigits: opts.compact ? 0 : 2,
  }).format(n);
}

export function fmtNum(v: unknown, digits = 2): string {
  const n = num(v);
  if (n === null) return "—";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(n);
}

export function fmtInt(v: unknown): string {
  const n = num(v);
  if (n === null) return "—";
  return new Intl.NumberFormat("en-US").format(Math.round(n));
}

/** Fraction (0.0123) → "1.23%". Pass alreadyPct=true if value is already %. */
export function fmtPct(v: unknown, digits = 2, alreadyPct = false): string {
  const n = num(v);
  if (n === null) return "—";
  const pct = alreadyPct ? n : n * 100;
  return `${pct >= 0 ? "" : ""}${pct.toFixed(digits)}%`;
}

export function fmtSignedPct(v: unknown, digits = 2, alreadyPct = false): string {
  const n = num(v);
  if (n === null) return "—";
  const pct = alreadyPct ? n : n * 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(digits)}%`;
}

export function fmtBps(v: unknown): string {
  const n = num(v);
  if (n === null) return "—";
  return `${n.toFixed(1)} bps`;
}

/** Compact "3s ago", "5m ago", "2h ago" from an ISO timestamp. */
export function fmtAgo(iso: string | null | undefined, nowMs = Date.now()): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const s = Math.max(0, Math.round((nowMs - t) / 1000));
  if (s < 2) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

export function fmtClock(iso: string | null | undefined): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  return new Date(t).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

/** Truncate a uuid/string for display. */
export function shortId(v: string | null | undefined, head: number = DISPLAY_LIMITS.shortIdChars): string {
  if (!v) return "—";
  return v.length <= head ? v : `${v.slice(0, head)}…`;
}

export function titleCase(s: string): string {
  return s
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Bytes → "1.4 GB" / "512 MB". */
export function fmtBytes(v: unknown): string {
  const n = num(v);
  if (n === null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let val = n;
  let i = 0;
  while (val >= 1024 && i < units.length - 1) {
    val /= 1024;
    i += 1;
  }
  return `${val.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

/** Seconds → "3d 4h" / "2h 15m" / "45s" (compact duration). */
export function fmtDuration(seconds: unknown): string {
  const n = num(seconds);
  if (n === null || n < 0) return "—";
  const d = Math.floor(n / 86400);
  const h = Math.floor((n % 86400) / 3600);
  const m = Math.floor((n % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m`;
  return `${Math.floor(n)}s`;
}

/** Seconds-since-epoch of a start time → uptime string. */
export function fmtUptimeSince(epochSeconds: unknown, nowMs = Date.now()): string {
  const n = num(epochSeconds);
  if (n === null) return "—";
  return fmtDuration(nowMs / 1000 - n);
}
