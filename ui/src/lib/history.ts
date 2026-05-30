import { useSyncExternalStore } from "react";

/**
 * In-memory rolling history for live charts. The operator API is point-in-time
 * (no metric time-series), so we accumulate each poll's scalars into bounded
 * ring buffers and render those as streaming sparklines/area charts. History is
 * session-local by design — a *live* monitor, not a historical store.
 */
export interface Point {
  t: number;
  v: number;
}

const CAPACITY = 240;
const series = new Map<string, Point[]>();
const subs = new Set<() => void>();
const EMPTY: Point[] = [];

function emit(): void {
  subs.forEach((s) => s());
}

function append(key: string, v: number, t: number): void {
  const prev = series.get(key) ?? [];
  // The same poll is recorded by multiple observers (Shell + active screen);
  // collapse identical timestamps so a point isn't duplicated on the chart.
  if (prev.length && prev[prev.length - 1].t === t) {
    const base = prev.slice();
    base[base.length - 1] = { t, v };
    series.set(key, base);
    return;
  }
  const base = prev.length >= CAPACITY ? prev.slice(prev.length - CAPACITY + 1) : prev.slice();
  base.push({ t, v });
  series.set(key, base);
}

/** Record a batch of named scalars from one poll; emits a single update. */
export function record(entries: Record<string, number | null | undefined>, t = Date.now()): void {
  let changed = false;
  for (const [key, raw] of Object.entries(entries)) {
    const v = typeof raw === "number" ? raw : raw == null ? NaN : Number(raw);
    if (Number.isFinite(v)) {
      append(key, v, t);
      changed = true;
    }
  }
  if (changed) emit();
}

function subscribe(cb: () => void): () => void {
  subs.add(cb);
  return () => {
    subs.delete(cb);
  };
}

/** Subscribe a component to one metric's rolling history. */
export function useSeries(key: string): Point[] {
  return useSyncExternalStore(
    subscribe,
    () => series.get(key) ?? EMPTY,
    () => EMPTY,
  );
}

export function latest(key: string): number | null {
  const s = series.get(key);
  return s && s.length ? s[s.length - 1].v : null;
}
