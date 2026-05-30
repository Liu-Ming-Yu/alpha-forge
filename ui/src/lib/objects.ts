/** Small defensive helpers for rendering loosely-typed API payloads. */

export function isObj(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === "object" && !Array.isArray(v);
}

/** Top-level primitive (string/number/bool/null) fields, for KeyValue lists. */
export function scalarEntries(o: unknown): [string, string | number | boolean | null][] {
  if (!isObj(o)) return [];
  return Object.entries(o).filter(
    ([, v]) => v === null || ["string", "number", "boolean"].includes(typeof v),
  ) as [string, string | number | boolean | null][];
}

/** Read a list out of a wrapped payload, e.g. {entries:[...]} → [...]. */
export function listFrom<T = unknown>(o: unknown, key: string): T[] {
  if (Array.isArray(o)) return o as T[];
  if (isObj(o) && Array.isArray(o[key])) return o[key] as T[];
  return [];
}

export function pick<T = unknown>(o: unknown, ...keys: string[]): T | undefined {
  if (!isObj(o)) return undefined;
  for (const k of keys) if (o[k] !== undefined) return o[k] as T;
  return undefined;
}
