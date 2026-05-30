import { createStore } from "./store";
import { TIME } from "./uiConfig";

export type Theme = "dark" | "light" | "system";
export const PAUSED_CADENCE = 0;
export const CADENCE_OPTIONS = [
  { value: PAUSED_CADENCE, label: "Off", shortLabel: "Pause", ariaLabel: "Pause live updates" },
  { value: TIME.secondMs, label: "1s", shortLabel: "1s", ariaLabel: "Refresh every 1s" },
  { value: 3 * TIME.secondMs, label: "3s", shortLabel: "3s", ariaLabel: "Refresh every 3s", default: true },
  { value: 10 * TIME.secondMs, label: "10s", shortLabel: "10s", ariaLabel: "Refresh every 10s" },
] as const;
export type Cadence = (typeof CADENCE_OPTIONS)[number]["value"];
export const DEFAULT_CADENCE =
  CADENCE_OPTIONS.find((option) => "default" in option && option.default)?.value
  ?? CADENCE_OPTIONS[0].value;
export type Density = "comfortable" | "compact";

export interface Settings {
  apiBase: string;
  apiKey: string;
  theme: Theme;
  cadence: Cadence;
  density: Density;
}

const STORAGE_KEY = "qp.console.settings.v1";

const DEFAULTS: Settings = {
  apiBase: "",
  apiKey: "",
  theme: "dark",
  cadence: DEFAULT_CADENCE,
  density: "comfortable",
};

function load(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULTS;
    return { ...DEFAULTS, ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    return DEFAULTS;
  }
}

export const settingsStore = createStore<Settings>(load());

settingsStore.subscribe(() => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settingsStore.get()));
  } catch {
    /* storage unavailable — keep running with in-memory state */
  }
});

export function updateSettings(patch: Partial<Settings>): void {
  settingsStore.set(patch);
}

/** Resolve "system" against the OS preference; returns "dark" | "light". */
export function effectiveTheme(theme: Theme): "dark" | "light" {
  if (theme !== "system") return theme;
  const prefersLight =
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-color-scheme: light)").matches;
  return prefersLight ? "light" : "dark";
}

export function applyThemeClass(theme: Theme): void {
  const resolved = effectiveTheme(theme);
  const root = document.documentElement;
  root.classList.toggle("dark", resolved === "dark");
  root.classList.toggle("light", resolved === "light");
  root.style.colorScheme = resolved;
}
