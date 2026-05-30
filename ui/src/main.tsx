import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";
import { applyThemeClass, settingsStore } from "./lib/settings";
import { QUERY_DEFAULTS, QUERY_TIMING } from "./lib/uiConfig";

// Theme: apply once, then react to setting changes and OS preference changes.
applyThemeClass(settingsStore.get().theme);
settingsStore.subscribe(() => applyThemeClass(settingsStore.get().theme));
window
  .matchMedia?.("(prefers-color-scheme: light)")
  .addEventListener?.("change", () => {
    if (settingsStore.get().theme === "system") applyThemeClass("system");
  });

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: QUERY_DEFAULTS.retryCount,
      refetchOnWindowFocus: false,
      gcTime: QUERY_TIMING.cacheGcMs,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/app">
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
