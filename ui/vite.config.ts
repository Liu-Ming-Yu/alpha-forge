import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The console is served by the operator API under the public `/app` prefix
// (see src/quant_platform/views/operator_api/static.py). Using the same base
// in dev means React Router's basename and asset URLs match prod exactly.
const API_TARGET = "http://127.0.0.1:8000";

// Every non-`/app` path the console talks to is an operator-API endpoint.
// In dev, proxy them to the running API so the client always uses same-origin
// relative URLs (no CORS, identical to prod).
const API_PREFIXES = [
  "/console",
  "/operator",
  "/dashboard",
  "/health",
  "/cash",
  "/blotter",
  "/metrics",
  "/audit",
  "/orders",
  "/signals",
  "/strategy",
  "/research",
  "/v1",
];

export default defineConfig({
  base: "/app/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_PREFIXES.map((p) => [p, { target: API_TARGET, changeOrigin: true }]),
    ),
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    chunkSizeWarningLimit: 1100,
  },
});
