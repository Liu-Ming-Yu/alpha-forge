# ADR-013 — Browser operator console (single-page UI served by the operator API)

**Status:** Accepted. A greenfield React/TypeScript single-page application (the "operator console") ships under `ui/`, is built to `ui/dist/`, and is served as static files by the existing operator API process. It is **capability-driven** (the UI renders only what `/operator/capabilities` advertises), **live** (TanStack Query polling on an operator-controlled cadence, with a client-side rolling-history ring buffer driving the live charts), and **read-mostly** — the only mutating action it exposes is the one write control the API already has (`POST /v1/kill-switch/clear`), behind a typed confirmation. Two small read-only backend endpoints are added: a public `GET /console/info` (bootstrap) and an authenticated `GET /v1/config/effective` (the modes/settings inspector). No change to the trading hot path.

## Context

The platform is a single-operator IBKR cash-account quant system: a Python modular monolith with a FastAPI **operator API** (`views/operator_api`) that already serves a rich read-only surface — `/dashboard/summary` (an aggregator), broker health, cash/ledger, order blotter, paper-gate metrics, strategy lifecycle, regime, signal decay, engine budgets/exposure, signal contributions, forecast evidence, research campaigns, feature audits, IC reports, paper-soak, readiness, and promotion-candidate — all behind `X-API-Key`, plus `/operator/capabilities` (advertises features, `write_controls`, and the `viewer/operator/admin` roles) and one write control, `POST /v1/kill-switch/clear`.

Until now the only operator surface was `curl` and the CLI. The goal: a browser UI for the **whole** system that (a) shows live graphs and monitors and (b) lets the operator see/manage modes and settings — held to strong HCI principles and an Apple-grade aesthetic.

Constraints that shaped the decision:
- **Modular monolith, single process.** A second long-running service (e.g. a Next.js server) is at odds with the deployment model and the safety story.
- **Architecture checks** enforce ≤300-line Python modules, no cross-service imports, and reject generated artifacts under `src/`. The frontend toolchain (`node_modules`, `dist`) must live *outside* `src/` and be excluded from those checks.
- **Safety-first.** The API is read-only by design; CORS is denied by default; auth is dual-opt-in. The UI must not invent control surfaces the backend doesn't actually have, and must gate the one dangerous action it does expose.
- **Windows + Python 3.11 + uv** for the backend; **Node 24 + npm** confirmed available for the frontend build.

## Decision

**1. Stack — Vite + React 18 + TypeScript, Tailwind, TanStack Query, Recharts, Framer Motion, lucide-react.** A static SPA is the right tool for an internal real-time console: no SSR/SEO needs, instant client-side navigation, trivial to host. The chosen libraries are the smallest set that delivers polish (Tailwind tokens + Framer motion + Recharts) without a bespoke component framework.

**2. Hosting — the operator API serves the built SPA.** `views/operator_api/static.py` mounts `ui/dist` (resolved relative to the repo root, overridable via `QP__API__CONSOLE_DIST_DIR`) with an SPA fallback to `index.html`; API routes keep priority. Result: `serve-api` serves the console at `/` on the same origin as the API — **no CORS in production**, one process, one deploy, matching the monolith. If `ui/dist` is absent, a friendly placeholder explains how to build it. Development uses the Vite dev server (`:5173`) with a proxy to `:8000`, so the client always speaks same-origin relative URLs.

**3. Live data — polling + client-side history.** TanStack Query polls `/dashboard/summary` (one request hydrates most of Overview) plus per-screen queries, on an operator-selectable cadence (1s / 3s / 10s / paused). Because the API is point-in-time (no NAV/metric time-series endpoint), the client keeps a bounded in-memory **ring buffer** of polled scalars (cash, available, exposure, throttle tokens, Sharpe, IC, regime vol …) to render genuinely live sparklines/area charts. Server-side history is a deliberate future enhancement, not a launch dependency.

**4. Capability-driven, read-mostly control.** The UI reads `/operator/capabilities` and renders only what is advertised — unsupported `write_controls` are hidden, not shown-disabled-with-excuses. The single mutating action, kill-switch clear, lives in Execution under a "Danger zone" with a typed confirmation (the endpoint already requires a `confirmation` string). Modes/settings are presented through a read-only **effective-config inspector** (`GET /v1/config/effective`, whitelisted non-secret fields) with an explicit "set at launch" affordance, because run-mode is a process-launch concern (CLI/env), not a runtime toggle — the UI is honest about that rather than faking a live switch.

**5. Two new read-only endpoints (small, boundary-respecting).**
- Public `GET /console/info` → `{api_version, requires_auth, modes, profiles}` so the shell can render + drive the connect flow before a key exists (mirrors the public `/health`; no secrets).
- Authenticated `GET /v1/config/effective` → a whitelisted snapshot (run mode/profile hints, broker host/port **without** account id, storage backends, alpha source weights, risk/execution limits, feature flags, capabilities) for the Settings inspector.

## Options considered

### Served-by-API SPA vs separate frontend server (Next.js) vs server-rendered templates (Jinja/HTMX)
Chose the **served-by-API SPA**. Next.js adds a second process, a Node runtime in production, and a CORS/auth seam — all friction for a single-operator monolith. Server-rendered HTMX keeps one process but makes the live-chart, capability-gated, Apple-grade interaction model far harder to reach. A static SPA mounted on the existing API keeps one process and one origin while giving full control over the client experience. Trade-off: a build step and a `node_modules`/`dist` footprint outside `src/` (gitignored, excluded from artifact checks).

### Live transport — polling vs SSE vs WebSocket
Chose **polling** for launch. The API is already request/response JSON; polling needs zero backend change, degrades gracefully, and — paired with smooth chart animation and a steady cadence — reads as "live." SSE (tailing the Redis Streams event bus) is the natural upgrade for true push and is left as a clean future step behind the same client data hook. WebSockets are overkill for a one-way monitor.

### Client-side history vs new time-series endpoints
Chose **client-side ring buffer** for the live charts to avoid expanding the backend surface (and a migration) on day one. Cost: history resets on reload — acceptable for a *live monitor* (vs. historical analytics, which the research/IC endpoints already cover). A server-side NAV/metric series is the documented follow-up.

### Capability-gating vs static menu
Chose **capability-gating** (HCI error-prevention + "aesthetic and minimalist"): the API already advertises `write_controls`/`features`/`roles`, so the UI shows exactly what this instance can do rather than dead controls.

## HCI principles applied

- **Visibility of system status** — persistent connection + live indicators, a global `as_of` clock, per-card freshness, a broker status "lamp," and a prominent kill-switch banner when active.
- **Match between system and real world** — navigation follows the trading hot path (Overview → Strategy → Execution → Research → Settings); domain language throughout; metric tooltips (what *is* IC / slippage ratio).
- **User control & freedom** — pause-live toggle, cadence control, reversible preferences, confirm-and-cancel on the only destructive action.
- **Consistency & standards** — one design-token set; one component kit; Apple HIG-aligned color/typography/motion.
- **Error prevention** — capability-gated controls; typed confirmation for kill-switch; the `/dashboard/summary` per-section `{error}` envelopes are surfaced as graceful per-card error states, never a blank screen.
- **Recognition over recall** — persistent nav, selected-run context carried across screens.
- **Flexibility & efficiency** — theme (light/dark/system), density, cadence, keyboard focus order.
- **Aesthetic & minimalist design** — progressive disclosure: each card shows only its essential metric, drill-downs for detail.
- **Help users recover from errors** — explicit error cards with retry; a connect screen on 401.
- **Accessibility** — AA contrast, focus-visible rings, status conveyed by icon+text (not color alone), `prefers-reduced-motion` respected.

## Aesthetic system (Apple-grade)

System font stack (`-apple-system`/SF → Inter fallback), tabular-nums for metrics; an 8-pt spacing grid; layered translucent surfaces with hairline borders, large radii, and soft shadows; vibrancy/backdrop-blur on the nav rail and top bar; a restrained neutral palette with one accent (SF blue) and semantic green/amber/red used sparingly; purposeful Framer motion (staggered card entrance, animated metric transitions, a live pulse) that yields to `prefers-reduced-motion`. Dark-first (trading context) with a light theme.

## Information architecture (screens)

- **Overview** — command center: live NAV/available-cash hero with streaming area chart, mode/profile pill, broker-health lamp, kill-switch state; a grid of live monitors (regime, exposure, throttle, paper-gate metrics, data freshness, readiness/promotion, recent audit feed).
- **Strategy** — runs list + selected-run lifecycle (health badge, Sharpe/IC/drawdown gauges), signal decay, signal-source contributions, forecast evidence (pass/stale/blockers), engine budgets + combined exposure.
- **Execution** — broker health detail, live order blotter, unmatched fills, compliance violations, cash-ledger detail, order-allocation drill-down; kill-switch control (gated, typed confirm).
- **Research** — campaigns list + detail, feature audits, IC reports, paper-soak sections, readiness snapshot, promotion candidate.
- **Settings** — Connection (API base + key, test), Modes & Config (effective-config inspector, read-only), Preferences (theme/cadence/density), Capabilities (what this instance allows), Danger zone (kill-switch clear).

## Consequences

- **Positive.** One process, one origin, no CORS in prod; the UI can never drift ahead of backend capability (it reads them); the trading hot path is untouched; the live feel is achieved with zero new streaming infrastructure; the design system is reusable.
- **Negative / deferred.** A frontend build step and a Node toolchain are now part of the dev/release loop (documented in USEME). Live charts are session-local until a server-side series exists. True push (SSE over the event bus) and richer write controls (mode changes, promotions) are deferred behind the capability flags that already exist for them.
- **Revisit when.** (1) Multi-operator/role enforcement becomes real → wire the advertised `viewer/operator/admin` roles into the UI and add server-side preferences. (2) Historical dashboards are needed → add NAV/metric time-series endpoints and switch the charts off the ring buffer. (3) Latency/scale pressure → upgrade the data hook from polling to SSE.
