# Scope — Text / Event Alpha Ingestion Path

> **Sister doc:** [`text-event-v2-family.md`](text-event-v2-family.md)
> describes the **feature-factory** text family (LLM-extracted features
> registered under `family="text"`, `version="text-event-v2"`). That family
> ships 27 features across news + filings + earnings calls and is **distinct
> from the SEC ingestion pipeline scoped below** — different code path,
> different storage, different promotion gates. Operators choosing where to
> invest:
>
> - **Use this doc** if you want to *run the existing SEC EDGAR ingestion +
>   DeepSeek extraction + paper-alpha-catalyst pipeline* against a real
>   universe to test whether LLM-on-filings carries alpha.
> - **Use the v2 family doc** if you are *adding new text features, working
>   with the news/filing/earnings-call FeatureFrame*, or *promoting an
>   evidence-gated text feature* under the standard FeatureFamily contract.

## Context

Six classical campaigns and a GPU XGBoost run proved that price-derived factors
carry no promotable alpha on liquid US large-caps (best honest walk-forward OOS
IC ~0.05; XGBoost overfit to 0.02). The remaining path to genuine alpha is
**orthogonal signal** — information not in price. This document scopes the
text/event ingestion path to that end.

## Headline finding: this is a *run* project, not a *build* project (Tier 1)

The platform's text/event infrastructure is **already built and functional**.
The work is largely executing an existing pipeline, not writing one.

| Component | State | Notes |
|---|---|---|
| SEC EDGAR ingestion (`text-events ingest-sec`) | ✅ Functional | Public API, no key — only a User-Agent. Pulls 8-K/10-K/10-Q. |
| LLM feature extraction (`text-events extract-features`) | ✅ Functional | DeepSeek wired (key in `.env`); extracts 15 features; cached + budgeted. |
| Text feature builder (`paper-alpha-catalyst-v10`) | ✅ Functional | Feature audits already on disk. |
| Event-reaction builder (`paper-alpha-event-reaction-v2`) | ✅ Functional | Derived from SEC filing counts + price. Audits on disk. |
| Campaign `--signal-type text` / `event` | ✅ Functional | Walk-forward + gates. |
| Gates: `text-gate`, `signal-gate`, `production-candidate` | ✅ Functional | 20-obs / min-IC 0.05 promotion gate. |
| Ensemble wiring (text/event weights) | ✅ Functional | Config-driven; currently `shadow` mode. |
| `TextEventStore` (Postgres) | ✅ Functional | Durable, content-addressed, deduped. |
| **News ingestion** (`text-events ingest-news --vendor tws`) | ✅ TWS historical news | Uses IB/TWS API news subscriptions; persists headlines plus article bodies when available. |
| **Transcript ingestion** (`TranscriptTextProvider`) | ❌ Stub | Returns empty — no vendor wired. |
| **Intraday feature bundler** | ⚠️ Incomplete | Candidates exist; no `build_paper_alpha_intraday_*`; also needs minute bars. |

## Two tiers

### Tier 1 — SEC filings + LLM (cheap, built — do this first)

Everything needed is built. The scope is execution:

| WS | Work | Effort |
|----|------|--------|
| T1 | **Build CIK map** for the 337-name universe — small script over SEC's free `company_tickers.json` → `infra/config/sec_cik_map.json` | 0.5 day |
| T2 | **Run SEC ingestion** — `text-events ingest-sec` for 337 names, 2022–2026, forms 8-K/10-K/10-Q. ~10k filings, rate-limited (~0.25s/req) → ~1–2 h wall-clock | 0.5 day |
| T3 | **Run LLM extraction** — `text-events extract-features` (DeepSeek v4, prompt v4). ~10k filings × ~20k tokens ≈ 200M tokens ≈ **$60–100 one-time**. Budget cap is $25/day → either raise it or spread 3–4 days | 0.5 day + wall-clock |
| T4 | **Backfill features** — `paper-alpha-catalyst-v10` (text) and `paper-alpha-event-reaction-v2` (event) for the 337 universe; reuses `features backfill` | 0.5 day |
| T5 | **Campaign validation** — `research-campaign run --signal-type text` and `--signal-type event`. This answers: *does SEC-text alpha exist?* | 0.5 day |
| T6 | **Promotion** (only if T5 validates) — wire into ensemble `paper` mode, accumulate 20-day `signal-gate` evidence, `production-candidate` | 0.5 day + 20-day soak |

**Tier 1 total: ~3 days of work + LLM-extraction wall-clock + a 20-day soak.**

### Tier 2 — News + transcripts (expensive, unbuilt — defer)

SEC filings are *sparse* (a name files ~4–30×/year). The dense, timely text
alpha lives in **news headlines** and **earnings-call transcripts** — both
currently stubs.

| Work | Effort | Cost |
|------|--------|------|
| Expand beyond TWS news provider coverage (Benzinga / Polygon news / Tiingo news) | 5–10 days | Paid vendor sub |
| Wire transcripts into `TranscriptTextProvider` | 3–5 days | Paid vendor or scraper |

Defer Tier 2 until Tier 1 shows whether LLM-on-filings has signal — and whether
sparsity is the binding constraint.

## Costs & dependencies

- **SEC ingestion:** free (public EDGAR; User-Agent already set in `.env`).
- **LLM extraction:** DeepSeek key already in `.env`. ~$60–100 one-time backfill; ongoing daily cost for live (within the $25/day cap).
- **No new infrastructure** — Postgres `text_events` table and Alembic migration `006` already exist.
- Intraday is **out of scope**: needs minute bars (no data feed) and an incomplete bundler.

## Risks & honest caveats

1. **SEC-text alpha is unproven.** The original failing campaign
   (`walk_forward_durable_current_alpha_llm_*`) *included* LLM features and
   scored OOS IC 0.15 — encouraging, but on 15 names and it failed on costs.
   Tier 1's campaign (T5) is a genuine test, not a guaranteed win.
2. **SEC filings are sparse.** Quarterly/event-driven text gives a low-frequency
   signal. If T5 shows weak coverage-limited IC, the real lever is Tier 2.
3. **Event features are semi-price-derived.** `event_reaction_v2` blends filing
   *counts* with momentum — only partially orthogonal to classical factors.
4. **LLM cost scales with universe and history.** Budget the backfill; the
   replay cache makes re-runs free.

## Recommendation

Execute **Tier 1 (WS T1–T5)** — ~3 days of work to a real verdict on whether
LLM-extracted SEC-filing features carry promotable alpha. It is overwhelmingly
a *run-the-existing-pipeline* exercise. Decide on Tier 2 (news/transcripts) only
after T5's campaign result is in hand.
