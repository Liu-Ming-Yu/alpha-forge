<div align="center">

<p>
  <strong>Language</strong><br>
  <kbd>English · Current</kbd>
  <a href="README.zh-CN.md"><kbd>简体中文</kbd></a>
</p>

# Alpha Forge

### Introducing Alpha Forge &mdash; an agentic AI operating system for systematic trading.

<p>
  Built for a world where markets move through price, language, liquidity,
  positioning, macro pressure, and machine-readable narrative, Alpha Forge
  unifies research, LLM intelligence, machine learning, governance, and execution
  into one production architecture that works with IBKR.
</p>

<p>
  It turns filings, earnings calls, news, market data, microstructure,
  ownership, estimates, options, and macro signals into governed alpha
  candidates. Then it tests them, challenges them, versions them, audits them,
  and blocks them until evidence is strong enough.
</p>

<p>
  <sub>Agentic LLM Intelligence / Representation Learning / Autonomous Research / Governed Execution</sub>
</p>

<p>
  <a href="https://www.interactivebrokers.com/en/trading/ib-api.php?menu=A" aria-label="Open the official Interactive Brokers API page">
    <img src="docs/assets/ibkr-integration-badge.svg" alt="IBKR-ready execution: TWS API, IB Gateway, and governed paper-to-live pathway" width="520">
  </a>
</p>

<p>
  <a href="#command-deck"><kbd>Command Deck</kbd></a>
  <a href="#architecture-map"><kbd>Architecture Map</kbd></a>
  <a href="#agentic-llm-intelligence-layer"><kbd>LLM Layer</kbd></a>
  <a href="#autonomous-research-factory"><kbd>Research Factory</kbd></a>
  <a href="#production-execution-fortress"><kbd>Execution</kbd></a>
</p>

</div>

---

## Command Deck

<table>
  <tr>
    <td width="33%">
      <strong>Discover</strong><br>
      <sub>Explore the product surface.</sub><br><br>
      <a href="#showcase">Showcase</a><br>
      <a href="#architecture-map">Architecture Map</a><br>
      <a href="#technology-stack">Technology Stack</a>
    </td>
    <td width="33%">
      <strong>Investigate</strong><br>
      <sub>Open the core intelligence systems.</sub><br><br>
      <a href="#agentic-llm-intelligence-layer">Agentic LLM Layer</a><br>
      <a href="#text-event-v2">Text-Event-v2</a><br>
      <a href="#hybrid-ml--representation-learning-engine">Hybrid ML Engine</a>
    </td>
    <td width="33%">
      <strong>Operate</strong><br>
      <sub>Understand deployment and safety.</sub><br><br>
      <a href="#v2-shared-account-orchestrator">V2 Orchestrator</a><br>
      <a href="#production-execution-fortress">Execution Fortress</a><br>
      <a href="#governance--observability-fabric">Governance Fabric</a>
    </td>
  </tr>
</table>

<details open>
<summary><strong>Open Mission Control</strong> &mdash; Alpha Forge at a glance</summary>

| Mission | System Response |
| --- | --- |
| Convert language into alpha | Agentic LLM layer turns narrative into text-event-v2 features |
| Prevent uncontrolled AI behavior | Manifests, evals, startup assertions, audits, and human-review gates |
| Search across feature space | Campaigns, walk-forward validation, IC diagnostics, and bootstrap evidence |
| Merge many strategies safely | V2 account-level orchestrator resolves proposals into one portfolio target |
| Block unsafe execution | Fail-closed broker path, reconciliation, kill switches, and durable journals |

</details>

<details>
<summary><strong>Open Alpha Lifecycle</strong> &mdash; From raw signal to controlled live deployment</summary>

```text
Language + Market Data
  -> Agentic Intelligence
  -> Feature Families
  -> Campaign Research
  -> Walk-Forward Validation
  -> Evidence Package
  -> Production-Candidate Gate
  -> Shadow Mode
  -> Paper Trading
  -> Paper Soak
  -> Live Preflight
  -> Controlled Live
```

</details>

## Showcase

| System | Signature Capability |
| --- | --- |
| Agentic LLM Intelligence Layer | Multi-agent text-event-v2 alpha generation with tool use, manifests, traceable reasoning, evals, live assertions, and human-review gates |
| Hybrid ML + Representation Learning Engine | XGBoost, learned embeddings, PCA representations, formulaic alpha mining, microstructure, ownership, estimates, options, macro, and text-event intelligence |
| Autonomous Research Factory | Campaign-driven alpha discovery, walk-forward validation, IC diagnostics, bootstrap evidence, regime testing, and production-candidate packaging |
| V2 Shared-Account Orchestrator | Multi-engine portfolio target fusion with governed Shadow &rarr; Paper &rarr; Live promotion |
| Production Execution Fortress | Fail-closed broker execution, durable state, reconciliation, kill switches, strict service boundaries, and automated verification |
| Governance & Observability Fabric | Manifest registry, startup assertions, model provenance, event journals, readiness reports, promotion gates, and operator visibility |

## Architecture Map

![Alpha Forge architecture poster](docs/assets/alpha-forge-architecture.svg)

<details open>
<summary><strong>Explore Architecture Layers</strong> &mdash; tap into the stack</summary>

| Layer | What It Controls | Why It Matters |
| --- | --- | --- |
| Language Intelligence | Agents, text-event-v2 extraction, manifests, evals, assertions | Converts narrative into governed alpha inputs |
| Research Graph | Feature families, campaigns, walk-forward validation, diagnostics | Turns hypotheses into reproducible evidence |
| Portfolio Fusion | Multi-engine proposals, account budgets, risk overlays | Prevents strategy modules from fighting over the same account |
| Execution Safety | Pre-trade checks, broker adapters, kill switches, reconciliation | Stops unsafe action before it reaches the broker |
| Governance Fabric | Readiness reports, promotion gates, event journals, operator visibility | Makes AI-native trading auditable and controllable |

</details>

<details>
<summary><strong>Open Service Boundary Console</strong> &mdash; how the monolith stays disciplined</summary>

| Boundary | Rule |
| --- | --- |
| `core` | Owns contracts, domain models, and events |
| `application` | Owns use cases and read models |
| `services` | Own domain logic behind service boundaries |
| `infrastructure` | Owns durable adapters and external systems |
| `bootstrap` | Composes concrete runtime dependencies |
| `cli` / `views` | Stay thin and operator-facing |

</details>

---

## Agentic LLM Intelligence Layer

### From language to governed alpha.

The platform treats language as a first-class market data substrate.

Filings, earnings calls, macro commentary, news, transcripts, social narratives,
central-bank language, analyst revisions, and event-driven text are converted
into structured text-event-v2 features. These features are not uncontrolled LLM
outputs. They are manifest-bound, versioned, evaluated, audited, and
promotion-gated before they can influence a portfolio.

At the center is a governed multi-agent intelligence layer: a swarm of
specialized AI agents that reason over market context, challenge hypotheses,
inspect risks, and prepare candidate signals for quantitative validation.

<details open>
<summary><strong>Open Agent Console</strong> &mdash; specialized intelligence roles</summary>

| Agent | Role |
| --- | --- |
| Market Oracle | Converts market events, regime changes, and narrative shifts into structured research hypotheses |
| Sentiment Synthesizer | Extracts sentiment, uncertainty, surprise, and directional pressure from noisy language streams |
| Strategy Debater | Runs adversarial critique against candidate alpha ideas before they reach formal research |
| Risk Guardian | Detects overfitting, regime fragility, crowding, stale data, and hidden exposure |
| Evidence Auditor | Checks whether a signal has enough statistical and operational support for promotion |
| Execution Examiner | Reviews whether a strategy can survive cash, liquidity, broker, and order-routing constraints |

</details>

This is not &ldquo;prompt engineering.&rdquo;

It is governed machine reasoning for alpha research: agent outputs are
traceable, reproducible, inspected, and connected to downstream evidence.

## Text-Event-v2

### LLM-native alpha, engineered for production.

The text-event-v2 family turns unstructured language into quantitative features
that can be ranked, backtested, stress-tested, and governed like any other alpha
source.

It supports:

<details open>
<summary><strong>Open Text-Event Capability Matrix</strong> &mdash; LLM-native features under governance</summary>

| Capability | Description |
| --- | --- |
| Event Extraction | Converts raw text into structured market events, catalysts, risks, and directional hypotheses |
| Narrative Regime Detection | Tracks shifts in market language, macro tone, risk appetite, and sector-level narratives |
| Filing Intelligence | Extracts management tone, uncertainty, litigation risk, guidance changes, and operating pressure |
| Earnings Call Reasoning | Identifies surprise, confidence, hedging language, and forward-looking signal changes |
| Manifest-Governed Features | Every LLM-derived feature is tied to a schema, prompt contract, model version, and audit trail |
| Live Startup Assertions | Production startup fails if text models, manifests, or required feature contracts are missing or inconsistent |
| Promotion Readiness | Text-derived signals must survive the same evidence pipeline as price, fundamental, or microstructure features |

</details>

The result is a language intelligence layer that behaves less like a chatbot
and more like a governed research instrument.

---

## Hybrid ML + Representation Learning Engine

### Classical alpha, modern ML, and learned structure &mdash; in one research graph.

The research engine combines traditional quantitative signals with machine
learning and learned representations across the full feature universe.

It integrates:

<details open>
<summary><strong>Open Feature Universe</strong> &mdash; explicit domain knowledge plus learned structure</summary>

| Feature / Model Family | Purpose |
| --- | --- |
| Price & Momentum Features | Trend, reversal, volatility, distance-to-high, and cross-sectional ranking signals |
| Microstructure-v3 | Liquidity, spread, volume pressure, intraday behavior, and trading-friction-aware features |
| Ownership-v1 | Institutional behavior, positioning shifts, insider activity, and ownership structure |
| Estimates-v1 | Analyst expectations, revisions, dispersion, and forward-looking earnings pressure |
| Options-v1 | Volatility surface, skew, flow, and implied market expectation signals |
| Macro-v1 | Rate, inflation, liquidity, sector, and broad regime context |
| Formulaic Alpha Mining | Evolutionary search over WorldQuant-style grammars with auto-promotion constraints |
| Learned Representations-v1 | PCA-style embeddings and compressed cross-family structure from existing features |
| Text-Event-v2 | LLM-derived structured features from filings, calls, news, and narrative sources |
| XGBoost Ranking Pipelines | Nonlinear interaction discovery, cross-sectional scoring, and feature attribution |

</details>

This creates a research system that can discover alpha from both explicit domain
knowledge and latent structure hidden across feature families.

The platform does not rely on one model, one signal, or one narrative. It builds
an ensemble of evidence.

---

## Autonomous Research Factory

### Alpha discovery with memory, discipline, and promotion control.

Quant Platform is designed as an autonomous research factory: generate
hypotheses, build features, run experiments, evaluate results, preserve
artifacts, and promote only what survives.

The research loop is structured:

```text
Hypothesis -> Feature Family -> Campaign -> Walk-Forward Validation -> Diagnostics -> Evidence Package -> Promotion Decision
```

Each experiment produces reproducible artifacts: feature manifests, model
configs, fold results, IC diagnostics, bootstrap checks, turnover profiles,
drawdown curves, and production-candidate reports.

<details open>
<summary><strong>Open Research Diagnostics</strong> &mdash; how attractive illusions are rejected</summary>

| Research Capability | Description |
| --- | --- |
| Campaign-Based Research | Experiments are organized as campaigns instead of ad hoc notebooks |
| Walk-Forward Validation | Strategies are tested across rolling out-of-sample folds |
| IC Diagnostics | Signal quality is measured through rank correlation, stability, and negative-streak behavior |
| Bootstrap Evidence | Statistical uncertainty is measured before promotion |
| Turnover Analysis | Execution drag and capacity pressure are treated as first-class constraints |
| Regime Testing | Performance is analyzed across risk-on, risk-off, volatile, and low-liquidity periods |
| Production-Candidate Reports | Strategies are packaged with the evidence needed for governed deployment |

</details>

The system is built to reject attractive illusions: high-return backtests,
fragile edges, accidental leakage, unstable IC, and strategies that look good
only before costs.

---

## V2 Shared-Account Orchestrator

### Many engines. One account. One coherent portfolio.

Real trading is not a collection of independent strategy demos. Multiple
engines compete for the same cash, risk budget, exposure limits, and broker
connection.

The V2 Shared-Account Orchestrator solves this by merging independent strategy
proposals into a single account-level portfolio target.

Strategy engines can include:

<details open>
<summary><strong>Open Engine Rack</strong> &mdash; proposal sources for one account-level target</summary>

| Engine | Role |
| --- | --- |
| Cross-Sectional Equity Ranker | Selects and weights securities based on alpha scores |
| ETF Macro Allocator | Adjusts broad exposure based on macro and regime signals |
| Risk Overlay Engine | Reduces exposure when drawdown, volatility, or signal quality deteriorates |
| Text-Event Engine | Incorporates LLM-derived event signals into portfolio intent |
| Future Strategy Modules | Additional engines can submit proposals through the same governed interface |

</details>

The orchestrator applies cash constraints, exposure limits, stale-state checks,
throttles, kill switches, reconciliation, and broker-readiness checks before any
target becomes executable.

Deployment is staged:

```text
Shadow Mode -> Paper Trading -> Paper Soak -> Live Preflight -> Controlled Live
```

Nothing becomes live by accident.
Nothing bypasses governance.
Nothing trades without a validated account-level view.

---

## Production Execution Fortress

### Designed to stop before it fails.

The execution layer is built around a conservative principle:

If the system cannot prove it is safe to act, it does not act.

Orders pass through a fail-closed execution path with broker abstraction,
pre-trade validation, state reconciliation, event journaling, and
promotion-state checks.

<details open>
<summary><strong>Open Safety Matrix</strong> &mdash; proof before action</summary>

| Safeguard | Purpose |
| --- | --- |
| Fail-Closed Execution | Blocks action when state is stale, invalid, incomplete, or unverifiable |
| Durable Portfolio State | Persists snapshots, parent-child provenance, and account state across restarts |
| Event Journal | Records execution events, promotion events, and operational state transitions |
| Broker Reconciliation | Compares intended portfolio state with broker/account reality |
| Kill Switches | Stops trading when operational or risk conditions become unsafe |
| Order Throttling | Prevents runaway order submission and broker abuse |
| Paper/Live Separation | Keeps simulation, paper, and live paths explicitly separated |
| IBKR Gateway/TWS Integration | Supports broker-backed paper and live execution through controlled adapters |

</details>

The system is not optimized to appear active.
It is optimized to be correct, observable, and controlled.

---

## Governance & Observability Fabric

### The control plane for AI-native trading.

Modern AI systems are expected to be observable, testable, and governable,
especially when agents use tools and interact with external systems. OpenAI&rsquo;s
[Agents](https://platform.openai.com/docs/guides/agents),
[Agents SDK tracing](https://openai.github.io/openai-agents-python/tracing/),
and [human-in-the-loop](https://openai.github.io/openai-agents-python/human_in_the_loop/)
documentation directly emphasize tools, integrations, observability, guardrails,
and human review as core production concepts.

Quant Platform applies those ideas to systematic trading.

<details open>
<summary><strong>Open Governance Console</strong> &mdash; the control plane for AI-native trading</summary>

| Governance Layer | Function |
| --- | --- |
| Feature Registry | Tracks all feature families and their schemas |
| Model Manifest System | Records model versions, feature contracts, prompts, configs, and artifacts |
| Startup Assertions | Prevents production startup when required contracts are missing |
| Readiness Reports | Summarizes whether a strategy is eligible for paper or live progression |
| Promotion Gates | Blocks strategies that fail evidence, risk, or operational requirements |
| Traceable Agent Actions | Makes LLM/agent behavior auditable instead of opaque |
| Human Review Hooks | Allows sensitive transitions to require explicit approval |
| Operator API | Provides controlled visibility into system state and readiness |

</details>

The platform does not trust AI output because it sounds intelligent.
It trusts only what can be measured, replayed, audited, and governed.

---

## Technology Stack

<details open>
<summary><strong>Open Stack Matrix</strong> &mdash; production components behind Alpha Forge</summary>

| Area | Stack |
| --- | --- |
| Language & Tooling | Python 3.11, uv, pyproject.toml, pre-commit, ruff, mypy, pytest |
| ML / Representation Learning | XGBoost, learned representations, PCA embeddings, feature attribution, rank modeling |
| LLM / Agentic AI | Text-event-v2 pipelines, manifest-governed agents, tool-using research workflows, audit trails |
| Data & Research Storage | Parquet, object-store-ready artifacts, campaign outputs, reproducible experiment persistence |
| State & Infrastructure | PostgreSQL, Alembic, Redis/event-bus patterns, durable event journals |
| Execution | Simulated backend, paper backend, IBKR ibapi, controlled live-adapter pathway |
| Deployment | Docker, docker-compose, Makefile orchestration, FastAPI operator API |
| Verification | make verify, import-boundary checks, module-size guards, type-debt enforcement, regression tests |
| Governance | Readiness reports, production-candidate gates, startup assertions, promotion evidence, paper-soak validation |

</details>

---

## Operating Surface

The installed console script and module entrypoint are equivalent:

```bash
quant-platform --help
python -m quant_platform --help
```

## Setup

Use Python 3.11 for project verification.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,api]"
```

On macOS/Linux/WSL:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,api]"
```

Optional extras:

```bash
python -m pip install -e ".[ml]"       # XGBoost research
python -m pip install -e ".[backtest]" # vectorbt backtests
```

`ibapi` is not a default dependency. Install it from the IBKR TWS API
distribution when using real IBKR paper/live paths.

## Configuration

Settings are loaded by `PlatformSettings` from `.env` and environment variables
with the `QP__` prefix.

Start from:

```bash
copy infra\config\settings.example.env .env
```

Minimal local in-memory development:

```bash
set QP__STORAGE__POSTGRES_DSN=
set QP__STORAGE__REDIS_URL=
set QP__STORAGE__EVENT_BUS_BACKEND=in_memory
set QP__BROKER__PAPER_TRADING=true
```

Durable paper/live requires at least:

```bash
QP__STORAGE__POSTGRES_DSN=postgresql+psycopg://user:pass@host:5432/quant_platform
QP__STORAGE__REDIS_URL=redis://localhost:6379/0
QP__STORAGE__EVENT_BUS_BACKEND=redis_streams
QP__API__OPERATOR_API_KEY=<strong random key>
```

Common IBKR ports:

| App | Mode | Port |
| --- | --- | ---: |
| TWS | Paper | `7497` |
| TWS | Live | `7496` |
| IB Gateway | Paper | `4002` |
| IB Gateway | Live | `4001` |

## Common Commands

Schema:

```bash
python -m quant_platform migrations-check
python -m quant_platform migrate
python -m quant_platform verify-schema
```

Single paper cycle:

```bash
python -m quant_platform run-cycle --initial-cash 50000
```

Bounded engine runs:

```bash
python -m quant_platform run-engine --mode shadow --cycles 5
python -m quant_platform run-engine --mode paper --execution-backend ib-paper --contracts-file infra/config/paper_contracts.json --cycles 1
python -m quant_platform run-engine --mode live --contracts-file ./contracts.json --cycles 1
```

V2 multi-engine proposal/orchestration path:

```bash
python -m quant_platform run-multi-engine ^
  --engines cross_sectional_equity,etf_macro_allocator ^
  --budgets-file ./budgets.json ^
  --mode paper ^
  --contracts-file infra/config/paper_contracts.json ^
  --cycles 1
```

Data and research:

```bash
python -m quant_platform ingest --start YYYY-MM-DD --end YYYY-MM-DD --contracts-file ./contracts.json
python -m quant_platform maintain --interval 900 --contracts-file ./contracts.json
python -m quant_platform compute-features --contracts-file ./contracts.json
python -m quant_platform features backfill --contracts-file ./contracts.json --start YYYY-MM-DDT00:00:00+00:00 --end YYYY-MM-DDT00:00:00+00:00 --feature-set-version paper-alpha-composite-v1 --date-policy nyse-sessions
python -m quant_platform boosting gpu-check
python -m quant_platform research-campaign run --help
```

Operator API:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
set QP__API__OPERATOR_API_KEY=<generated key>
python -m quant_platform serve-api --host 127.0.0.1 --port 8000
curl -H "X-API-Key: %QP__API__OPERATOR_API_KEY%" http://127.0.0.1:8000/health/ready
```

## Verification

Fast local checks:

```bash
python scripts/check_import_boundaries.py
python scripts/check_service_coupling.py
python scripts/check_module_size.py
python scripts/check_type_debt.py
python scripts/check_lint_debt.py --skip-ruff-probe
python -m pytest -q tests/unit/test_engine_loop.py
```

Full offline gate:

```bash
make verify
```

Durable and live gates are opt-in:

```bash
set QP_VERIFY_DURABLE=1
set QP_VERIFY_LIVE_IBKR=1
set IBAPI_PACKAGE_PATH=<path-to-TWS-API/source/pythonclient>
make verify
```

## Documentation Map

Start with [USEME.md](USEME.md) for operator commands and
[CONTEXT.md](CONTEXT.md) for project vocabulary.

Architecture:

- [System overview](docs/architecture/system-overview.md)
- [Service boundaries](docs/architecture/service-boundaries.md)
- [Production roadmap](docs/architecture/production-roadmap.md)
- [Risk register](docs/architecture/risk-register.md)
- [Source audit checklist](docs/architecture/source-audit-checklist.md)
- [V2 execution flow](docs/architecture/v2-execution-flow.md)
- [Core contracts](docs/interfaces/core-contracts.md)

Runbooks live under [docs/runbooks](docs/runbooks/). Use them for operations,
incident response, paper promotion, backups, IBKR recovery, data recovery, and
production readiness.
