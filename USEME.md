# USEME

This is the practical operator/developer entrypoint for the current workspace.
Use it when you need to install, run, verify, or orient inside the project.

## Quick Identity

`quant-platform` is a single-operator IBKR cash-account quant platform. It is a
modular monolith: the service packages run in one process, but dependency
direction is enforced by contracts, architecture checks, and tests.

Hot path:

```text
data -> research -> signal -> portfolio -> execution -> broker
```

The platform supports shadow runs, simulated paper runs, IBKR paper runs, and
guarded live sessions. It also has governed XGBoost/text research paths and a
V2 shared-account orchestrator for multi-engine execution.

## First Setup

Use Python 3.11 for verification:

```bash
py -3.11 -m venv .venv-verify
.venv-verify\Scripts\python.exe -m pip install --upgrade pip
.venv-verify\Scripts\python.exe -m pip install -c constraints/py311.txt -e ".[dev,api,ml]"
```

For the normal dev environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,api]"
```

`make verify` recreates `.venv-verify`. Treat that environment as disposable.

## Environment

Copy the sample config:

```bash
copy infra\config\settings.example.env .env
```

Local in-memory defaults:

```bash
QP__STORAGE__POSTGRES_DSN=
QP__STORAGE__REDIS_URL=
QP__STORAGE__EVENT_BUS_BACKEND=in_memory
QP__BROKER__PAPER_TRADING=true
```

Durable paper/live basics:

```bash
QP__STORAGE__POSTGRES_DSN=postgresql+psycopg://user:pass@host:5432/quant_platform
QP__STORAGE__REDIS_URL=redis://localhost:6379/0
QP__STORAGE__EVENT_BUS_BACKEND=redis_streams
QP__API__OPERATOR_API_KEY=<strong random key>
```

IBKR host/port defaults:

| Host setup | Common value |
| --- | --- |
| Windows IBKR from WSL2/Docker | `QP__BROKER__HOST=host.docker.internal` |
| Same host namespace | `QP__BROKER__HOST=127.0.0.1` |
| TWS paper | `QP__BROKER__PORT=7497` |
| TWS live | `QP__BROKER__PORT=7496` |
| IB Gateway paper | `QP__BROKER__PORT=4002` |
| IB Gateway live | `QP__BROKER__PORT=4001` |

## CLI Discovery

```bash
python -m quant_platform --help
python -m quant_platform run-engine --help
python -m quant_platform features --help
python -m quant_platform production-candidate --help
```

The root command currently exposes runtime, broker, data, migration, research,
governance, API, event-bus, text-event, paper-soak, dataset-quorum, simulator,
and smoke-test command groups.

## Everyday Commands

Schema:

```bash
python -m quant_platform migrations-check
python -m quant_platform migrate
python -m quant_platform verify-schema
```

Paper cycle:

```bash
QP__STORAGE__POSTGRES_DSN= ^
QP__STORAGE__REDIS_URL= ^
QP__STORAGE__EVENT_BUS_BACKEND=in_memory ^
python -m quant_platform run-cycle --initial-cash 50000
```

Bounded engine:

```bash
python -m quant_platform run-engine --mode shadow --cycles 5
python -m quant_platform run-engine --mode paper --execution-backend ib-paper --contracts-file infra/config/paper_contracts.json --cycles 1
```

Continuous paper supervision:

```bash
python -m quant_platform supervise ^
  --mode paper ^
  --execution-backend ib-paper ^
  --contracts-file infra/config/paper_contracts.json ^
  --interval 300
```

V2 multi-engine:

```bash
QP__V2__ENABLED=true ^
QP__V2__ACCOUNT_ORCHESTRATOR_ENABLED=true ^
python -m quant_platform run-multi-engine ^
  --engines cross_sectional_equity,etf_macro_allocator ^
  --budgets-file ./budgets.json ^
  --mode paper ^
  --contracts-file infra/config/paper_contracts.json ^
  --cycles 1
```

Operator API:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
set QP__API__OPERATOR_API_KEY=<generated key>
python -m quant_platform serve-api --host 127.0.0.1 --port 8000
curl -H "X-API-Key: %QP__API__OPERATOR_API_KEY%" http://127.0.0.1:8000/health/ready
curl -H "X-API-Key: %QP__API__OPERATOR_API_KEY%" http://127.0.0.1:8000/dashboard/summary
```

Data/research:

```bash
python -m quant_platform ingest --start YYYY-MM-DD --end YYYY-MM-DD --contracts-file ./contracts.json
python -m quant_platform maintain --interval 900 --contracts-file ./contracts.json
python -m quant_platform features backfill --help
python -m quant_platform features audit run --help
python -m quant_platform research-campaign run --help
python -m quant_platform boosting gpu-check
python -m quant_platform tearsheet --help
```

Walk-forward backtest of the latest stack (full alpha pipeline; A–H arms):

```bash
python scripts/backtest_latest_stack.py --arms G --out-root data/parquet/research/backtest_latest_stack_realized_v2
python scripts/backtest_latest_stack.py --arms A,B,C,D,E,F,G,H --max-workers 4 \
  --out-root data/parquet/research/backtest_latest_stack_realized_v2
```

Per-arm evidence lands as ``arm_<canonical_name>.json`` alongside a
``run_manifest.json`` index. ``H`` (regime overlay) additionally pins a
``regime_detector`` metadata block per ADR-005. See
[memory/project_backtest_latest_stack.md](https://github.com/Liu-Ming-Yu/Quant/blob/main/) for
the lineage and the canonical evidence path.

Formulaic alpha mining (Phase 4 — random + evolutionary search over the
WorldQuant-style expression grammar, with admission gate + auto-promotion):

```bash
python scripts/mine_alphas.py --help          # mine candidates → JSONL
python scripts/promote_alphas.py --help       # promote admitted candidates into the auto-library
```

Sharadar fundamentals (one-off pulls for the fundamentals-v1 family;
requires ``NASDAQ_DATA_LINK_API_KEY`` in ``.env``):

```bash
python scripts/pull_sharadar_sf1.py --help
python scripts/build_sharadar_ticker_map.py --help
```

Governance:

```bash
python -m quant_platform preflight --help
python -m quant_platform readiness --help
python -m quant_platform paper-soak --help
python -m quant_platform production-candidate --help
python -m quant_platform dataset-quorum --help
python -m quant_platform simulator-calibration --help
```

## Verification

Targeted checks after architecture or docs work:

```bash
python scripts/check_import_boundaries.py
python scripts/check_service_coupling.py
python scripts/check_module_size.py
python scripts/check_type_debt.py
python scripts/check_lint_debt.py --skip-ruff-probe
python -m pytest -q tests/unit/test_engine_loop.py
```

Full offline verification:

```bash
make verify
```

Optional gates:

```bash
set QP_VERIFY_DURABLE=1
set QP_VERIFY_LIVE_IBKR=1
set IBAPI_PACKAGE_PATH=<path-to-TWS-API/source/pythonclient>
make verify
```

## Source Map

| Path | Purpose |
| --- | --- |
| `src/quant_platform/core` | Domain models, events, protocol contracts |
| `src/quant_platform/application` | Use-case request/result models and operator read models |
| `src/quant_platform/services` | Pure service logic for data, research, signal, portfolio, execution, governance |
| `src/quant_platform/research/features` | Feature factory: per-family packages (`price_volume`, `fundamentals`, `formulaic`, `learned`, `regime`, `text`, etc.) registered via `bootstrap_default_families()` into a process-global `FeatureRegistry` |
| `src/quant_platform/infrastructure` | Postgres, Redis, metrics, stores, support adapters |
| `src/quant_platform/bootstrap` | Runtime composition for sessions, engines, broker, API, governance, persistence |
| `src/quant_platform/engines` | Engine lifecycle, proposal generation, V2 orchestration |
| `src/quant_platform/cli` | Argparse registry and command bindings |
| `src/quant_platform/views/operator_api` | FastAPI operator API |
| `scripts` | Architecture, verification, backup, and safety scripts |
| `tests` | Unit, integration, e2e, and live opt-in tests |
| `infra` | Example env files, observability config, contract and feature-card inputs |
| `docs` | Architecture notes, contracts, and runbooks |

## Current Review Notes

- Import boundaries are enforced and currently clean when `scripts/check_import_boundaries.py` passes.
- Direct cross-service imports are blocked by `scripts/check_service_coupling.py`.
- Production module size is guarded at 300 lines by `scripts/check_module_size.py`.
- Type debt is explicitly ratcheted: source `# type: ignore` comments are blocked.
- Lint debt is tracked by `scripts/check_lint_debt.py`.
- Generated source/test artifacts are rejected by `scripts/check_generated_artifacts.py`.

## Before Commit

Run at least the targeted checks for the area you touched. For production
acceptance work, run `make verify` and record any skipped durable/live gates
explicitly.

Do not commit:

- `.env` secrets
- broker account identifiers
- `.venv`, `.venv-verify`, caches, or coverage files
- generated Parquet data
- research artifacts unless the change explicitly adds tracked fixture data
