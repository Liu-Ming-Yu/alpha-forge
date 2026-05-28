# Contributing

This project is allowed to touch live brokerage workflows, so every change is
treated as production work even when the immediate target is research or docs.

## CI Discipline

No PR is mergeable with red checks. Treat failing unit tests, integration tests,
typecheck, lint, coverage, architecture ratchets, and generated-artifact checks
as release blockers.

For source acceptance, full `pytest -q` and `mypy src` matter. Marker-only
subsets are useful diagnostics, not substitutes for acceptance evidence.

Use `make verify` for the full offline gate. Durable Postgres/Redis tests and
live IBKR tests are opt-in and must be called out when skipped.

## Architecture Boundaries

The expected dependency direction is:

```text
cli/views -> bootstrap -> infrastructure
cli/views -> application -> services -> core
engines/bootstrap compose concrete runtime dependencies
core imports no outer layers
```

Rules:

- `core` owns contracts, domain models, and events.
- `services` depend on core contracts and same-service helpers.
- `application` owns use cases and read models; it must not construct concrete
  infrastructure.
- `infrastructure` owns adapters and must not import entrypoints.
- `bootstrap` owns composition.
- CLI and API handlers should stay thin.

Run these before opening a PR that changes imports or module layout:

```bash
python scripts/check_import_boundaries.py
python scripts/check_service_coupling.py
python scripts/check_module_size.py
```

## Safety Rules

- Do not commit `.env` secrets, API keys, broker account identifiers, or private
  credentials.
- Do not add new source `# type: ignore` or bare `# noqa` debt.
- Do not introduce live submission paths that bypass cash, risk, execution
  policy, kill-switch, and reconciliation gates.
- New operator-controlled state should be durable or have an explicit restart
  hydration path.
- In-memory adapters are acceptable for local tests and paper development, not
  for live readiness evidence.

## Documentation

When changing runtime behavior, update the closest doc:

- `README.md` for top-level project shape and setup.
- `USEME.md` for operator commands.
- `CONTEXT.md` for vocabulary.
- `docs/architecture/*` for design and risk decisions.
- `docs/runbooks/*` for repeatable operations and incidents.
