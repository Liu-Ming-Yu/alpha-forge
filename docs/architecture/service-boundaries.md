# Service Boundaries

## Dependency Rules

The intended dependency direction is:

```text
cli/views -> application -> services -> core
cli/views -> bootstrap -> infrastructure
engines/bootstrap compose concrete runtime dependencies
core imports no outer layer
```

Specific rules:

1. `core` owns domain models, events, and protocol contracts. It never imports
   services, application, infrastructure, bootstrap, CLI, views, engines, or
   session code.
2. `services` own package-local business logic. They depend on core contracts,
   core domain models, same-service helpers, and pure shared helpers.
3. `application` owns use cases, request/response models, and operator read
   models. It must not construct concrete infrastructure adapters.
4. `infrastructure` owns concrete adapters and should not import application,
   bootstrap, CLI, views, engines, or session entrypoints.
5. `bootstrap` owns concrete runtime composition.
6. `engines` own strategy runtime and proposal/account orchestration. They may
   use bootstrap/session APIs where they are the runtime composition surface.
7. `cli` and `views` stay thin: parse input, call application/bootstrap, render
   results.

## Enforced Ratchets

Run these checks after import or structure changes:

```bash
python scripts/check_import_boundaries.py
python scripts/check_service_coupling.py
python scripts/check_service_bootstrap_coupling.py
python scripts/check_composition_layering.py
python scripts/check_composition_complexity.py
python scripts/check_module_size.py
```

What they enforce:

- Clean layer direction.
- No direct service-to-service imports.
- No accidental service dependency on bootstrap/runtime composition.
- No oversized production modules above the 300-line threshold unless explicitly
  documented.
- No composition layer growth that hides business logic in wiring code.

## Stable Facades

Some packages keep stable public facades while implementation is split into
focused modules. This is intentional when tests, CLI code, API code, or operator
workflows depend on historical imports.

Examples:

- `quant_platform.config`
- `quant_platform.engines.engine_runner`
- `quant_platform.engines.session.public_api`
- `quant_platform.bootstrap.engine`
- `quant_platform.bootstrap.engine.loop`
- `quant_platform.infrastructure.postgres.repositories`
- `quant_platform.infrastructure.v2.postgres`
- `quant_platform.application.operator_api.read_models`

When splitting a facade, preserve public exports and add targeted regression
tests around the behavior that moved.

## Current Layer Map

| Layer | Current ownership |
| --- | --- |
| `core/domain` | Instruments, market data, orders, portfolio, production, research, settlement, signals |
| `core/contracts` | Protocols for data, research, portfolio, execution, infrastructure, Redis, production |
| `application` | Data commands, feature governance, operator use cases, API read models, research requests, runtime handles |
| `services` | Data, research, signal, portfolio, execution, governance packages |
| `infrastructure` | Postgres, Redis event bus, V2 repositories, metrics, performance repositories, artifact store |
| `bootstrap` | Broker, data, engine, governance, operator API, persistence, session, signal models, text events |
| `engines` | Engine runner, session cycle, proposal generation, multi-engine merge, V2 account orchestration |
| `views/operator_api` | FastAPI app, middleware, security, route contexts, routers |
| `cli` | Command registry, request factories, presentation |

## Review Checklist

Before accepting a structural change:

- Does the code still pass the import-boundary and service-coupling ratchets?
- Did the change keep entrypoints thin?
- Did concrete adapters remain in infrastructure/bootstrap?
- Did business rules stay in services/application/core instead of CLI/API?
- Did a facade split preserve public imports?
- Did tests cover the behavior that moved?
- Did docs name the current package paths?
