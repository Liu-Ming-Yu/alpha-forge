# Platform conventions

> This is the codebase's "design system" — there is no UI, but the platform has
> strong conventions across CLI commands, config env vars, artifact paths,
> structured-log event names, status/heartbeat JSON, and script vs library
> separation. Any new operational tooling should follow these patterns.
> Inconsistencies = forks of an existing convention.

## A. CLI command structure

**Pattern:** top-level domain, optional subcommand, kebab-case throughout.

| Example | Shape |
|---|---|
| `text-events ingest-sec`, `text-events extract-features` | `<domain> <verb-noun>` |
| `features backfill`, `features build-samples`, `features audit` | `<domain> <verb>` |
| `research-campaign run` | `<domain>-<scope> <verb>` |
| `signal-gate assert / record / status` | `<gate> <verb>` |
| `runtime supervise`, `runtime smoke`, `runtime run-cycle` | `<runtime> <verb>` |
| `migrate`, `verify-schema`, `preflight` | bare top-level verbs for one-shot ops |

**Rules**
- Subcommands belong on the same domain group, never bare under root.
- Long flags use kebab-case (`--feature-set-version`, `--data-source`).
- Flag names mirror the request dataclass field names (snake_case→kebab-case
  is automatic via argparse).

## B. Config env-var prefixing

**Pattern:** `QP__<SECTION>__<FIELD>` (double-underscore separators).

| Example | Maps to |
|---|---|
| `QP__BROKER__HOST` | `settings.broker.host` |
| `QP__STORAGE__POSTGRES_DSN` | `settings.storage.postgres_dsn` |
| `QP__LLM__MAX_TOKENS` | `settings.llm.max_tokens` |
| `QP__DATA_INGEST__POLYGON_API_KEY` | `settings.data_ingest.polygon_api_key` |
| `QP__FACTORS__FITTED_WEIGHTS_MANIFEST` | `settings.factors.fitted_weights_manifest` |

**Rules**
- New settings live on a pydantic `*Settings` model nested under
  `PlatformSettings`. The env var follows from the nesting path automatically.
- Document every new field in `infra/config/settings.example.env`.
- Sensitive values (API keys, passwords) live only in `.env` (gitignored).
  See `docs/runbooks/secrets-rotation.md`.

## C. Artifact paths under `data/parquet/`

```
data/parquet/
├── bars/<instrument_uuid>/<year>.parquet        # OHLCV bar store
├── backtests/<run-uuid>.parquet                 # legacy synthetic backtests (ignore)
├── tearsheets/<run-uuid>/...                    # legacy synthetic tearsheets (ignore)
├── paper_soak/...                               # paper-trading evidence
└── research/
    ├── _inputs/                                  # samples and intermediate inputs
    ├── feature_audits/<feature>/<feature_set_version>/<audit-uuid>/...
    ├── walk_forward/<run-uuid>/...               # eligibility, fold metrics, manifest
    ├── walk_forward/_blocked/<slug>/...          # blocked campaign summaries
    ├── walk_forward/models/xgboost/<run-id>/...  # XGBoost search outputs
    └── text_events/
        ├── sec_filings/sec/<content-hash>.txt    # raw SEC filing bodies
        ├── extractions/<provider>/<model>/<prompt_version>/<event_id>_<hash>.json
        ├── sec_<start>_<end>/source_data_manifest.json
        ├── extract_<start>_<end>/                # per-extraction-window outputs
        └── _blocked/<slug>/...                   # blocked extraction summaries
```

**Rules**
- Library code writes to these paths only; never to `infra/config/`,
  `src/`, or `tests/`.
- Hashed leaf filenames for content-addressable artifacts.
- `data/parquet/**` is git-ignored; backups live under `backups/`
  (also git-ignored).

## D. Structured log event names

**Pattern:** snake_case dotted, `<subsystem>.<event>`, attributes on key=value.

| Example | Emitter |
|---|---|
| `session.storage_backend` | session factory |
| `text_event_store.backend` | text event repo |
| `text_extractor.extracted` | LLM extractor |
| `text_extractor.failure` | LLM extractor (added in PR #26) |
| `text_extractor.cache_artifact_hit` | replay cache |
| `text_extractor.truncated` | extractor input truncation |
| `text_extractor.timeout` | per-call timeout |
| `ingest.complete` | bar ingest |
| `daily_ingest.complete` | bar daily ingest |
| `migrate.complete` | alembic migrate |
| `broker_gateway.disconnected` | IB gateway |
| `loop.run_started` / `loop.run_completed` | engine loop |

**Rules**
- New events emit via `structlog.get_logger(__name__).info("subsys.event", **attrs)`.
- Subsystem prefix matches the module (`text_extractor`, `ingest`, etc.).
- Attributes are JSON-safe primitives (no objects unless serializable).
- `level=warning` for recoverable failures (the audit pattern). `level=error`
  for unrecoverable.

## E. Status / heartbeat JSON shape

Introduced by `scripts/extract_status.py` for the LLM extraction. Reuse this
shape for any new long-running operational task.

```json
{
  "started_at": "ISO-8601 UTC",
  "updated_at": "ISO-8601 UTC",
  "elapsed_seconds": 0.0,
  "total_events": 0,         // or total_items / total_steps for non-event jobs
  "extracted": 0,            // or processed
  "skipped_<reason>": 0,     // explicit per-reason counts
  "failed": 0,
  "in_flight": 0,
  "rate_per_minute": 0.0,
  "eta_seconds": 0.0,
  "terminal": false          // true on final flush
}
```

**Rules**
- Written atomically (`.tmp` + rename).
- Heartbeat interval ≤30s so a stale file is a real signal.
- Companion reader script under `scripts/<job>_status.py` (mirror of
  `extract_status.py`) — reuses the `_human_seconds`, `_bar`, `STUCK?`
  detection idioms.

## F. Script vs library separation

**Library** (`src/quant_platform/...`) — anything imported, tested, gated by
import-boundary ratchets, module-size ratchet, mypy strict zones.

**Scripts** (`scripts/...`) — operator tooling: universe builders, status
readers, one-shot ad-hoc diagnostics, backup/restore. Allowed to use stdlib
freely. Loose on type strictness. Never imported by library code.

**Rules**
- Scripts may import library code, never the other way around.
- Script docstring header documents `Usage::` and prints `--help`.
- Scripts emit human-readable text to stdout; structured logs go through
  `structlog` only when the script reuses library code that logs.
- Long-running scripts use the heartbeat pattern from §E.

## G. Backup / artifact preservation

Introduced ad hoc during the LLM extraction backfill; codify here.

```
backups/<job-slug>_<UTC-timestamp>/
├── manifest.json                  # counts, slug, optional checksums
├── <tar.gz of irrecoverable on-disk artifacts>
└── <postgres dump(s).sql.gz or .csv.gz>
```

Per `scripts/backup_durable.py`. Backups are git-ignored.

## H. Dual version stamps on artifact-backed families

**Pattern:** any feature family that consumes a frozen on-disk artifact
carries **two independent version strings** on the artifact payload:

| Field | Purpose | Bumped when |
|---|---|---|
| `artifact_version` | Storage-schema version of the artifact dataclass itself (field names, shapes, JSON layout). | The artifact dataclass adds/removes/renames a field, or changes a shape contract. |
| `family_version` | The feature-family-version string the artifact was trained for, e.g. `"learned-representations-v1"`. | The family's catalogue/version changes, even if the on-disk schema is unchanged. |

**Why both:** `artifact_version` lets the loader reject incompatible
payloads at deserialisation time (wrong shape). `family_version` lets the
compute path reject artifacts trained for a different family generation
(right shape, wrong target). The two are orthogonal — an operator may
retrain v1 artifacts many times under the same `artifact_version`, and a
single artifact-schema bump may cover multiple family versions.

**Rules**
- Persist both fields on every artifact (frozen dataclass + JSON round-trip).
- The compute path asserts `artifact.family_version == config.version`
  before applying any transform.
- The loader asserts `payload["artifact_version"] == <module constant>`
  before deserialising.
- Bump `artifact_version` (a module-level constant like
  `ARTIFACT_SCHEMA_VERSION`) only when the dataclass changes. Bump
  `family_version` independently when the family bumps.

**Reference implementation:** `learned-representations-v1`
(`src/quant_platform/research/features/learned/artifact.py`,
constant `ARTIFACT_SCHEMA_VERSION = "pca-artifact-v2"` — bumped from
`v1` on 2026-05-26 when the dataclass added a per-feature `scale`
field, see PR #61 / branch `fix-learned-pca-standardisation`).

## Quick audit (Track 1 self-check)

| Axis | Status before Track 1 | After Track 1 |
|---|---|---|
| CLI naming | consistent ✅ | unchanged ✅ |
| Config env vars | consistent ✅ | LLM defaults migrate into `LLMSettings` defaults, env vars unchanged ✅ |
| Artifact paths | consistent ✅ | backups under `backups/<slug>_<ts>/` — new but follows pattern ✅ |
| Structured-log events | consistent (PR #26 added `text_extractor.failure`) ✅ | new `alert.*`, `backup.*` events follow `<subsys>.<event>` ✅ |
| Status JSON shape | one instance (`extract_status.json`) ✅ | reused by supervise watchdog status — same shape ✅ |
| Script vs library | clean separation ✅ | new scripts follow pattern ✅ |

**No new conventions introduced.** Track 1 either reinforces or extends
existing patterns; none are forked.
