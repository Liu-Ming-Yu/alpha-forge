# LLM Text Feature Promotion Runbook

Use this for text/catalyst features and LLM-backed signal promotion.

## Principle

Text/LLM features can run in shadow mode for evidence collection. Live influence
requires feature cards, feature audits, forecast evidence, text model manifest,
signal gates, production-candidate approval, and a fresh startup assertion.

## What ships in the family

The `text-event-v2` feature family (registered under `family="text"`) emits
**27 features** across three document kinds: news, SEC filings (10-K /
10-Q / 8-K), and earnings-call transcripts. All 27 ship with
`expected_direction="unknown"` and `larger_is_better=False` — they are
evidence-gated by construction.

| Document kind | Prompt version | Feature count | Cadence |
|---|---|---:|---|
| News | `news-prompt-v1` | 10 | Daily |
| SEC filings (10-K / 10-Q / 8-K) | `filing-prompt-v1` | 10 | Sparse on publication date |
| Earnings-call transcripts | `earnings-call-prompt-v1` | 7 | Sparse on publication date |

Routing from a `SourceDocument.kind` to a prompt + extraction schema goes
through `get_prompt_for_kind(source_kind)` and
`extraction_from_dict_for_kind(source_kind, payload)`. Only whitelisted
kinds (`KNOWN_SOURCE_KINDS` in `schemas.py`) are accepted — unknown kinds
fail loudly at dispatch rather than producing wrong-schema panels.

See [`docs/text-event-v2-family.md`](../text-event-v2-family.md) for the
full feature catalog and PIT-safety contract.

## Shadow Collection

```bash
python -m quant_platform text-events --help
python -m quant_platform text-events ingest-news --vendor tws --help
python -m quant_platform run-engine --mode shadow --cycles 1
```

The shadow cycle can extract text features and store feature vectors without
changing order submission.

## Required Evidence

- Text event source and artifact references.
- Text feature card hashes.
- Feature audits for text features.
- Forecast evidence for the target horizon.
- Text model manifest.
- Signal/text gate output.
- Production-candidate output.
- LLM live startup assertion.

## Live Startup Assertion

The assertion proves settings and evidence still match at startup. If it is
missing, expired, or mismatched, live text influence must fail closed.

Check related commands:

```bash
python -m quant_platform text-gate --help
python -m quant_platform signal-gate --help
python -m quant_platform production-candidate --help
```

## Rollback

- Disable live text influence.
- Return source weight to zero or shadow-only mode.
- Preserve artifacts and manifest.
- Re-run readiness and production-candidate checks.

## Red Flags

- Prompt version differs from manifest.
- Feature-card hash mismatch.
- Forecast evidence too old.
- LLM provider key unavailable.
- Live mode enabled without startup assertion.

## Schema versioning

`SCHEMA_VERSION` is `v2`; `LOADABLE_SCHEMA_VERSIONS = {"v1", "v2"}`. The
loader accepts legacy `v1` JSONL by force-routing it through
`NewsExtraction` (v1 only knew about news). Re-emitted JSONL preserves
the original version stamp.

When changing a prompt or extraction schema in a way that alters LLM
output meaning:

1. Bump the corresponding `*_PROMPT_VERSION` (e.g. `news-prompt-v1` →
   `news-prompt-v2`).
2. If the dataclass shape changes, bump `SCHEMA_VERSION` and add the
   old value to `LOADABLE_SCHEMA_VERSIONS` so existing JSONL still
   loads under explicit dispatch rules.
3. Bump the family version (`text-event-v2` → `text-event-v3`).
4. Re-run extraction; do not silently mix versions in the same panel.

Promotion of a feature's `expected_direction` from `"unknown"` to
`"+"`/`"-"` is a family-version bump, not an in-place edit. The
governance gate enforces version match between the manifest and the
running family.
