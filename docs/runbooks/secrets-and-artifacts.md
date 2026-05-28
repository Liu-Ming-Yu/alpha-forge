# Secrets And Artifact Hygiene

## Secrets

Inject production secrets through the runtime environment, CI secret store, or
secret manager. Do not commit:

- `.env` with real values.
- Broker account identifiers.
- API keys or tokens.
- Database passwords.
- Private keys.
- Raw operator API keys.

## Scanning

Run:

```bash
python scripts/check_secrets.py
```

The full offline gate runs this check through `make verify`.

## Generated Artifacts

Do not commit:

- `.venv` or `.venv-verify`.
- `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`.
- `.coverage`.
- Generated Parquet data.
- Research artifacts unless they are intentionally tracked fixtures.

Run:

```bash
python scripts/check_generated_artifacts.py --clean
python scripts/check_generated_artifacts.py
```
