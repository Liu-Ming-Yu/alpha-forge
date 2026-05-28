# Secrets rotation runbook

**Scope:** the secrets the paper-trading stack actually depends on.
Live trading secrets (account-funded IB credentials) are out of scope
until the live preflight gate passes.

**Cadence:** rotate annually or on any of the trigger events below.

---

## 1. Inventory

All secrets live in the gitignored `.env` at the repo root (loaded by
pydantic-settings via the `QP__SECTION__FIELD` env-var prefix). Nothing
secret should ever appear in `infra/config/settings.example.env` —
that file is the *shape* of `.env`, with placeholders only.

| Env var                                  | What it is                                  | Rotation trigger                         |
|------------------------------------------|---------------------------------------------|------------------------------------------|
| `POSTGRES_PASSWORD`                      | Docker-compose Postgres password            | Annually; on any operator change         |
| `DEEPSEEK_API_KEY`                       | LLM provider key                            | Annually; on exposure; on budget reset   |
| `QP__DATA_INGEST__TIINGO_API_TOKEN`      | Bar / EOD vendor token                      | Annually; on vendor rotation request     |
| `QP__DATA_INGEST__POLYGON_API_KEY`       | Reference data + secondary bar source key   | Annually                                 |
| `QP__API__OPERATOR_API_KEY`              | Operator-facing HTTP key for `/health/ready`| Quarterly; on any operator change        |
| `QP__BROKER__ACCOUNT_ID`                 | IBKR paper account number (not strictly a secret, but config-coupled to the key file) | When TWS / paper account is reset |

> **Not in `.env`**: TWS/Gateway client credentials live inside the IBKR
> desktop application's encrypted store; we never write them to disk
> from this repo.

---

## 2. Detection — what catches an accidental commit

The repo runs `detect-secrets` in CI. The baseline lives at
`.secrets.baseline`. Any new high-entropy or known-pattern secret in a
tracked file fails the pipeline. If you add a *legitimately* synthetic
key for a test fixture, suffix the line with `# pragma: allowlist
secret` and add a comment explaining why it's safe.

Local pre-commit:

```bash
.venv/bin/detect-secrets-hook --baseline .secrets.baseline <changed files>
```

---

## 3. Rotation procedure (general shape)

1. **Generate the new secret** in the provider UI (DeepSeek dashboard,
   Tiingo account page, etc.). Note the issued timestamp.
2. **Update `.env`** on every host that runs the stack — operator dev
   box, paper VPS if/when separated.
3. **Reload the stack** so pydantic-settings picks up the new value:
   ```bash
   docker compose restart quant-platform-api quant-platform-paper-engine
   # If running directly (systemd / PowerShell wrapper):
   systemctl restart quant-paper       # Linux
   # or stop the PS loop and let it relaunch.
   ```
4. **Verify the new key works** before revoking the old one:
   ```bash
   python -m quant_platform runtime smoke    # exits 0 on success
   python -m quant_platform runtime run-cycle --once --execution-backend ib-paper
   # For DeepSeek specifically:
   python -c "from quant_platform.bootstrap import build_settings; \
     s = build_settings(); \
     print('llm.provider=', s.llm.provider, 'model=', s.llm.model)"
   ```
5. **Revoke the old key** in the provider UI.
6. **Record the rotation** in `docs/runbooks/secrets-rotation-log.md`
   (create on first rotation): secret name, new key fingerprint
   (first 4 + last 4 chars), date, who performed it.

---

## 4. Per-secret notes

### `POSTGRES_PASSWORD`

The Docker Compose `depends_on` health check uses `pg_isready -d
quant_platform`, which doesn't authenticate; rotating the password
**won't be caught by compose health alone**. After rotation, run:

```bash
psql "$QP__STORAGE__POSTGRES_DSN" -c '\dt' >/dev/null && echo ok
```

If you wipe the Postgres volume *and* rotate the password, both the
new password (in `.env` for the `postgres` service) and the DSN
substitution must match — the docker-compose env block builds the DSN
from `POSTGRES_PASSWORD`, so just `.env` is enough.

### `DEEPSEEK_API_KEY`

LLM extraction is content-addressable cached in `data/parquet/research/
text_events/extractions/`. Rotating the key does not invalidate the
cache. Make sure `QP__LLM__SHADOW_MODE_ENABLED=true` and
`QP__LLM__LIVE_MODE_ENABLED=false` before testing — the live gate will
fail-closed if the key is missing.

### `QP__DATA_INGEST__TIINGO_API_TOKEN` / `QP__DATA_INGEST__POLYGON_API_KEY`

Used by `data ingest --data-source {tiingo|polygon}`. Verify after
rotation with a one-symbol smoke:

```bash
python -m quant_platform data ingest \
  --data-source tiingo \
  --contracts-file infra/config/universe_300.json \
  --instrument-symbol AAPL --start 2026-04-01 --end 2026-04-05
```

### `QP__API__OPERATOR_API_KEY`

The `quant-platform-api` health check passes this as
`X-API-Key`. After rotation, hit:

```bash
curl -sf -H "X-API-Key: $QP__API__OPERATOR_API_KEY" \
  http://localhost:8000/health/ready
```

---

## 5. Trigger events (rotate immediately, do not wait for annual)

- Secret pasted into chat / shared screen / public PR / public log.
- `detect-secrets` flags a real key in a tracked file (rotate even
  after force-pushing the removal — assume it was scraped).
- Operator change (someone leaves the project).
- Provider notifies of a breach.

Recovery after an exposure:

1. Revoke the exposed key first (do not wait for replacement).
2. Generate the new key, update `.env`, restart the stack.
3. Audit usage logs on the provider dashboard for unfamiliar IPs /
   request spikes since the suspected exposure window.
4. Record the incident + audit findings in the rotation log.

---

## 6. Reconstructing `.env` from scratch

If `.env` is lost (e.g. fresh clone on a new operator box), the
template lives at `infra/config/settings.example.env`. Fill it in by
fetching each secret from the provider dashboards listed above; there
is no out-of-band copy in the repo or backups by design.

The `scripts/backup_durable.py` snapshots intentionally *exclude*
`.env`, so a full restore from backups still requires the operator to
re-paste credentials. This is a deliberate cost of not coupling
secret storage to artifact backup.
