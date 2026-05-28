# Operator API Key Lifecycle

The operator API is default-secure. Protected endpoints require an API key
unless unauthenticated mode is explicitly enabled and acknowledged.

## Current Key

Generate:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Configure:

```bash
QP__API__OPERATOR_API_KEY=<generated key>
```

Requests can use:

```bash
curl -H "X-API-Key: <key>" http://127.0.0.1:8000/health/ready
curl -H "Authorization: Bearer <key>" http://127.0.0.1:8000/dashboard/summary
```

## Unauthenticated Escape Hatch

Only use for isolated local development:

```bash
QP__API__ALLOW_UNAUTHENTICATED=true
QP__API__ACKNOWLEDGE_UNAUTHENTICATED_RISK=true
```

Do not expose this mode to a network.

## Rotation

1. Generate a new key.
2. Update runtime secret/config.
3. Restart the API or reload config according to deployment.
4. Verify new key works.
5. Confirm old key no longer works.

## V2 Key State

V2 operator API-key records can be stored as hashed durable state. Key
revocation should set `revoked_at` and be auditable without exposing raw keys.
