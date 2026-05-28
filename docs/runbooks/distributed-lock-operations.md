# Distributed-Lock Operations Runbook

Use this for Redis-backed strategy-cycle lock issues.

## Purpose

The distributed lock prevents concurrent strategy cycles from acting on the same
account/session state. If a lease is lost while a cycle is running, the cycle
must halt instead of continuing with uncertain ownership.

## Configuration

```bash
QP__STORAGE__REDIS_URL=redis://localhost:6379/0
```

The in-memory path is acceptable for local paper tests, not durable production
readiness.

## Metrics

Watch:

- Lock acquire attempts and failures.
- Lease renewal failures.
- Lease-loss events.
- Cycle errors after lease loss.

## Response: Lease Lost

1. Confirm only one engine process is running.
2. Keep or activate the kill switch.
3. Inspect Redis health and eviction policy.
4. Inspect platform logs for lock owner/run identifiers.
5. Reconcile account and open orders.
6. Resume only after ownership is unambiguous.

## Manual Lock Cleanup

Delete a stale lock only after confirming the owning process is stopped and no
cycle is running. Record the key, owner, and operator decision.
