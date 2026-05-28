# Event-Bus Retention Runbook

Use this for Redis Streams retention, pending entries, and dead-letter handling.

## Configuration

```bash
QP__STORAGE__EVENT_BUS_BACKEND=redis_streams
QP__STORAGE__REDIS_URL=redis://localhost:6379/0
```

## Metrics

Watch:

- `quant_event_bus_publish_total`
- `quant_event_bus_stream_length`
- `quant_event_bus_pending_entries_total`
- `quant_event_bus_dead_letter_total`

Alert on pending entries or dead letters that keep increasing.

## Retention

The Redis Streams sweeper trims streams to the configured retention policy.
Confirm trimming does not remove events before consumers have acknowledged them.

## Dead Letters

When DLQ entries appear:

1. Inspect the event payload and failure reason.
2. Identify the subscriber group.
3. Fix the consumer issue.
4. Replay only after confirming idempotency.
5. Record the replay decision.

## Failure Modes

- Sweeper stopped.
- Consumer crashed and left a pending-entry backlog.
- Event payload is incompatible with the consumer version.
- Redis memory pressure or eviction policy removed state.
