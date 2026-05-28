# Redis Streams Event Bus Runbook

Use this when `QP__STORAGE__EVENT_BUS_BACKEND=redis_streams`.

## Configuration

```bash
QP__STORAGE__REDIS_URL=redis://localhost:6379/0
QP__STORAGE__EVENT_BUS_BACKEND=redis_streams
```

## Behavior

Redis Streams provide durable event publication and consumer-group semantics.
The platform tracks stream length, pending entries, publish counts, and dead
letters through metrics.

## Checks

```bash
python -m quant_platform event-bus --help
```

Inspect:

- Stream length.
- Pending entries by group.
- Dead-letter entries.
- Consumer lag.

## Dead Letter Response

1. Inspect payload and consumer failure.
2. Fix consumer or schema issue.
3. Replay only if the consumer is idempotent.
4. Trim or archive only after operator review.

## Risks

- Redis eviction policy can remove state.
- Consumer crash can leave pending entries.
- Schema drift can poison consumers.
- Replay can duplicate effects if handlers are not idempotent.
