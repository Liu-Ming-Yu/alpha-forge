# Observability And Alertmanager Runbook

Use this for Prometheus metrics and Alertmanager rules.

## Principle

Most counters reset when the process restarts. Alerts should use rolling
`rate()` or `increase()` windows for counters and direct threshold checks for
gauges.

## Metrics To Watch

- Order submit latency and outcomes.
- Fill latency.
- Cycle error rate.
- Kill-switch active state.
- Reconciliation discrepancies.
- Event-bus pending entries and dead letters.
- Distributed-lock lease loss.
- Operator API readiness and scrape health.

## Counter Rule Pattern

Use:

```promql
increase(metric_total[10m]) > threshold
```

or:

```promql
rate(metric_total[10m]) > threshold
```

Avoid raw `metric_total > threshold` for process-local counters.

## Gauge Rule Pattern

Use direct thresholds for gauges such as dead-letter depth, pending entries, or
kill-switch active state.

## Restart Handling

Add enough `for:` duration to absorb one missed scrape or brief process restart.
For critical gauges, alert immediately only when the value itself represents a
halt condition.

## Multi-Instance Note

If multiple processes expose metrics, aggregate by logical labels such as
engine, run mode, stream, group, or outcome. Do not assume one process is the
only source unless deployment guarantees it.
