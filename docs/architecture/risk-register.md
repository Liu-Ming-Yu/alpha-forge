# Risk Register

This register tracks material production risks that remain relevant to the
current architecture. Retired risks are kept only when they explain an existing
guard or test.

Severity scale:

- `S1`: could route wrong live orders or materially corrupt money/state.
- `S2`: could block recovery, hide risk, or invalidate promotion evidence.
- `S3`: operational or maintainability risk with limited direct trading impact.

## Active Risks

| ID | Severity | Risk | Current mitigation | Owner area |
| --- | --- | --- | --- | --- |
| R-EXE-03 | S2 | Simulator costs/fills are still simpler than live routing distributions. | Participation-aware fill model, `execution_quality.json`, simulator-calibration command, paper fill comparison before promotion. | `execution_service`, `research_service` |
| R-OBS-03 | S3 | Prometheus metrics are in-process and reset on restart. | Alertmanager rules use `rate()`/`increase()` windows; durable evidence is stored separately when required. | `infrastructure`, `views` |
| R-GOV-03 | S2 | Research-production divergence can still enter through human process. | Shadow/paper/live parity tests, live-session default assertions, review checklist, governed campaign evidence. | `research_service`, `engines` |
| R-GOV-04 | S2 | New operator-controlled state may be introduced in-memory first. | Architecture checklist requires durable storage or restart hydration; kill switch and pacing store are reference implementations. | `application`, `infrastructure` |
| R-DOC-01 | S3 | Stale docs can point operators at old command names or paths. | Docs are now organized around CLI help, current package layout, and runbook index. | `docs`, `cli` |

## Closed Or Reduced Risks

| ID | Status | Closure evidence |
| --- | --- | --- |
| R-DAT-01 | Closed | Redis Streams sweeper, DLQ, stream length and pending metrics. |
| R-DAT-03 | Closed | Feature retention command and repository prune support. |
| R-DAT-04 | Closed baseline | Tiingo/Polygon fallback chain and dataset quorum evidence. |
| R-GOV-01 | Closed | Multi-day parity tests exercise regime transitions. |
| R-GOV-02 | Closed | Model registry CLI supports promote, retire, list, diff, rollback. |
| R-EXE-04 | Reduced | IB historical pacing state mirrors to Redis and hydrates on connect. |
| R-EXE-05 | Closed baseline | V2 orchestrator attaches to paper/live sessions when enabled. |
| R-OBS-06 | Closed | Order submit and fill latency metrics are observed. |
| R-ARCH-01 | Closed baseline | Strict import-boundary mode and service-coupling ratchet pass. |

## Risk Acceptance Rules

- `S1` risks require a blocking fix before live promotion.
- `S2` risks require either a mitigation with evidence or explicit operator
  acceptance for a bounded paper/live experiment.
- `S3` risks can remain on the roadmap if they are visible in docs and ratchets
  prevent further drift.

## Review Prompts

- Can this change route live orders differently?
- Can this change make stale data look fresh?
- Can this change make a research artifact look promotable when it is not?
- Can this change hide broker/account divergence?
- Can this change survive a process restart with incorrect state?
- Does this change weaken a ratchet or create undocumented debt?
