"""Cross-cutting infrastructure primitives.

Adapters and primitives not tied to a single backend cluster: clock,
circuit breaker, distributed lock, Redis connection factory, artifact store,
schema-migration runner, in-memory audit sink and the simulated broker.
"""

from __future__ import annotations
