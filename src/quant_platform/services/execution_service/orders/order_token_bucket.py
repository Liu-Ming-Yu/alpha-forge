"""Token-bucket math for order throttling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class OrderTokenBucket:
    """Mutable token bucket whose read path can be side-effect free."""

    capacity: int
    refill_rate: float
    tokens: float
    last_refill: datetime

    @classmethod
    def full(cls, *, capacity: int, refill_rate: float, now: datetime) -> OrderTokenBucket:
        return cls(
            capacity=capacity,
            refill_rate=refill_rate,
            tokens=float(capacity),
            last_refill=now,
        )

    def peek(self, now: datetime) -> float:
        """Return token count at ``now`` without mutating the bucket."""
        elapsed = (now - self.last_refill).total_seconds()
        return min(float(self.capacity), self.tokens + elapsed * self.refill_rate)

    def consume(self, now: datetime, amount: float = 1.0) -> None:
        """Refill to ``now`` and consume ``amount`` tokens."""
        self.tokens = self.peek(now)
        self.last_refill = now
        self.tokens = max(0.0, self.tokens - amount)
