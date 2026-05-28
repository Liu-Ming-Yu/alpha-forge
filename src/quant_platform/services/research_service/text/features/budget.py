"""Budget accounting for LLM text feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from quant_platform.services.research_service.text.features.errors import TextFeatureBudgetError


@dataclass
class TextFeatureBudget:
    """Track per-day provider call and estimated cost limits."""

    provider: str
    model: str
    prompt_version: str
    replay_only: bool
    max_daily_calls: int | None = None
    max_daily_estimated_cost_usd: float | None = None
    estimated_cost_per_call_usd: float = 0.0
    budget_day: date = field(default_factory=lambda: datetime.now(UTC).date())
    daily_calls: int = 0
    daily_estimated_cost_usd: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def assert_available(self) -> None:
        """Raise when the next provider call would exceed a daily budget."""
        self._reset_if_needed()
        projected_calls = self.daily_calls + 1
        projected_cost = self.daily_estimated_cost_usd + self.estimated_cost_per_call_usd
        if self.max_daily_calls is not None and projected_calls > self.max_daily_calls:
            raise TextFeatureBudgetError(
                "LLM provider daily call budget exceeded: "
                f"projected_calls={projected_calls} max_daily_calls={self.max_daily_calls}"
            )
        if (
            self.max_daily_estimated_cost_usd is not None
            and projected_cost > self.max_daily_estimated_cost_usd
        ):
            raise TextFeatureBudgetError(
                "LLM provider daily estimated cost budget exceeded: "
                f"projected_cost_usd={projected_cost:.6f} "
                f"max_daily_estimated_cost_usd={self.max_daily_estimated_cost_usd:.6f}"
            )

    def record_provider_attempt(self) -> None:
        """Record one attempted provider call and refresh runtime metadata."""
        self._reset_if_needed()
        self.daily_calls += 1
        self.daily_estimated_cost_usd += self.estimated_cost_per_call_usd
        self.metadata = {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "replay_only": self.replay_only,
            "daily_calls_used": self.daily_calls,
            "max_daily_calls": self.max_daily_calls,
            "estimated_cost_usd": self.estimated_cost_per_call_usd,
            "daily_estimated_cost_usd": self.daily_estimated_cost_usd,
            "max_daily_estimated_cost_usd": self.max_daily_estimated_cost_usd,
            "within_call_budget": (
                self.max_daily_calls is None or self.daily_calls <= self.max_daily_calls
            ),
            "within_cost_budget": (
                self.max_daily_estimated_cost_usd is None
                or self.daily_estimated_cost_usd <= self.max_daily_estimated_cost_usd
            ),
        }

    def _reset_if_needed(self) -> None:
        today = datetime.now(UTC).date()
        if today != self.budget_day:
            self.budget_day = today
            self.daily_calls = 0
            self.daily_estimated_cost_usd = 0.0


__all__ = ["TextFeatureBudget"]
