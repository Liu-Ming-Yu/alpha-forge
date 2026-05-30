"""Execution throttle and order-routing settings."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field


class ThrottleSettings(BaseModel):
    """Token-bucket parameters for outbound order pacing."""

    capacity: int = 10
    refill_rate: float = 2.0


class ExecutionSettings(BaseModel):
    """Execution-path controls consumed by ``OrderThrottle``.

    ``trading_hours_enforced`` gates order submission on the US equities
    RTH session.  Default ``False`` so unit tests and the backtest harness
    (both of which replay historical or FakeClock timestamps) do not need
    explicit overrides; production deployments flip it ``True`` via the
    example ``.env`` under ``QP__EXECUTION__TRADING_HOURS_ENFORCED=true``.
    """

    trading_hours_enforced: bool = False
    rebalance_threshold: Decimal = Field(
        default=Decimal("0.01"),
        ge=Decimal("0"),
        description=(
            "Minimum |target_weight - current_weight| for the order planner to emit a "
            "trade (turnover control). Default 1% suits concentrated books; a diffuse "
            "N-name book (e.g. 30 names at ~0.7%/name under a conviction sizer) needs a "
            "lower value (~0.1%) or its low-weight tail is silently skipped. "
            "Set via QP__EXECUTION__REBALANCE_THRESHOLD."
        ),
    )
    passive_limit_enabled: bool = False
    reprice_interval_seconds: int = 300
    max_reprices_per_order: int = Field(
        default=3,
        ge=0,
        description="Maximum passive cancel/replace attempts for one order.",
    )
    min_reprice_improvement_bps: float = Field(
        default=5.0,
        ge=0.0,
        description="Minimum reference-price move before passive repricing is considered.",
    )
    adverse_drift_escalate_bps: float = Field(
        default=25.0,
        ge=0.0,
        description="Adverse move threshold after which passive orders escalate urgency.",
    )
    close_auction_enabled: bool = False
    order_timeout_seconds: int = 1800
    post_submit_lifecycle_drain_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Seconds to keep polling the broker lifecycle feed after order submission "
            "so fills/cancels that arrive just after broker ACKs are included in the "
            "same strategy-cycle result. Set 0 to disable post-submit waiting."
        ),
    )
    lifecycle_drain_poll_seconds: float = Field(
        default=0.1,
        gt=0.0,
        description="Polling interval used during post-submit lifecycle draining.",
    )
    size_urgency_thresholds: list[tuple[int, float]] = Field(
        default_factory=lambda: [(500, 0.25), (2000, 0.50), (10000, 0.75), (50000, 1.0)],
        description=(
            "Ordered list of (max_shares, urgency) breakpoints for size-based urgency. "
            "The urgency for an order is the value whose max_shares is the first threshold "
            "the order size does not exceed.  Orders larger than all thresholds get urgency 1.0. "
            "Configured via QP__EXECUTION__SIZE_URGENCY_THRESHOLDS as a JSON array."
        ),
    )
