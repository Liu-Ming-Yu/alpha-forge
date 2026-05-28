"""Event-bus command registration."""

from __future__ import annotations

from typing import Any

from quant_platform.application.operator.requests import EventBusSweepRequest
from quant_platform.cli.registry import bind_command


def register_event_bus(sub: Any) -> None:
    eb_p = sub.add_parser(
        "event-bus",
        help="Operator utilities for the Redis Streams event bus.",
    )
    eb_sub = eb_p.add_subparsers(dest="event_bus_command", required=True)
    dlq_sweep_p = eb_sub.add_parser(
        "sweep-dead-letters",
        help=(
            "Scan PEL on a stream and move over-retried entries to "
            "<stream>.dlq.  Prints the number moved and the current DLQ depth."
        ),
    )
    dlq_sweep_p.add_argument(
        "--stream",
        required=True,
        help=(
            "Full stream name (with prefix), e.g. 'qp:events:OrderFilled'.  "
            "The DLQ target is always '<stream>.dlq'."
        ),
    )
    bind_command(
        dlq_sweep_p,
        use_case_name="event_bus.sweep_dead_letters",
        request_factory=lambda args: EventBusSweepRequest(stream=args.stream),
        request_type=EventBusSweepRequest,
    )


__all__ = ["register_event_bus"]
