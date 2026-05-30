"""Runtime and service command registrations."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from quant_platform.application.operator.requests import (
    NoInputRequest,
    RunCycleRequest,
    ServeApiRequest,
    SuperviseRequest,
)
from quant_platform.cli.registry import bind_command


def register(sub: Any) -> None:
    run_p = sub.add_parser("run-cycle", help="Execute a single rebalance cycle.")
    run_p.add_argument(
        "--initial-cash",
        type=Decimal,
        default=Decimal("50000"),
        help="Starting settled cash (paper mode). Default: 50000.",
    )
    bind_command(
        run_p,
        use_case_name="runtime.run_cycle",
        request_factory=lambda args: RunCycleRequest(initial_cash=args.initial_cash),
        request_type=RunCycleRequest,
    )

    sup_p = sub.add_parser("supervise", help="Loop rebalance cycles with a sleep interval.")
    sup_p.add_argument(
        "--initial-cash",
        type=Decimal,
        default=Decimal("50000"),
        help="Starting settled cash (paper mode). Default: 50000.",
    )
    sup_p.add_argument(
        "--interval",
        type=float,
        default=300.0,
        help="Seconds between cycles. Default: 300.",
    )
    sup_p.add_argument(
        "--engine",
        choices=["cross_sectional_equity", "etf_macro_allocator", "arm_g", "arm_q"],
        default="cross_sectional_equity",
        help="Engine to supervise. Default: cross_sectional_equity.",
    )
    sup_p.add_argument(
        "--mode",
        choices=["shadow", "paper"],
        default="paper",
        help="Supervised execution mode. Default: paper.",
    )
    sup_p.add_argument(
        "--execution-backend",
        choices=["simulated", "ib-paper"],
        default="simulated",
        help=(
            "Concrete execution backend for paper mode. "
            "Default: simulated. Use ib-paper only with --mode paper."
        ),
    )
    sup_p.add_argument(
        "--contracts-file",
        default=None,
        help=(
            "Path to JSON mapping instrument_id UUID -> IB contract spec. "
            "Required when --execution-backend ib-paper."
        ),
    )
    sup_p.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Diagnostic cap for supervised cycles. Omit for continuous supervision.",
    )
    bind_command(
        sup_p,
        use_case_name="runtime.supervise",
        request_factory=lambda args: SuperviseRequest(
            initial_cash=args.initial_cash,
            interval_seconds=args.interval,
            mode=args.mode,
            max_cycles=args.max_cycles,
            contracts_file=args.contracts_file,
            engine_name=args.engine,
            execution_backend=args.execution_backend,
        ),
        request_type=SuperviseRequest,
    )

    health_p = sub.add_parser("health", help="Check broker connectivity and session status.")
    bind_command(
        health_p,
        use_case_name="runtime.health",
        request_factory=lambda _args: NoInputRequest(),
        request_type=NoInputRequest,
    )


def register_api(sub: Any) -> None:
    api_p = sub.add_parser(
        "serve-api",
        help="Start the read-only operator HTTP API (requires pip install -e '.[api]').",
    )
    api_p.add_argument(
        "--initial-cash",
        type=Decimal,
        default=Decimal("50000"),
        help="Starting settled cash for the read-model projection. Default: 50000.",
    )
    api_p.add_argument("--host", default="127.0.0.1", help="Bind address. Default: 127.0.0.1.")
    api_p.add_argument("--port", type=int, default=8000, help="Port. Default: 8000.")
    bind_command(
        api_p,
        use_case_name="runtime.serve_api",
        request_factory=lambda args: ServeApiRequest(
            initial_cash=args.initial_cash,
            host=args.host,
            port=args.port,
        ),
        request_type=ServeApiRequest,
    )


def register_smoke(sub: Any) -> None:
    smoke_p = sub.add_parser(
        "smoke",
        help="Post-deployment smoke test: verify Postgres, Redis, broker, and preflight all pass.",
    )
    bind_command(
        smoke_p,
        use_case_name="runtime.smoke",
        request_factory=lambda _args: NoInputRequest(),
        request_type=NoInputRequest,
    )
