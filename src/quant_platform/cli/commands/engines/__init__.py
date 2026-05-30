"""Engine command registrations."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from quant_platform.application.operator.requests import RunEngineRequest, RunMultiEngineRequest
from quant_platform.cli.registry import bind_command


def register(sub: Any) -> None:
    eng_p = sub.add_parser(
        "run-engine",
        help="Run Engine 1 (cross-sectional equity) in shadow, paper, or live mode.",
    )
    eng_p.add_argument(
        "--engine",
        choices=["cross_sectional_equity", "etf_macro_allocator", "arm_g", "arm_q"],
        default="cross_sectional_equity",
        help="Engine to run. Default: cross_sectional_equity.",
    )
    eng_p.add_argument(
        "--mode",
        choices=["shadow", "paper", "live"],
        default="shadow",
        help="Execution mode. Default: shadow.",
    )
    eng_p.add_argument(
        "--execution-backend",
        choices=["simulated", "ib-paper"],
        default="simulated",
        help=(
            "Concrete execution backend for paper mode. "
            "Default: simulated. Use ib-paper only with --mode paper."
        ),
    )
    eng_p.add_argument(
        "--initial-cash",
        type=Decimal,
        default=Decimal("50000"),
        help="Starting cash. Default: 50000.",
    )
    eng_p.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of rebalance cycles to execute. Default: 1.",
    )
    eng_p.add_argument(
        "--contracts-file",
        default=None,
        help=(
            "Path to JSON mapping instrument_id UUID -> IB contract spec. "
            "Required when --mode live or --execution-backend ib-paper."
        ),
    )
    bind_command(
        eng_p,
        use_case_name="engine.run",
        request_factory=lambda args: RunEngineRequest(
            mode=args.mode,
            initial_cash=args.initial_cash,
            cycles=args.cycles,
            contracts_file=args.contracts_file,
            engine_name=args.engine,
            execution_backend=args.execution_backend,
        ),
        request_type=RunEngineRequest,
    )

    me_p = sub.add_parser(
        "run-multi-engine",
        help="V2: run multiple engines in proposal mode with shared account orchestration.",
    )
    me_p.add_argument(
        "--engines",
        required=True,
        help="Comma-separated engine names, e.g. cross_sectional_equity,etf_macro_allocator",
    )
    me_p.add_argument(
        "--mode",
        choices=["shadow", "paper", "live"],
        default="paper",
        help="Execution mode. Default: paper.",
    )
    me_p.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of merged cycles. Default: 1.",
    )
    me_p.add_argument(
        "--initial-cash",
        type=Decimal,
        default=Decimal("50000"),
        help="Starting cash for paper sessions. Default: 50000.",
    )
    me_p.add_argument(
        "--contracts-file",
        default=None,
        help="Path to JSON instrument contracts. Required for live mode.",
    )
    me_p.add_argument(
        "--budgets-file",
        required=True,
        help=(
            'Path to JSON budgets file: {"engine_name": {"capital_weight": 0.6, '
            '"max_gross": 0.6, "max_turnover": 0.5}}'
        ),
    )
    bind_command(
        me_p,
        use_case_name="engine.run_multi",
        request_factory=lambda args: RunMultiEngineRequest(
            mode=args.mode,
            engine_names=tuple(e.strip() for e in args.engines.split(",") if e.strip()),
            budgets_file=args.budgets_file,
            cycles=args.cycles,
            initial_cash=args.initial_cash,
            contracts_file=args.contracts_file,
        ),
        request_type=RunMultiEngineRequest,
    )
