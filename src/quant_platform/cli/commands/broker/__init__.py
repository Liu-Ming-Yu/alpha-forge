"""Broker command registrations."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from quant_platform.application.operator.requests import (
    BrokerContractsRequest,
    PaperLifecycleRequest,
    PassiveRepriceRequest,
)
from quant_platform.cli.registry import bind_command

MAX_PAPER_LIFECYCLE_NOTIONAL = Decimal("100")


def register(sub: Any) -> None:
    ib_p = sub.add_parser(
        "ib-gateway-smoke",
        help="Run read-only account, position, health, and open-order checks against IB Gateway.",
    )
    ib_p.add_argument("--contracts-file", required=True)
    bind_command(
        ib_p,
        use_case_name="broker.ib_gateway_smoke",
        request_factory=lambda args: BrokerContractsRequest(contracts_file=args.contracts_file),
        request_type=BrokerContractsRequest,
    )

    life_p = sub.add_parser(
        "ib-paper-lifecycle",
        help="Run a paper-only non-marketable submit/open/cancel lifecycle check.",
    )
    life_p.add_argument("--contracts-file", required=True)
    life_p.add_argument("--instrument-id", required=True, type=uuid.UUID)
    life_p.add_argument(
        "--max-notional-usd",
        type=Decimal,
        default=MAX_PAPER_LIFECYCLE_NOTIONAL,
    )
    bind_command(
        life_p,
        use_case_name="broker.ib_paper_lifecycle",
        request_factory=lambda args: PaperLifecycleRequest(
            contracts_file=args.contracts_file,
            instrument_id=args.instrument_id,
            max_notional_usd=args.max_notional_usd,
        ),
        request_type=PaperLifecycleRequest,
    )

    reprice_p = sub.add_parser(
        "passive-reprice-once",
        help="Run one conservative passive-limit cancel pass and print decisions.",
    )
    reprice_p.add_argument("--mode", choices=["paper", "live"], default="paper")
    reprice_p.add_argument("--contracts-file", required=True)
    reprice_p.add_argument("--initial-cash", type=Decimal, default=Decimal("50000"))
    bind_command(
        reprice_p,
        use_case_name="broker.passive_reprice_once",
        request_factory=lambda args: PassiveRepriceRequest(
            mode=args.mode,
            contracts_file=args.contracts_file,
            initial_cash=args.initial_cash,
        ),
        request_type=PassiveRepriceRequest,
    )
