"""Research backtest command registration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from quant_platform.application.research import (
    BacktestEvidenceAssertRequest,
    BacktestIntradayRequest,
    BacktestRunRequest,
)
from quant_platform.cli.commands.research.request_factories import (
    backtest_evidence_assert_request,
    backtest_intraday_request,
    backtest_run_request,
)
from quant_platform.cli.registry import bind_command


def register_backtest(sub: Any) -> None:
    bt_p = sub.add_parser(
        "backtest",
        help="Backtest runners and fail-closed evidence utilities.",
    )
    bt_sub = bt_p.add_subparsers(dest="backtest_command", required=True)
    run_p = bt_sub.add_parser(
        "run",
        help=(
            "Run a vectorized backtest over stored bar and feature data. "
            "Requires the [backtest] extra: pip install 'quant-platform[backtest]'."
        ),
    )
    run_p.add_argument(
        "--contracts-file",
        required=True,
        help="Path to JSON instrument-contracts file (instrument_id -> contract spec).",
    )
    run_p.add_argument("--start", required=True, type=datetime.fromisoformat)
    run_p.add_argument("--end", required=True, type=datetime.fromisoformat)
    run_p.add_argument("--initial-capital", type=float, default=100_000.0)
    run_p.add_argument("--strategy-name", default="vectorbt_backtest")
    run_p.add_argument("--strategy-version", default="0.1.0")
    run_p.add_argument("--feature-set-version", default="1.0.0")
    run_p.add_argument("--bar-seconds", type=int, default=86400)
    run_p.add_argument("--rebalance-every", type=int, default=1)
    run_p.add_argument("--top-n", type=int, default=10)
    run_p.add_argument("--output-root", type=Path, default=Path("data/backtest"))
    bind_command(
        run_p,
        use_case_name="research.backtest",
        request_factory=backtest_run_request,
        request_type=BacktestRunRequest,
    )

    intraday_p = bt_sub.add_parser(
        "intraday",
        help="Run canonical 1-minute event-driven intraday replay plus vectorized reconciliation.",
    )
    intraday_p.add_argument("--contracts-file", required=True)
    intraday_p.add_argument("--data-file", type=Path)
    intraday_p.add_argument("--vendor", default="file")
    intraday_p.add_argument("--start", required=True, type=datetime.fromisoformat)
    intraday_p.add_argument("--end", required=True, type=datetime.fromisoformat)
    intraday_p.add_argument("--decision-time", action="append", required=True)
    intraday_p.add_argument("--initial-capital", type=float, default=100_000.0)
    intraday_p.add_argument("--strategy-name", default="intraday_backtest")
    intraday_p.add_argument("--strategy-version", default="0.1.0")
    intraday_p.add_argument("--feature-set-version", default="1.0.0")
    intraday_p.add_argument("--model-version", default="classical")
    intraday_p.add_argument("--universe-name", default="intraday_research")
    intraday_p.add_argument("--dataset-id", action="append", default=[])
    intraday_p.add_argument("--output-root", type=Path, default=Path("data/backtest"))
    bind_command(
        intraday_p,
        use_case_name="research.backtest",
        request_factory=backtest_intraday_request,
        request_type=BacktestIntradayRequest,
    )

    evidence_p = bt_sub.add_parser(
        "evidence",
        help="Assert a backtest evidence manifest is promotion-safe.",
    )
    evidence_sub = evidence_p.add_subparsers(dest="backtest_evidence_command", required=True)
    assert_p = evidence_sub.add_parser("assert", help="Fail closed on missing or failed evidence.")
    assert_p.add_argument("--manifest", required=True, type=Path)
    bind_command(
        assert_p,
        use_case_name="research.backtest",
        request_factory=backtest_evidence_assert_request,
        request_type=BacktestEvidenceAssertRequest,
    )


__all__ = ["register_backtest"]
