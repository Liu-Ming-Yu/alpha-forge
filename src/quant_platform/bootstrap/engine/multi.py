"""V2 multi-engine runtime composition helpers."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema
from quant_platform.engines.market_data.price_seeding import latest_contract_market_prices

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import PlatformSettings
    from quant_platform.engines.engine_runner import EngineRunner

log = structlog.get_logger(__name__)


def create_single_engine_runner(
    *,
    settings: PlatformSettings,
    engine_name: str,
    mode: str,
    execution_backend: str,
    initial_cash: Decimal,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
) -> EngineRunner:
    """Create one plugin-backed engine runner from operator strings."""
    from quant_platform.engines.engine_runner import ExecutionBackend, RunMode
    from quant_platform.engines.framework.plugins import create_engine_from_plugin

    return create_engine_from_plugin(
        engine_name,
        run_mode=RunMode(mode),
        initial_cash=initial_cash,
        settings=settings,
        instrument_contracts=instrument_contracts,
        execution_backend=ExecutionBackend(execution_backend),
    )


async def run_multi_engine_v2(
    settings: PlatformSettings,
    *,
    mode: str,
    engine_names: list[str],
    budgets_file: str | None = None,
    cycles: int = 1,
    initial_cash: Decimal = Decimal("50000"),
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None = None,
) -> None:
    """Run bounded V2 account-orchestrator cycles over one shared runner.

    Single-engine and multi-engine differ only by ``engine_names``/``budgets``;
    the per-cycle execution lives in ``AccountOrchestratorLoopRunner`` so this
    bounded path and ``supervise`` (via ``run_engine_loop``) never diverge.
    """
    from quant_platform.bootstrap.engine.orchestrator_runner import AccountOrchestratorLoopRunner
    from quant_platform.engines.engine_runner import RunMode

    if not (settings.v2.enabled and settings.v2.account_orchestrator_enabled):
        raise RuntimeError(
            "run-multi-engine requires QP__V2__ENABLED=true and "
            "QP__V2__ACCOUNT_ORCHESTRATOR_ENABLED=true"
        )

    await verify_postgres_schema(settings)

    budgets = resolve_engine_budgets(
        budgets_file=budgets_file, engine_names=engine_names, mode=mode
    )
    contracts = dict(instrument_contracts or {})
    if RunMode(mode) == RunMode.LIVE and not contracts:
        raise ValueError("LIVE mode requires --contracts-file")

    runner = AccountOrchestratorLoopRunner(
        settings,
        engine_names=engine_names,
        budgets=budgets,
        mode=mode,
        initial_cash=initial_cash,
        instrument_contracts=contracts,
    )
    await runner.initialize()
    try:
        for cycle_i in range(cycles):
            result = await runner.run_cycle(feature_data={})
            log.info(
                "multi_engine.cycle",
                cycle=cycle_i + 1,
                submitted=len(result.submitted_ids),
                approved=len(result.approved),
                rejected=len(result.rejected),
            )
    finally:
        await runner.shutdown()


def _canonical_engine_name(name: str) -> str:
    """Resolve a plugin CLI key (e.g. ``cross_sectional_equity``) to the canonical
    engine name carried by proposals (e.g. ``cross_sectional_equity_v1``).

    Budgets must be keyed by the canonical name so the proposal/budget merge
    lookup lines up; falls back to ``name`` for non-plugin engines.
    """
    from quant_platform.engines.framework.plugins import get_strategy_plugin

    try:
        return get_strategy_plugin(name).name
    except ValueError:
        return name


def resolve_engine_budgets(
    *,
    budgets_file: str | None,
    engine_names: list[str],
    mode: str,
) -> tuple[Any, ...]:
    if budgets_file is not None:
        return load_budgets(budgets_file, engine_names)
    if len(engine_names) != 1:
        raise ValueError(
            "auto-budget (budgets_file=None) requires exactly one engine; "
            f"got {engine_names}. Provide --budgets-file for multi-engine runs."
        )
    from quant_platform.core.domain.production import EngineBudget

    return (
        EngineBudget(
            engine_name=_canonical_engine_name(engine_names[0]),
            engine_version="0.1.0",
            run_mode=mode,
            capital_weight=Decimal("1.0"),
            max_gross=Decimal("1.0"),
            max_turnover=Decimal("1.0"),
            enabled=True,
        ),
    )


def load_budgets(
    budgets_file: str,
    engine_names: list[str],
) -> tuple[Any, ...]:
    """Parse a JSON budgets file into a tuple of EngineBudget objects."""
    from quant_platform.core.domain.production import EngineBudget

    with Path(budgets_file).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    budgets = []
    for name in engine_names:
        spec = raw.get(name)
        if spec is None:
            raise ValueError(f"No budget entry for engine '{name}' in {budgets_file}")
        budgets.append(
            EngineBudget(
                engine_name=_canonical_engine_name(name),
                engine_version=spec.get("engine_version", "0.1.0"),
                capital_weight=Decimal(str(spec["capital_weight"])),
                max_gross=Decimal(str(spec["max_gross"])),
                max_turnover=Decimal(str(spec["max_turnover"])),
                enabled=spec.get("enabled", True),
                run_mode=spec.get("run_mode", "live"),
            )
        )
    return tuple(budgets)


__all__ = [
    "create_single_engine_runner",
    "latest_contract_market_prices",
    "load_budgets",
    "run_multi_engine_v2",
]
