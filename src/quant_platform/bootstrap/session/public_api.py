"""Session composition API: build a paper, IB-paper, or live ``Session``.

Composition-tier implementation. The symmetric *runtime* API
(``run_strategy_cycle`` and friends) lives in
``engines/session/public_api.py``; ``quant_platform.session`` re-exports both.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.application.runtime.state import Session
from quant_platform.application.runtime.state import (
    SessionDrawdownGuard as _SessionDrawdownGuard,
)
from quant_platform.bootstrap import signal_models as _signal_model_bootstrap
from quant_platform.bootstrap.session.defaults import (
    _DEV_DEFAULT_PREFIXES,
)
from quant_platform.bootstrap.session.defaults import (
    assert_live_session_defaults as _assert_live_session_defaults,
)
from quant_platform.bootstrap.session.defaults import (
    session_default_fields as _session_default_fields,
)
from quant_platform.bootstrap.session.preflight import (
    run_sector_mapping_preflight as _run_sector_mapping_preflight,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import (
        Clock,
        SignalModel,
    )
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.services.portfolio_service.portfolio_constructor import (
        LongOnlyPortfolioConstructor,
        SimpleRegimeDetector,
    )
    from quant_platform.services.signal_service.regime_detector import MarketRegimeDetector

__all__ = [
    "Session",
    "_DEV_DEFAULT_PREFIXES",
    "_SessionDrawdownGuard",
    "_assert_live_session_defaults",
    "_run_sector_mapping_preflight",
    "_session_default_fields",
    "create_ib_paper_session",
    "create_live_session",
    "create_paper_session",
]


def _default_signal_model(settings: PlatformSettings) -> SignalModel:
    """Compatibility shim for the bootstrap signal-model builder."""
    return _signal_model_bootstrap.build_default_signal_model(settings)


def _default_primary_signal_model(settings: PlatformSettings) -> SignalModel:
    """Compatibility shim for the bootstrap primary signal-model builder."""
    return _signal_model_bootstrap.build_default_primary_signal_model(settings)


def _alpha_non_classical_cap(settings: PlatformSettings) -> float:
    return _signal_model_bootstrap.alpha_non_classical_cap(settings)


def _assert_promoted_alpha_sources_configured(settings: PlatformSettings) -> None:
    _signal_model_bootstrap.assert_promoted_alpha_sources_configured(settings)


def create_paper_session(
    settings: PlatformSettings | None = None,
    *,
    initial_cash: Decimal = Decimal("50000"),
    strategy_run_id: uuid.UUID | None = None,
    clock: Clock | None = None,
    signal_model: SignalModel | None = None,
    portfolio_constructor: LongOnlyPortfolioConstructor | None = None,
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None = None,
    regime_detector: SimpleRegimeDetector | MarketRegimeDetector | None = None,
) -> Session:
    """Build a paper-trading session backed by the simulated broker."""
    from quant_platform.bootstrap.session.public_factory import create_paper_session_impl

    return create_paper_session_impl(
        settings=settings,
        initial_cash=initial_cash,
        strategy_run_id=strategy_run_id,
        clock=clock,
        signal_model=signal_model,
        portfolio_constructor=portfolio_constructor,
        instrument_contracts=instrument_contracts,
        regime_detector=regime_detector,
    )


def create_live_session(
    settings: PlatformSettings | None = None,
    *,
    initial_snapshot: AccountSnapshot,
    strategy_run_id: uuid.UUID | None = None,
    signal_model: SignalModel | None = None,
    portfolio_constructor: LongOnlyPortfolioConstructor | None = None,
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None = None,
    regime_detector: SimpleRegimeDetector | MarketRegimeDetector | None = None,
) -> Session:
    """Build a live-trading session backed by the configured broker gateway."""
    from quant_platform.bootstrap.session.public_factory import create_live_session_impl

    return create_live_session_impl(
        settings=settings,
        initial_snapshot=initial_snapshot,
        strategy_run_id=strategy_run_id,
        signal_model=signal_model,
        portfolio_constructor=portfolio_constructor,
        instrument_contracts=instrument_contracts,
        regime_detector=regime_detector,
    )


def create_ib_paper_session(
    settings: PlatformSettings | None = None,
    *,
    initial_snapshot: AccountSnapshot,
    strategy_run_id: uuid.UUID | None = None,
    signal_model: SignalModel | None = None,
    portfolio_constructor: LongOnlyPortfolioConstructor | None = None,
    instrument_contracts: dict[uuid.UUID, dict[str, object]] | None = None,
    regime_detector: SimpleRegimeDetector | MarketRegimeDetector | None = None,
) -> Session:
    """Build a paper-trading session backed by paper TWS/Gateway."""
    from quant_platform.bootstrap.session.public_factory import create_ib_paper_session_impl

    return create_ib_paper_session_impl(
        settings=settings,
        initial_snapshot=initial_snapshot,
        strategy_run_id=strategy_run_id,
        signal_model=signal_model,
        portfolio_constructor=portfolio_constructor,
        instrument_contracts=instrument_contracts,
        regime_detector=regime_detector,
    )


def _maybe_attach_v2_orchestrator(
    session: Session,
    settings: PlatformSettings,
) -> Session:
    """Build and attach AccountExecutionOrchestrator when V2 is enabled."""
    from quant_platform.bootstrap.session.public_factory import maybe_attach_v2_orchestrator

    return maybe_attach_v2_orchestrator(session, settings)
