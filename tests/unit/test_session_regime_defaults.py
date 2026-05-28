"""Tests for the paper/live regime-detector defaults.

Enforces the parity invariant introduced by R-GOV-03: both paths must
default to ``MarketRegimeDetector`` when ``settings.regime.enabled`` is
True.  Mirrors the live-session assertion and prevents the exact
divergence that the Parity sprint fixed on the backtest side.
"""

from __future__ import annotations

from quant_platform.config import PlatformSettings, RegimeSettings
from quant_platform.services.portfolio_service.portfolio_constructor import (
    SimpleRegimeDetector,
)
from quant_platform.services.signal_service.regime_detector import (
    MarketRegimeDetector,
)
from quant_platform.session import create_paper_session


def _settings(regime_enabled: bool) -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        regime=RegimeSettings(enabled=regime_enabled),
    )


def test_paper_session_defaults_to_market_regime_when_enabled() -> None:
    session = create_paper_session(settings=_settings(True))
    assert isinstance(session.regime_detector, MarketRegimeDetector)


def test_paper_session_falls_back_to_stub_when_disabled() -> None:
    session = create_paper_session(settings=_settings(False))
    assert isinstance(session.regime_detector, SimpleRegimeDetector)


def test_paper_session_honors_explicit_override() -> None:
    override = SimpleRegimeDetector()
    session = create_paper_session(
        settings=_settings(True),
        regime_detector=override,
    )
    assert session.regime_detector is override


def test_paper_session_emits_session_defaults_log(capsys) -> None:
    """The session.defaults log line must be emitted on paper session start.

    ``structlog`` is configured with ``PrintLoggerFactory`` (see
    ``config.configure_logging``), so the output goes to stdout rather
    than through the stdlib ``logging`` module.  That means ``caplog``
    never sees it — capture stdout directly instead.
    """
    create_paper_session(settings=_settings(True))
    captured = capsys.readouterr()
    assert "session.defaults" in captured.out
