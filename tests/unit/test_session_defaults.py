"""CI parity guard against dev-stub defaults in live sessions (Phase 5.1).

This is the backstop for R-GOV-03: any new operator-facing plug-in
whose default resolves to a class starting with ``Simple`` or
``InMemory`` must either be wired to a production implementation or
explicitly added to ``_DEV_DEFAULT_ALLOWLIST``.  The test builds a
stub ``Session`` that mimics the live-mode wiring and asserts that
:func:`_assert_live_session_defaults` sees no violations.

Kept hermetic on purpose: instantiating ``create_live_session`` would
drag in ``ibapi`` which is absent from the CI image.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from quant_platform.config import PlatformSettings
from quant_platform.session import (
    _DEV_DEFAULT_PREFIXES,
    _assert_live_session_defaults,
    _session_default_fields,
)


class _ProdEventBus: ...


class _ProdRegimeDetector: ...


class _ProdOrderRepo: ...


class _ProdPositionRepo: ...


class _ProdFeatureRepo: ...


class _ProdTextEventStore: ...


class _ProdPerformanceRepo: ...


class _ProdBarStore: ...


class _ProdAuditSink: ...


class _StubEventBus: ...  # starts with ``_Stub`` but used below to forge


class SimpleEventBus: ...  # a genuine dev-default shape (starts with "Simple")


class InMemoryOrderRepo: ...


@dataclass
class _FakeSession:
    """Shape-compatible with what ``_session_default_fields`` reads."""

    event_bus: object
    regime_detector: object
    order_repo: object
    position_repo: object
    feature_repo: object
    text_event_store: object
    performance_repo: object
    bar_store: object
    audit_sink: object
    settings: PlatformSettings


def _production_session() -> _FakeSession:
    return _FakeSession(
        event_bus=_ProdEventBus(),
        regime_detector=_ProdRegimeDetector(),
        order_repo=_ProdOrderRepo(),
        position_repo=_ProdPositionRepo(),
        feature_repo=_ProdFeatureRepo(),
        text_event_store=_ProdTextEventStore(),
        performance_repo=_ProdPerformanceRepo(),
        bar_store=_ProdBarStore(),
        audit_sink=_ProdAuditSink(),
        settings=PlatformSettings(_env_file=None),
    )


def test_production_wiring_passes_assertion() -> None:
    session = _production_session()
    _assert_live_session_defaults(session)  # type: ignore[arg-type]


def test_simple_stub_in_live_is_rejected() -> None:
    session = _production_session()
    session.event_bus = SimpleEventBus()
    with pytest.raises(RuntimeError, match="dev-stub defaults"):
        _assert_live_session_defaults(session)  # type: ignore[arg-type]


def test_inmemory_stub_in_live_is_rejected() -> None:
    session = _production_session()
    session.order_repo = InMemoryOrderRepo()
    with pytest.raises(RuntimeError, match="dev-stub defaults"):
        _assert_live_session_defaults(session)  # type: ignore[arg-type]


def test_allow_dev_defaults_escape_hatch() -> None:
    session = _production_session()
    session.event_bus = SimpleEventBus()
    session.settings = PlatformSettings(_env_file=None, allow_dev_defaults=True)
    _assert_live_session_defaults(session)  # type: ignore[arg-type]


def test_dev_default_prefixes_are_stable() -> None:
    """Regression lock: changing the prefix list must be deliberate."""
    assert _DEV_DEFAULT_PREFIXES == ("Simple", "InMemory")


def test_session_default_fields_covers_expected_plugins() -> None:
    """Adding a critical plug-in must show up here so the guard covers it."""
    expected = {
        "event_bus",
        "regime_detector",
        "order_repo",
        "position_repo",
        "feature_repo",
        "text_event_store",
        "performance_repo",
        "bar_store",
        "audit_sink",
    }
    assert set(_session_default_fields(_production_session())) == expected  # type: ignore[arg-type]
