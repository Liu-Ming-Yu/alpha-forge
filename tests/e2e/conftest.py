"""Shared fixtures for end-to-end tests (IBGateway + Postgres + Redis)."""

from __future__ import annotations

import os
import uuid

import pytest


def _postgres_dsn() -> str:
    dsn = os.environ.get("QP__STORAGE__POSTGRES_DSN", "")
    if not dsn:
        pytest.skip("QP__STORAGE__POSTGRES_DSN is required for e2e tests")
    return dsn


def _redis_url() -> str:
    url = os.environ.get("QP__STORAGE__REDIS_URL", "")
    if not url:
        pytest.skip("QP__STORAGE__REDIS_URL is required for e2e tests")
    return url


def _live_enabled() -> bool:
    return (
        os.environ.get("QP_LIVE_IBKR_REQUIRED", "").strip() == "1"
        or os.environ.get("QP_VERIFY_LIVE_IBKR", "").strip() == "1"
    )


def _require_env(name: str, *, default: str | None = None) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if default is not None:
        return default
    if _live_enabled():
        pytest.fail(f"{name} is required for e2e tests")
    pytest.skip(f"{name} is not configured")


def _require_int_env(name: str) -> int:
    raw = _require_env(name)
    try:
        return int(raw)
    except ValueError:
        pytest.fail(f"{name} must be an integer")


def _require_ibapi() -> None:
    try:
        __import__("ibapi")
    except Exception as exc:
        if _live_enabled():
            pytest.fail(f"ibapi is required for e2e tests: {exc}")
        pytest.skip(f"ibapi is not installed: {exc}")


@pytest.fixture
def postgres_dsn() -> str:
    return _postgres_dsn()


@pytest.fixture
def redis_url() -> str:
    return _redis_url()


@pytest.fixture
def test_instrument() -> tuple[uuid.UUID, dict]:
    symbol = os.environ.get("QP__LIVE_IBKR__TEST_SYMBOL", "").strip().upper()
    con_id_raw = os.environ.get("QP__LIVE_IBKR__TEST_CON_ID", "").strip()
    if not symbol or not con_id_raw:
        pytest.skip("QP__LIVE_IBKR__TEST_SYMBOL and QP__LIVE_IBKR__TEST_CON_ID required")
    con_id = int(con_id_raw)
    instrument_id = uuid.uuid5(uuid.NAMESPACE_URL, f"ibkr-e2e:{con_id}")
    spec = {
        "symbol": symbol,
        "exchange": os.environ.get("QP__LIVE_IBKR__TEST_EXCHANGE", "SMART"),
        "currency": os.environ.get("QP__LIVE_IBKR__TEST_CURRENCY", "USD"),
        "con_id": con_id,
        "sec_type": "STK",
    }
    return instrument_id, spec
