"""Shared guards for live IBKR paper-order tests."""

from __future__ import annotations

import os

import pytest

PAPER_PORTS = frozenset({4002, 7497})
TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUTHY


def live_enabled() -> bool:
    return env_flag("QP_LIVE_IBKR_REQUIRED") or env_flag("QP_VERIFY_LIVE_IBKR")


def paper_orders_enabled() -> bool:
    return env_flag("QP_LIVE_IBKR_ALLOW_PAPER_ORDERS")


def skip_unless_paper_orders_enabled() -> None:
    if not live_enabled():
        pytest.skip(
            "live IBKR tests are opt-in; set QP_LIVE_IBKR_REQUIRED=1 or QP_VERIFY_LIVE_IBKR=1"
        )
    if not paper_orders_enabled():
        pytest.skip("paper order tests are opt-in; set QP_LIVE_IBKR_ALLOW_PAPER_ORDERS=1")


def require_env(name: str, *, default: str | None = None) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if default is not None:
        return default
    pytest.fail(f"{name} is required for live IBKR paper order tests")


def require_int_env(name: str) -> int:
    raw = require_env(name)
    try:
        return int(raw)
    except ValueError:
        pytest.fail(f"{name} must be an integer")


def require_ibapi() -> None:
    try:
        __import__("ibapi")
    except Exception as exc:
        pytest.fail(f"ibapi is required for live IBKR paper order tests: {exc}")


def require_paper_order_safety(*, account_id: str, port: int) -> None:
    paper_raw = require_env("QP__BROKER__PAPER_TRADING", default="true").lower()
    if paper_raw not in TRUTHY:
        pytest.fail("paper order tests require QP__BROKER__PAPER_TRADING=true")
    if not account_id.upper().startswith("DU"):
        pytest.fail("paper order tests require an IBKR paper account id beginning with DU")
    if port not in PAPER_PORTS:
        pytest.fail("paper order tests require paper IBKR API port 4002 or 7497")


def order_client_id(*, offset: int = 1) -> int:
    raw_order_client_id = os.environ.get("QP__LIVE_IBKR__ORDER_CLIENT_ID", "").strip()
    if raw_order_client_id:
        try:
            return int(raw_order_client_id)
        except ValueError:
            pytest.fail("QP__LIVE_IBKR__ORDER_CLIENT_ID must be an integer")
    return require_int_env("QP__BROKER__CLIENT_ID") + offset
