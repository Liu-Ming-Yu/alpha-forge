"""Integration tests for PostgresKillSwitchStore against a real Postgres database."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from quant_platform.infrastructure.postgres.repositories import create_pg_engine
from quant_platform.services.execution_service.stores.kill_switch_store import (
    PostgresKillSwitchStore,
)

pytestmark = pytest.mark.integration_durable

_UTC = UTC


def _postgres_dsn() -> str:
    dsn = os.environ.get("QP__STORAGE__POSTGRES_DSN", "")
    if not dsn:
        pytest.skip("QP__STORAGE__POSTGRES_DSN is required for durable tests")
    return dsn


def _store() -> PostgresKillSwitchStore:
    return PostgresKillSwitchStore(create_pg_engine(_postgres_dsn()))


@pytest.mark.asyncio
async def test_activate_and_get() -> None:
    store = _store()
    now = datetime.now(tz=_UTC)

    await store.activate(reason="test_activate_and_get", activated_by="pytest", as_of=now)
    state = await store.get()

    assert state.active is True
    assert state.reason == "test_activate_and_get"
    assert state.activated_by == "pytest"

    await store.clear(operator_id="cleanup", as_of=datetime.now(tz=_UTC))


@pytest.mark.asyncio
async def test_clear_deactivates_kill_switch() -> None:
    store = _store()
    now = datetime.now(tz=_UTC)

    await store.activate(reason="test_clear", activated_by="pytest", as_of=now)
    await store.clear(operator_id="alice", as_of=datetime.now(tz=_UTC))
    state = await store.get()

    assert state.active is False
    assert state.activated_by == "alice"


@pytest.mark.asyncio
async def test_upsert_overwrites_previous_reason() -> None:
    store = _store()
    now = datetime.now(tz=_UTC)

    await store.activate(reason="first_reason", activated_by="pytest", as_of=now)
    await store.activate(reason="second_reason", activated_by="pytest", as_of=datetime.now(tz=_UTC))
    state = await store.get()

    assert state.reason == "second_reason"

    await store.clear(operator_id="cleanup", as_of=datetime.now(tz=_UTC))


@pytest.mark.asyncio
async def test_default_state_is_inactive_after_clear() -> None:
    store = _store()
    now = datetime.now(tz=_UTC)

    await store.clear(operator_id="pytest", as_of=now)
    state = await store.get()

    assert state.active is False
