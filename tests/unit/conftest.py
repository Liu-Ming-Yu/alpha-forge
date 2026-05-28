"""Unit test fixtures.

Clears durable Postgres state (kill switch, positions, orders) before each
test that may be running in an environment where integration tests have
previously left persistent state.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest_asyncio

_OFFLINE_STORAGE_DEFAULTS = {
    "QP__STORAGE__POSTGRES_DSN": "",
    "QP__STORAGE__REDIS_URL": "",
    "QP__STORAGE__EVENT_BUS_BACKEND": "in_memory",
}

for _key, _value in _OFFLINE_STORAGE_DEFAULTS.items():
    os.environ.setdefault(_key, _value)


def _explicit_postgres_dsn() -> str:
    """Return the test-requested Postgres DSN without falling back to repo .env."""
    if "QP__STORAGE__POSTGRES_DSN" not in os.environ:
        return ""
    try:
        from quant_platform.config import PlatformSettings

        return str(PlatformSettings(_env_file=None).storage.postgres_dsn or "").strip()
    except Exception:
        return os.environ.get("QP__STORAGE__POSTGRES_DSN", "").strip()


@pytest_asyncio.fixture(autouse=True)
async def _clear_durable_state() -> None:
    """Clear Postgres kill-switch and position state before each unit test.

    When integration tests run first (e.g. in a combined test run), they can
    leave the kill switch active or inject phantom snapshots that hijack
    get_latest_snapshot().  This fixture prevents that contamination.
    """
    dsn = _explicit_postgres_dsn()
    if not dsn:
        return

    try:
        from sqlalchemy import text

        from quant_platform.infrastructure.postgres.repositories import create_pg_engine
        from quant_platform.services.execution_service.stores.kill_switch_store import (
            PostgresKillSwitchStore,
        )

        engine = create_pg_engine(dsn)
        store = PostgresKillSwitchStore(engine)
        await store.clear(operator_id="unit_test_fixture", as_of=datetime.now(tz=UTC))

        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM position_snapshots"))
            await conn.execute(text("DELETE FROM account_snapshots"))
            await conn.execute(text("DELETE FROM fill_events"))
            await conn.execute(text("DELETE FROM order_intents"))
    except Exception:
        pass
