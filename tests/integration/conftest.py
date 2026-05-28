"""Integration test fixtures: shared setup for durable-backend tests.

Autouse fixture clears kill-switch state and position snapshots before every
test so that durable state from one test does not contaminate later tests.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def _clear_shared_state_between_tests() -> None:
    """Reset durable Postgres state before each test.

    Clears:
    - kill_switch_state (singleton row may be left active)
    - account_snapshots + position_snapshots (ORDER BY as_of DESC LIMIT 1 is
      global; a far-future phantom from one test would hijack the next test's
      reconcile query)
    - fill_events + order_intents (broker_order_id uniqueness spans test runs)
    """
    dsn = os.environ.get("QP__STORAGE__POSTGRES_DSN", "").strip()
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
        await store.clear(operator_id="test_fixture", as_of=datetime.now(tz=UTC))

        async with engine.begin() as conn:
            # Delete position_snapshots first (FK → account_snapshots).
            await conn.execute(text("DELETE FROM position_snapshots"))
            await conn.execute(text("DELETE FROM account_snapshots"))
            # Delete fill_events first (FK → order_intents).
            await conn.execute(text("DELETE FROM fill_events"))
            await conn.execute(text("DELETE FROM order_intents"))
            # Clear audit_log so time-window tests see only their own events.
            await conn.execute(text("DELETE FROM audit_log"))
    except Exception:
        pass
