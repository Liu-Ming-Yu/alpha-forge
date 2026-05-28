"""Alembic migration roundtrip tests (F3).

Requires real Postgres (integration_durable marker).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration_durable


@pytest.mark.asyncio
async def test_alembic_upgrade_downgrade_roundtrip() -> None:
    """Upgrade to head, downgrade to base, upgrade to head again.

    Validates that every migration's downgrade() function is implemented and
    does not leave the database in an inconsistent state.

    Requires QP__STORAGE__POSTGRES_DSN to be set in the environment.
    """
    dsn = os.environ.get("QP__STORAGE__POSTGRES_DSN", "")
    if not dsn:
        pytest.skip("QP__STORAGE__POSTGRES_DSN is required for durable tests")
    # psycopg 3 provides the sync DBAPI used by Alembic/SQLAlchemy here.
    sync_dsn = dsn.replace("postgresql://", "postgresql+psycopg://")

    # Build an Alembic config pointing at the packaged migrations.
    import importlib.resources as resources

    from alembic import command
    from alembic.config import Config

    from quant_platform.infrastructure.support import migrations as mig_module

    migration_root = resources.files("quant_platform.alembic")
    with resources.as_file(migration_root) as script_location:
        cfg = Config()
        cfg.set_main_option("script_location", str(script_location))
        cfg.set_main_option("sqlalchemy.url", sync_dsn)

        # 1. Upgrade to head.
        command.upgrade(cfg, "head")

        # 2. Verify we are at head.
        head = mig_module.packaged_head()
        # Use the environment to read the current revision.
        from alembic.runtime.migration import MigrationContext
        from sqlalchemy import create_engine

        sync_dsn_pg = dsn.replace("postgresql://", "postgresql+psycopg://")
        engine = create_engine(sync_dsn_pg)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current = context.get_current_revision()
        engine.dispose()

        assert current == head, f"Expected schema at head={head!r} after upgrade, got {current!r}"

        # 3. Downgrade to base (reverses every migration).
        command.downgrade(cfg, "base")

        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current_after_downgrade = context.get_current_revision()
        engine.dispose()

        assert current_after_downgrade is None, (
            f"Expected None after downgrade to base, got {current_after_downgrade!r}"
        )

        # 4. Upgrade to head again.
        command.upgrade(cfg, "head")

        engine = create_engine(sync_dsn_pg)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current_final = context.get_current_revision()
        engine.dispose()

        assert current_final == head, (
            f"Expected schema at head={head!r} after second upgrade, got {current_final!r}"
        )
