"""Packaged Alembic environment configuration."""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config
target_metadata = None

dsn_override = os.environ.get("QP__STORAGE__POSTGRES_DSN", "")
if dsn_override:
    config.set_main_option("sqlalchemy.url", dsn_override)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
