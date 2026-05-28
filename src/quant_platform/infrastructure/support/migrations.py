"""Install-safe Alembic migration helpers."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from alembic.config import Config

    from quant_platform.config import PlatformSettings


def _sync_dsn(dsn: str) -> str:
    # Alembic uses a sync SQLAlchemy engine. psycopg 3 provides both sync and
    # async DBAPI surfaces behind the same ``postgresql+psycopg`` dialect.
    return dsn.replace("postgresql://", "postgresql+psycopg://", 1)


@contextmanager
def alembic_config(dsn: str | None = None) -> Iterator[Config]:
    """Build an Alembic config from packaged migration resources."""
    try:
        from alembic.config import Config
    except Exception as exc:  # pragma: no cover - alembic is an install-time dep
        raise SystemExit(f"alembic is required for schema verification: {exc}") from exc

    migration_root = resources.files("quant_platform.alembic")
    with resources.as_file(migration_root) as script_location:
        versions_dir = Path(script_location) / "versions"
        if not versions_dir.exists():
            raise SystemExit(
                "Packaged Alembic migration assets are missing; cannot verify database schema head"
            )
        cfg = Config()
        cfg.set_main_option("script_location", str(script_location))
        cfg.set_main_option(
            "sqlalchemy.url",
            _sync_dsn(dsn or "postgresql://user:password@localhost:5432/quant_platform"),
        )
        yield cfg


def packaged_head() -> str:
    """Return the packaged Alembic head revision identifier."""
    return validate_packaged_migration_chain()


def validate_packaged_migration_chain() -> str:
    """Validate packaged Alembic revisions form one linear chain and return the head.

    This is intentionally offline: CI can run it without a database to catch
    duplicate revisions, missing parents, accidental branches, and packaged
    migration-resource drift before deployment.
    """
    migration_root = resources.files("quant_platform.alembic")
    with resources.as_file(migration_root) as script_location:
        versions_dir = Path(script_location) / "versions"
        if not versions_dir.exists():
            raise SystemExit(
                "Packaged Alembic migration assets are missing; cannot verify database schema head"
            )
        revisions: dict[str, str | None] = {}
        parents: set[str] = set()
        for revision_file in versions_dir.glob("*.py"):
            if revision_file.name == "__init__.py":
                continue
            try:
                tree = ast.parse(revision_file.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                raise SystemExit(
                    "Corrupt Alembic migration file: "
                    f"{revision_file.name} has a syntax error: {exc}"
                ) from exc
            revision: str | None = None
            down_revision: str | None = None
            for node in tree.body:
                if not isinstance(node, ast.AnnAssign | ast.Assign):
                    continue
                target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                value = node.value
                if target.id == "revision" and isinstance(value, ast.Constant):
                    revision = str(value.value)
                elif target.id == "down_revision" and isinstance(value, ast.Constant):
                    down_revision = None if value.value is None else str(value.value)
            if revision is None:
                raise SystemExit(f"{revision_file.name} has no revision identifier")
            if revision in revisions:
                raise SystemExit(f"duplicate Alembic revision identifier: {revision}")
            revisions[revision] = down_revision
            if down_revision:
                parents.add(down_revision)
        missing_parents = sorted(parent for parent in parents if parent not in revisions)
        if missing_parents:
            raise SystemExit(
                "Alembic migration chain references missing parent revisions: "
                + ", ".join(missing_parents)
            )
        roots = sorted(revision for revision, parent in revisions.items() if parent is None)
        if len(roots) != 1:
            raise SystemExit("Alembic migration chain must have exactly one root")
        heads = sorted(set(revisions) - parents)
    if len(heads) != 1:
        raise SystemExit("Alembic migration chain must have exactly one head")

    # Walk parent links from head to root.  If the walk cannot visit every
    # revision exactly once, the graph is disconnected or cyclic.
    seen: set[str] = set()
    current: str | None = heads[0]
    while current is not None:
        if current in seen:
            raise SystemExit(f"Alembic migration chain contains a cycle at {current}")
        seen.add(current)
        current = revisions[current]
    if seen != set(revisions):
        missing = sorted(set(revisions) - seen)
        raise SystemExit("Alembic migration chain is disconnected from head: " + ", ".join(missing))
    return heads[0]


async def verify_alembic_head(dsn: str) -> None:
    """Refuse to proceed if the database schema is behind the packaged head."""
    from sqlalchemy import text

    from quant_platform.infrastructure.postgres.repositories import create_pg_engine

    expected_head = packaged_head()
    engine = create_pg_engine(dsn)
    async with engine.connect() as conn:
        try:
            row = (
                (await conn.execute(text("SELECT version_num FROM alembic_version")))
                .mappings()
                .first()
            )
        except Exception:
            row = None
    current = str(row["version_num"]) if row else "<uninitialised>"
    if current != expected_head:
        raise SystemExit(
            "Database schema is not at the packaged Alembic head. "
            f"current={current!r} expected={expected_head!r}. "
            "Run: python -m quant_platform migrate"
        )


async def verify_postgres_schema_if_configured(settings: PlatformSettings) -> None:
    """Verify the Alembic head before opening any Postgres-backed runtime path."""
    if settings.storage.postgres_dsn:
        await verify_alembic_head(settings.storage.postgres_dsn)


def migrate_database(settings: PlatformSettings) -> str:
    """Upgrade the configured database to the packaged Alembic head."""
    if not settings.storage.postgres_dsn:
        raise SystemExit(
            "QP__STORAGE__POSTGRES_DSN is not set; nothing to migrate. "
            "Export a DSN or switch to in-memory mode."
        )
    try:
        from alembic import command
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"alembic is required for migrate: {exc}") from exc

    with alembic_config(settings.storage.postgres_dsn) as cfg:
        command.upgrade(cfg, "head")
    return packaged_head()
