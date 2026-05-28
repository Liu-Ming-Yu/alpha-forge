"""Migration command registrations."""

from __future__ import annotations

from typing import Any

from quant_platform.application.operator.requests import NoInputRequest
from quant_platform.cli.registry import bind_command


def register(sub: Any) -> None:
    migrate_p = sub.add_parser(
        "migrate",
        help="Upgrade the configured Postgres database to the packaged Alembic head.",
    )
    _bind_no_input(migrate_p, "infra.migrate")

    check_p = sub.add_parser(
        "migrations-check",
        help="Validate packaged Alembic revisions form a single offline migration chain.",
    )
    _bind_no_input(check_p, "infra.migrations_check")

    verify_p = sub.add_parser(
        "verify-schema",
        help="Verify the configured Postgres database is at the packaged Alembic head.",
    )
    _bind_no_input(verify_p, "infra.verify_schema")


def _bind_no_input(parser: Any, use_case_name: str) -> None:
    bind_command(
        parser,
        use_case_name=use_case_name,
        request_factory=lambda _args: NoInputRequest(),
        request_type=NoInputRequest,
    )
