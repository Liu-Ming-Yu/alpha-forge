"""CLI command catalog + job execution routes.

Exposes the entire ``python -m quant_platform`` surface to the console:

* ``GET  /v1/commands``            — the introspected command catalog (metadata
  only; always available behind auth).
* ``POST /v1/commands/run``        — launch a command as a job. Gated by
  ``QP__API__ENABLE_COMMAND_EXECUTION`` and a typed confirmation for commands
  flagged dangerous.
* ``GET  /v1/jobs`` / ``/v1/jobs/{id}`` / ``POST /v1/jobs/{id}/cancel`` — track,
  tail (cursor-based), and cancel running jobs.
"""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from quant_platform.views.operator_api.commands import (
    build_command_catalog,
    find_command,
    get_job_store,
    reconstruct_argv,
)
from quant_platform.views.operator_api.commands.jobs import start_job

if TYPE_CHECKING:
    import argparse

    from fastapi import FastAPI

    from quant_platform.views.operator_api.routers.context import OperatorApiRouteContext

_CATALOG_CACHE: dict[str, Any] | None = None


def _catalog() -> dict[str, Any]:
    global _CATALOG_CACHE  # noqa: PLW0603 - cache the static CLI surface once
    if _CATALOG_CACHE is None:
        _CATALOG_CACHE = build_command_catalog()
    return _CATALOG_CACHE


_PARSER: argparse.ArgumentParser | None = None


def _root_parser() -> argparse.ArgumentParser:
    global _PARSER  # noqa: PLW0603 - cache the (read-only) argparse root parser
    if _PARSER is None:
        from quant_platform.cli.app import build_parser

        _PARSER = build_parser()
    return _PARSER


def _validate_argv(argv: list[str]) -> tuple[bool, str | None]:
    """Dry-run the argv through argparse (no dispatch) to catch errors early.

    Safety invariant: ``parse_args`` invokes each argument's ``type=`` converter
    in-process. The CLI uses only pure scalar converters (int/float/Decimal/str),
    so this is side-effect-free. Do NOT introduce a side-effecting argparse type
    (e.g. ``FileType`` or a converter that does I/O) without gating validation —
    it would run here without the command-execution opt-in.
    """
    parser = _root_parser()
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            parser.parse_args(argv)
    except SystemExit:
        message = err.getvalue().strip()
        last = message.splitlines()[-1] if message else "invalid arguments"
        return False, last.replace("python -m quant_platform ", "").strip()
    except Exception as exc:  # noqa: BLE001 - surface any type-coercion error
        return False, str(exc)
    return True, None


class RunCommandBody(BaseModel):
    path: list[str]
    values: dict[str, Any] = {}
    confirm: str = ""


class ValidateCommandBody(BaseModel):
    path: list[str]
    values: dict[str, Any] = {}


def register_command_routes(app: FastAPI, ctx: OperatorApiRouteContext) -> None:
    deps = ctx.protected_dependencies
    store = get_job_store()

    def _execution_enabled() -> bool:
        return bool(getattr(ctx.settings.api, "enable_command_execution", False))

    @app.get("/v1/commands", dependencies=deps)
    async def list_commands() -> JSONResponse:
        return JSONResponse(content={**_catalog(), "execution_enabled": _execution_enabled()})

    @app.post("/v1/commands/run", dependencies=deps)
    async def run_command(body: RunCommandBody) -> JSONResponse:
        if not _execution_enabled():
            raise HTTPException(
                status_code=403,
                detail="command execution is disabled; set QP__API__ENABLE_COMMAND_EXECUTION=true",
            )
        command = find_command(body.path, _catalog())
        if command is None:
            raise HTTPException(status_code=404, detail=f"unknown command: {' '.join(body.path)}")
        if command.get("dangerous") and body.confirm != "RUN":
            raise HTTPException(
                status_code=400,
                detail='this command is flagged dangerous; resend with confirm="RUN"',
            )
        argv = reconstruct_argv(command, body.values)
        try:
            job = store.create(body.path, argv)
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        start_job(store, job)
        return JSONResponse(content=store.snapshot(job.id) or {}, status_code=202)

    @app.post("/v1/commands/validate", dependencies=deps)
    async def validate_command(body: ValidateCommandBody) -> JSONResponse:
        command = find_command(body.path, _catalog())
        if command is None:
            raise HTTPException(status_code=404, detail=f"unknown command: {' '.join(body.path)}")
        argv = reconstruct_argv(command, body.values)
        ok, error = _validate_argv(argv)
        return JSONResponse(content={"ok": ok, "error": error, "argv": argv})

    @app.get("/v1/jobs", dependencies=deps)
    async def list_jobs() -> JSONResponse:
        return JSONResponse(content={"jobs": store.list_snapshot()})

    @app.get("/v1/jobs/{job_id}", dependencies=deps)
    async def get_job(job_id: str, since: int = 0) -> JSONResponse:
        snapshot = store.snapshot(job_id, since=since)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return JSONResponse(content=snapshot)

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=deps)
    async def cancel_job(job_id: str) -> JSONResponse:
        if not store.cancel(job_id):
            raise HTTPException(status_code=404, detail="job not found or already finished")
        return JSONResponse(content={"status": "cancelling"})
