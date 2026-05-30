"""Unit tests for the CLI command catalog + job execution surface.

Covers:
- Catalog introspection captures the full CLI surface and groups it.
- Danger classification (mutators flagged, read-only ``*-check``/``smoke`` not).
- argv reconstruction from form values (empties skipped, options mapped).
- API gating: execution opt-in flag, dangerous-command confirmation, auth.
- The threaded subprocess runner executes a fast, safe command end-to-end.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from quant_platform.config import ApiSettings, PlatformSettings, StorageSettings
from quant_platform.views.operator_api.app import create_app
from quant_platform.views.operator_api.commands.catalog import (
    _is_dangerous,
    build_command_catalog,
    find_command,
    reconstruct_argv,
)
from quant_platform.views.operator_api.commands.jobs import _MAX_LOG_LINES, JobStore, start_job

_H = {"X-API-Key": "test-key"}


def _settings(*, enable_exec: bool = False) -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="", redis_url="", event_bus_backend="in_memory"),
        api=ApiSettings(operator_api_key="test-key", enable_command_execution=enable_exec),
    )


def _client(*, enable_exec: bool = False) -> TestClient:
    return TestClient(create_app(settings=_settings(enable_exec=enable_exec)))


def _count(catalog: dict[str, Any]) -> int:
    total = 0

    def walk(node: dict[str, Any]) -> None:
        nonlocal total
        if node["type"] == "command":
            total += 1
        else:
            for child in node["commands"]:
                walk(child)

    for group in catalog["groups"]:
        for command in group["commands"]:
            walk(command)
    return total


def test_catalog_covers_expected_groups() -> None:
    catalog = build_command_catalog()
    names = {g["name"] for g in catalog["groups"]}
    assert {"runtime", "broker", "engines", "migrations", "research", "governance"} <= names
    assert _count(catalog) > 50


def test_find_and_danger_classification() -> None:
    supervise = find_command(["supervise"])
    assert supervise is not None
    assert supervise["dangerous"] is True
    assert supervise["long_running"] is True
    # Read-only verbs override the danger heuristic.
    checks = find_command(["migrations-check"])
    assert checks is not None
    assert checks["dangerous"] is False
    assert find_command(["definitely-not-a-command"]) is None


def test_reconstruct_argv_skips_empty_and_maps_options() -> None:
    supervise = find_command(["supervise"])
    assert supervise is not None
    argv = reconstruct_argv(supervise, {"interval": "600", "mode": "shadow", "initial_cash": ""})
    assert argv[0] == "supervise"
    assert "--interval" in argv
    assert "600" in argv
    assert "--mode" in argv
    assert "shadow" in argv
    assert "--initial-cash" not in argv  # empty value omitted


def test_list_default_round_trips_through_validate() -> None:
    # A nargs='+' arg (research-campaign run --attribution-horizons) defaults to a
    # Python list [5, 10, 21]. The catalog must surface it as the comma form the UI
    # edits ("5,10,21"), and reconstruct must emit one option with many values
    # (--attribution-horizons 5 10 21), not the repr "[5, 10, 21]" which fails re-parse.
    run = find_command(["research-campaign", "run"])
    assert run is not None
    horizons = next(a for a in run["args"] if a["dest"] == "attribution_horizons")
    assert horizons["default"] == "5,10,21"
    assert horizons["nargs"] == "+"
    argv = reconstruct_argv(run, {"attribution_horizons": ["5", "10", "21"]})
    assert argv[-4:] == ["--attribution-horizons", "5", "10", "21"]

    client = _client()
    body = client.post(
        "/v1/commands/validate",
        headers=_H,
        json={
            "path": ["research-campaign", "run"],
            "values": {
                "contracts_file": "x.json",
                "start": "2020-01-01",
                "end": "2020-02-01",
                "model_version": "v1",
                "attribution_horizons": ["5", "10", "21"],
            },
        },
    ).json()
    assert body["ok"] is True, body
    assert "--attribution-horizons" in body["argv"]


def test_commands_endpoint_lists() -> None:
    response = _client().get("/v1/commands", headers=_H)
    assert response.status_code == 200
    body = response.json()
    assert "groups" in body
    assert body["execution_enabled"] is False


def test_commands_endpoint_requires_auth() -> None:
    assert _client().get("/v1/commands").status_code == 401


def test_run_requires_execution_flag() -> None:
    response = _client(enable_exec=False).post(
        "/v1/commands/run", headers=_H, json={"path": ["health"], "values": {}}
    )
    assert response.status_code == 403


def test_run_unknown_command_is_404() -> None:
    response = _client(enable_exec=True).post(
        "/v1/commands/run", headers=_H, json={"path": ["no-such-command"], "values": {}}
    )
    assert response.status_code == 404


def test_dangerous_command_requires_confirmation() -> None:
    client = _client(enable_exec=True)
    response = client.post("/v1/commands/run", headers=_H, json={"path": ["migrate"], "values": {}})
    assert response.status_code == 400
    assert "confirm" in response.json()["detail"].lower()


def test_validate_command_endpoint() -> None:
    client = _client()  # validation never executes, so the exec flag is irrelevant
    ok = client.post(
        "/v1/commands/validate", headers=_H, json={"path": ["migrations-check"], "values": {}}
    )
    assert ok.status_code == 200
    assert ok.json()["ok"] is True

    missing = client.post(
        "/v1/commands/validate",
        headers=_H,
        json={"path": ["research-campaign", "run"], "values": {}},
    )
    body = missing.json()
    assert body["ok"] is False
    assert "required" in (body["error"] or "").lower()

    bad_type = client.post(
        "/v1/commands/validate",
        headers=_H,
        json={"path": ["supervise"], "values": {"interval": "abc"}},
    )
    assert bad_type.json()["ok"] is False

    assert (
        client.post(
            "/v1/commands/validate", headers=_H, json={"path": ["nope"], "values": {}}
        ).status_code
        == 404
    )
    assert client.post("/v1/commands/validate", json={"path": ["x"]}).status_code == 401


def test_strong_danger_overrides_read_only_verb() -> None:
    # A read-only verb keeps a command safe...
    assert _is_dangerous("misc", ["status-report"]) is False
    # ...but an irreversibly-destructive verb wins even alongside a safe verb,
    # so a "delete --list"-style name can't masquerade as read-only.
    assert _is_dangerous("misc", ["delete-list"]) is True
    assert _is_dangerous("misc", ["purge-status"]) is True
    # Real commands are unchanged: migrate mutates, migrations-check is read-only.
    assert _is_dangerous("runtime", ["migrate"]) is True
    assert _is_dangerous("migrations", ["migrations-check"]) is False


def test_job_log_cursor_survives_ring_buffer_trim() -> None:
    # Once output exceeds the ring buffer, the absolute cursor must keep
    # advancing so the live tail never freezes (regression: a relative cursor
    # pinned at the cap and stopped delivering new lines).
    store = JobStore()
    job = store.create(["health"], ["health"])
    total = _MAX_LOG_LINES + 250
    for i in range(total):
        store.append_log(job, f"line-{i}")

    head = store.snapshot(job.id)
    assert head is not None
    assert head["log_cursor"] == total  # absolute total ever produced
    assert head["log_lines"] == total

    # A client caught up to (total-100) still receives exactly the last 100 lines.
    tail = store.snapshot(job.id, since=total - 100)
    assert tail is not None
    assert tail["logs"] == [f"line-{i}" for i in range(total - 100, total)]
    assert tail["log_cursor"] == total

    # A fully caught-up client gets nothing new — no freeze, no duplication.
    caught = store.snapshot(job.id, since=total)
    assert caught is not None
    assert caught["logs"] == []
    assert caught["log_cursor"] == total


def test_jobstore_runs_safe_command_end_to_end() -> None:
    store = JobStore()
    job = store.create(["--help"], ["--help"])
    start_job(store, job)
    snapshot: dict[str, Any] | None = None
    for _ in range(100):
        snapshot = store.snapshot(job.id)
        assert snapshot is not None
        if snapshot["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.1)
    assert snapshot is not None
    assert snapshot["status"] == "succeeded"
    assert snapshot["exit_code"] == 0
    assert any("quant_platform" in line or "usage" in line.lower() for line in snapshot["logs"])
