from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from tests.ibkr_paper_safety import require_paper_order_safety

ROOT = Path(__file__).resolve().parents[2]


def _constraints() -> dict[str, Version]:
    pins: dict[str, Version] = {}
    for line in (ROOT / "constraints" / "py311.txt").read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        name, sep, version = raw.partition("==")
        assert sep == "==", f"constraint must be an exact pin: {raw}"
        pins[canonicalize_name(name)] = Version(version)
    return pins


def test_constraints_satisfy_declared_project_requirements() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    requirements = list(project["dependencies"])
    for optional_requirements in project["optional-dependencies"].values():
        requirements.extend(optional_requirements)

    pins = _constraints()
    missing: list[str] = []
    incompatible: list[str] = []
    for raw_requirement in requirements:
        requirement = Requirement(raw_requirement)
        key = canonicalize_name(requirement.name)
        if key not in pins:
            missing.append(requirement.name)
            continue
        if not requirement.specifier.contains(pins[key], prereleases=True):
            incompatible.append(f"{requirement.name}=={pins[key]} not in {requirement.specifier}")

    assert missing == []
    assert incompatible == []


def test_docker_compose_overrides_host_local_runtime_addresses() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "QP__STORAGE__POSTGRES_DSN" in compose
    assert "@postgres:5432/quant_platform" in compose
    assert "QP__STORAGE__REDIS_URL: redis://redis:6379/0" in compose
    assert "QP__BROKER__HOST: ${QP__DOCKER_BROKER_HOST:-host.docker.internal}" in compose
    assert "QP__BROKER__PORT: ${QP__DOCKER_BROKER_PORT:-7497}" in compose
    assert "host.docker.internal:host-gateway" in compose
    assert "restart: unless-stopped" in compose
    assert '"supervise",' in compose
    assert '"--execution-backend",' in compose
    assert '"ib-paper",' in compose


def test_docker_compose_config_renders_with_required_local_password() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")

    env = os.environ.copy()
    env["POSTGRES_PASSWORD"] = "quant"
    result = subprocess.run(
        ["docker", "compose", "--profile", "paper", "config"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "@postgres:5432/quant_platform" in result.stdout
    assert "redis://redis:6379/0" in result.stdout
    assert 'QP__BROKER__PORT: "7497"' in result.stdout
    assert "restart: unless-stopped" in result.stdout
    assert "- supervise" in result.stdout


def test_paper_order_safety_accepts_paper_account_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QP__BROKER__PAPER_TRADING", "true")

    require_paper_order_safety(account_id="DU1234567", port=7497)
    require_paper_order_safety(account_id="du1234567", port=4002)


def test_paper_order_safety_rejects_live_account(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QP__BROKER__PAPER_TRADING", "true")

    with pytest.raises(pytest.fail.Exception):
        require_paper_order_safety(account_id="U1234567", port=7497)


def test_paper_order_safety_rejects_live_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QP__BROKER__PAPER_TRADING", "true")

    with pytest.raises(pytest.fail.Exception):
        require_paper_order_safety(account_id="DU1234567", port=7496)
