"""Unit tests for ``scripts/promote_latest_stack_arm.py``."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_platform.config import PlatformSettings, StorageSettings
from quant_platform.services.research_service.modeling.registry.latest_stack_promotion import (
    NotPromotableError,
)

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "promote_latest_stack_arm.py"
_spec = importlib.util.spec_from_file_location("promote_latest_stack_arm", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
promote_script = importlib.util.module_from_spec(_spec)
sys.modules["promote_latest_stack_arm"] = promote_script
_spec.loader.exec_module(promote_script)


def _evidence(**overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "arm": "long_only_top30_pv_formulaic_streakdial",
        "arm_cli_alias": "G",
        "arm_category": "portfolio_candidate",
        "production_candidate": True,
        "model_version": "ic-weighted-non-negative",
        "feature_set_version": "latest-stack-v1--g",
        "evidence_schema_version": "backtest-latest-stack-realized-v2.1",
        "run_id": "abc",
        "git_commit": "9d34da6",
        "saved_at_utc": "2026-05-28T19:17:52+00:00",
        "eligibility": {"passed": True, "checks": []},
        "eligibility_thresholds": {"name": "portfolio_candidate_v2"},
        "metrics": {"slippage_adjusted_sharpe": 1.0886},
        "universe_fingerprint": {},
        "bars_snapshot_fingerprint": {},
    }
    evidence.update(overrides)
    return evidence


def _write(tmp_path: Path, evidence: dict[str, object]) -> Path:
    path = tmp_path / "arm.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def test_main_emits_payload_for_eligible_arm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = promote_script.main(["--evidence", str(_write(tmp_path, _evidence()))])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy_name"] == "long_only_top30_pv_formulaic_streakdial"
    assert payload["as_of"] == "2026-05-28T19:17:52+00:00"
    assert payload["metadata"]["eligibility_thresholds"]["name"] == "portfolio_candidate_v2"


def test_main_writes_to_output_file(tmp_path: Path) -> None:
    out = tmp_path / "reg.json"
    code = promote_script.main(
        ["--evidence", str(_write(tmp_path, _evidence())), "--output", str(out)]
    )
    assert code == 0
    assert (
        json.loads(out.read_text(encoding="utf-8"))["model_version"] == "ic-weighted-non-negative"
    )


def test_main_exits_nonzero_for_ineligible_arm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence = _evidence(eligibility={"passed": False, "checks": []})
    code = promote_script.main(["--evidence", str(_write(tmp_path, evidence))])
    assert code == 2
    assert "NOT PROMOTABLE" in capsys.readouterr().err


def test_signal_type_choices_accept_classical_reject_unknown() -> None:
    parser = promote_script.build_arg_parser()
    args = parser.parse_args(
        [
            "--evidence",
            "x.json",
            "--register",
            "--signal-type",
            "classical",
            "--engine-version",
            "e",
        ]
    )
    assert args.signal_type == "classical"  # the latest-stack linear-ranker type
    with pytest.raises(SystemExit):
        parser.parse_args(["--evidence", "x.json", "--register", "--signal-type", "not-a-type"])


def _settings() -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        storage=StorageSettings(postgres_dsn="postgresql+psycopg://u:p@localhost/db"),
    )


class _FakeRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def register_model(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(model_id=uuid.uuid4(), model_version=kwargs["model_version"])


class _FakeHeartbeats:
    def __init__(self) -> None:
        self.beats: list[object] = []

    async def save_runtime_heartbeat(self, heartbeat: object) -> None:
        self.beats.append(heartbeat)


def test_register_from_evidence_composes_adapter_and_alpha_promote() -> None:
    registry = _FakeRegistry()
    heartbeats = _FakeHeartbeats()
    result = asyncio.run(
        promote_script.register_from_evidence(
            _settings(),
            _evidence(),
            signal_type="xgboost",
            engine_version="engine-v1",
            rollback_target="",
            artifact_manifest=None,
            as_of=None,
            model_registry=registry,
            heartbeat_repository=heartbeats,
        )
    )
    assert result["promoted"] is True
    assert len(registry.calls) == 1
    call = registry.calls[0]
    # Identity comes from the adapter (no operator transcription)...
    assert call["strategy_name"] == "long_only_top30_pv_formulaic_streakdial"
    assert call["model_version"] == "ic-weighted-non-negative"
    # ...and the gate result rides into the registry record.
    assert (
        call["metadata"]["evidence"]["eligibility_thresholds"]["name"] == "portfolio_candidate_v2"
    )
    assert len(heartbeats.beats) == 1  # audit heartbeat written


def test_register_from_evidence_rejects_ineligible_before_any_write() -> None:
    registry = _FakeRegistry()
    heartbeats = _FakeHeartbeats()
    evidence = _evidence(eligibility={"passed": False, "checks": []})
    with pytest.raises(NotPromotableError):
        asyncio.run(
            promote_script.register_from_evidence(
                _settings(),
                evidence,
                signal_type="xgboost",
                engine_version="engine-v1",
                rollback_target="",
                artifact_manifest=None,
                as_of=None,
                model_registry=registry,
                heartbeat_repository=heartbeats,
            )
        )
    assert registry.calls == []  # never reached the registry
