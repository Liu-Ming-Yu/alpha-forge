"""Unit tests for live engine runtime helpers."""

from __future__ import annotations

import pytest

from quant_platform.config import PlatformSettings, V2Settings
from quant_platform.engines.runtime.live import assert_v2_is_only_live_submitter


def _settings(
    *,
    enabled: bool,
    orchestrator: bool,
) -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        v2=V2Settings(
            enabled=enabled,
            account_orchestrator_enabled=orchestrator,
        ),
    )


def test_live_runtime_allows_single_engine_path_when_v2_orchestrator_disabled() -> None:
    assert_v2_is_only_live_submitter(_settings(enabled=False, orchestrator=True))
    assert_v2_is_only_live_submitter(_settings(enabled=True, orchestrator=False))


def test_live_runtime_blocks_single_engine_submitter_when_v2_owns_live_orders() -> None:
    with pytest.raises(RuntimeError, match="V2 account orchestrator is enabled"):
        assert_v2_is_only_live_submitter(_settings(enabled=True, orchestrator=True))
