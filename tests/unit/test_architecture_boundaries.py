from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _boundary_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "check_import_boundaries.py"
    spec = importlib.util.spec_from_file_location("check_import_boundaries", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_boundary_checker_has_no_violations() -> None:
    module = _boundary_module()

    violations = module.collect_violations()

    assert violations == []


def test_boundary_checker_flags_core_outer_import_as_hard_error() -> None:
    module = _boundary_module()
    path = module.SRC / "core" / "domain" / "research.py"

    violation = module._violation_for_import(  # noqa: SLF001
        path,
        "quant_platform.services.research_service.feature_quality.audit.feature_audit",
    )

    assert violation is not None
    assert "core must not import outer layers" in violation.message


def test_boundary_checker_blocks_service_infrastructure_import_as_hard_error() -> None:
    module = _boundary_module()
    path = module.SRC / "services" / "research_service" / "backtest_engine.py"

    violation = module._violation_for_import(  # noqa: SLF001
        path,
        "quant_platform.infrastructure.support.clock",
    )

    assert violation is not None
    assert "core ports and same-service helpers" in violation.message


def test_boundary_checker_blocks_service_prometheus_import_as_hard_error() -> None:
    module = _boundary_module()
    path = module.SRC / "services" / "research_service" / "feature_pipeline.py"

    violation = module._violation_for_import(  # noqa: SLF001
        path,
        "quant_platform.infrastructure.metrics",
    )

    assert violation is not None
    assert "telemetry.metrics" in violation.message
