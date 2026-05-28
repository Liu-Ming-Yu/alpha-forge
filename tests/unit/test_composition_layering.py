"""Guard the composition-tier layering ratchet."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _layering_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "check_composition_layering.py"
    spec = importlib.util.spec_from_file_location("check_composition_layering", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_no_unapproved_composition_edges() -> None:
    module = _layering_module()

    report = module.build_report()

    assert report.unapproved == ()


def test_no_stale_composition_approvals() -> None:
    module = _layering_module()

    report = module.build_report()

    assert report.stale_approvals == ()


def test_composition_edge_is_layer_cross_only() -> None:
    module = _layering_module()

    edges = module.collect_composition_edges()

    assert edges, "expected the composition tier to have tracked edges"
    for edge in edges:
        assert edge.source_layer != edge.target_layer
        assert edge.source_layer in module.COMPOSITION_SOURCE_LAYERS


def test_composition_tier_is_acyclic() -> None:
    module = _layering_module()

    report = module.build_report()

    assert report.layer_cycles == ()
    assert module.ENFORCE_ACYCLIC is True
