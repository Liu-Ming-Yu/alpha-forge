"""Ratchet the composition tier (bootstrap / engines / session).

``check_import_boundaries.py`` enforces clean-layer direction but deliberately
exempts the composition tier -- bootstrap, engines, and the session facade may
import anything. That exemption let two real dependency cycles
(bootstrap<->views, bootstrap<->engines) accumulate invisibly.

This checker closes that gap. It enumerates every cross-layer import whose
*source* is a composition-tier module and freezes the current graph as an
approved baseline. New cross-layer edges fail until they are either removed or
explicitly approved here, and approved edges that disappear (stale) also fail so
the baseline cannot rot. When ``ENFORCE_ACYCLIC`` is enabled the approved
layer-level graph must additionally be a DAG -- no composition-tier cycles.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "quant_platform"

# Composition-tier source layers whose outbound cross-layer edges are ratcheted.
COMPOSITION_SOURCE_LAYERS = frozenset({"bootstrap", "engines", "session", "research"})
# Layers an edge may point at and still be tracked (the composition tier plus
# the entrypoint layers, so bootstrap->views style back-edges are caught).
TRACKED_TARGET_LAYERS = frozenset({"bootstrap", "engines", "session", "research", "views", "cli"})

# The composition tier is acyclic (F3 complete). The approved graph below is a
# DAG; ENFORCE_ACYCLIC makes any future layer cycle a hard CI failure.
ENFORCE_ACYCLIC = True

# Frozen baseline of the composition-tier import graph -- a DAG:
#   {session, research} -> bootstrap -> engines, research -> engines, plus
#   bootstrap -> views (the operator-API seam). Do not grow it; the ratchet
#   fails any unapproved or stale edge, and ENFORCE_ACYCLIC fails any edge that
#   would re-introduce a layer cycle.
APPROVED_COMPOSITION_EDGES: frozenset[tuple[str, str]] = frozenset(
    {
        # bootstrap -> engines: composition building engine objects, wiring the
        # relocated session-runtime cluster, and the bootstrap-side session
        # composition helpers. Kept direction (composition assembles + drives
        # the runtime).
        (
            "src/quant_platform/bootstrap/engine/__init__.py",
            "quant_platform.engines.session.public_api",
        ),
        (
            "src/quant_platform/bootstrap/engine/multi.py",
            "quant_platform.engines.account.orchestrator",
        ),
        ("src/quant_platform/bootstrap/engine/multi.py", "quant_platform.engines.engine_runner"),
        (
            "src/quant_platform/bootstrap/engine/multi.py",
            "quant_platform.engines.market_data.price_seeding",
        ),
        ("src/quant_platform/bootstrap/engine/multi.py", "quant_platform.engines.multi_engine"),
        (
            "src/quant_platform/bootstrap/engine/multi.py",
            "quant_platform.engines.framework.plugins",
        ),
        (
            "src/quant_platform/bootstrap/engine/session_wiring.py",
            "quant_platform.engines.framework.types",
        ),
        (
            "src/quant_platform/bootstrap/engine/session_wiring.py",
            "quant_platform.engines.runtime.live",
        ),
        (
            "src/quant_platform/bootstrap/engine/session_wiring.py",
            "quant_platform.engines.market_data.provider",
        ),
        (
            "src/quant_platform/bootstrap/operator_adapters/broker.py",
            "quant_platform.engines.runtime.live",
        ),
        (
            "src/quant_platform/bootstrap/operator_adapters/broker.py",
            "quant_platform.engines.session.passive_reprice",
        ),
        (
            "src/quant_platform/bootstrap/operator_adapters/engine.py",
            "quant_platform.engines.engine_runner",
        ),
        (
            "src/quant_platform/bootstrap/session/__init__.py",
            "quant_platform.engines.session.runtime",
        ),
        (
            "src/quant_platform/bootstrap/session/__init__.py",
            "quant_platform.engines.session.strategy_cycle",
        ),
        (
            "src/quant_platform/bootstrap/session/public_factory.py",
            "quant_platform.engines.account.orchestrator",
        ),
        (
            "src/quant_platform/bootstrap/session/public_factory.py",
            "quant_platform.engines.multi_engine",
        ),
        # Intentional, permanent seam: the composition root assembles the
        # operator API by handing a Session to the FastAPI factory. Ratcheted
        # so no *other* bootstrap->views edge can appear.
        (
            "src/quant_platform/bootstrap/operator_api/app.py",
            "quant_platform.views.operator_api.app",
        ),
        # research -> bootstrap / engines: the research composition package
        # builds CLI-invoked research workflows on top of bootstrap's session,
        # signal-model, data, migration and feature-plugin composition helpers
        # (plus the engine session factory). One-directional -- nothing in
        # bootstrap or engines imports research, so the tier stays a DAG.
        (
            "src/quant_platform/research/adapters/__init__.py",
            "quant_platform.bootstrap.governance",
        ),
        (
            "src/quant_platform/research/backtesting/engine_factories.py",
            "quant_platform.bootstrap.session.public_api",
        ),
        (
            "src/quant_platform/research/backtesting/engine_factories.py",
            "quant_platform.engines.session.public_api",
        ),
        (
            "src/quant_platform/research/backtesting/ops.py",
            "quant_platform.bootstrap.session.public_api",
        ),
        (
            "src/quant_platform/research/backtesting/ops.py",
            "quant_platform.bootstrap.signal_models",
        ),
        (
            "src/quant_platform/research/campaign/signal_ops.py",
            "quant_platform.bootstrap.governance.repositories",
        ),
        ("src/quant_platform/research/common/__init__.py", "quant_platform.bootstrap.data"),
        (
            "src/quant_platform/research/common/__init__.py",
            "quant_platform.bootstrap.persistence.migrations",
        ),
        (
            "src/quant_platform/research/common/__init__.py",
            "quant_platform.bootstrap.session.public_api",
        ),
        (
            "src/quant_platform/research/common/__init__.py",
            "quant_platform.bootstrap.signal_models",
        ),
        (
            "src/quant_platform/research/features/backfill_compute.py",
            "quant_platform.bootstrap.data.feature_plugins",
        ),
        (
            "src/quant_platform/research/features/backfill_ops.py",
            "quant_platform.bootstrap.session.public_api",
        ),
        (
            "src/quant_platform/research/features/ops.py",
            "quant_platform.bootstrap.session.public_api",
        ),
        (
            "src/quant_platform/research/intraday/feature_backfill_ops/cli.py",
            "quant_platform.bootstrap.session.public_api",
        ),
        (
            "src/quant_platform/research/modeling/model_ops.py",
            "quant_platform.bootstrap.governance.repositories",
        ),
        # session facade -> composition tier: the public alias re-exporting the
        # composition (bootstrap) and runtime (engines) session APIs.
        ("src/quant_platform/session.py", "quant_platform.bootstrap.session.public_api"),
        ("src/quant_platform/session.py", "quant_platform.engines.session.public_api"),
    }
)


@dataclass(frozen=True)
class CompositionEdge:
    path: str
    imported: str
    source_layer: str
    target_layer: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.imported)


@dataclass(frozen=True)
class CompositionReport:
    current: tuple[CompositionEdge, ...]
    unapproved: tuple[CompositionEdge, ...]
    stale_approvals: tuple[tuple[str, str], ...]
    layer_cycles: tuple[tuple[str, str], ...]


def _module_name(path: Path, source_root: Path) -> str:
    rel = path.relative_to(source_root).with_suffix("")
    return "quant_platform." + ".".join(rel.parts)


def _layer_for(module: str) -> str | None:
    if module == "quant_platform.session" or module.startswith("quant_platform.session."):
        return "session"
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "quant_platform":
        return None
    return parts[1]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def collect_composition_edges(
    *,
    root: Path = ROOT,
    source_root: Path | None = None,
) -> tuple[CompositionEdge, ...]:
    root = root.resolve()
    source_root = (source_root or SOURCE_ROOT).resolve()
    edges: list[CompositionEdge] = []
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source_layer = _layer_for(_module_name(path, source_root))
        if source_layer not in COMPOSITION_SOURCE_LAYERS:
            continue
        rel = path.relative_to(root).as_posix()
        for imported in _imports(path):
            target_layer = _layer_for(imported)
            if target_layer is None or target_layer == source_layer:
                continue
            if target_layer not in TRACKED_TARGET_LAYERS:
                continue
            edges.append(
                CompositionEdge(
                    path=rel,
                    imported=imported,
                    source_layer=source_layer,
                    target_layer=target_layer,
                )
            )
    return tuple(sorted(set(edges), key=lambda edge: edge.key))


def _layer_cycles(edges: tuple[CompositionEdge, ...]) -> tuple[tuple[str, str], ...]:
    """Return layer pairs (a, b) that import each other in both directions."""
    directed = {(edge.source_layer, edge.target_layer) for edge in edges}
    cycles: set[tuple[str, str]] = set()
    for source, target in directed:
        if (target, source) in directed:
            cycles.add(tuple(sorted((source, target))))  # type: ignore[arg-type]
    return tuple(sorted(cycles))


def build_report(
    *,
    root: Path = ROOT,
    source_root: Path | None = None,
    approved: frozenset[tuple[str, str]] = APPROVED_COMPOSITION_EDGES,
) -> CompositionReport:
    current = collect_composition_edges(root=root, source_root=source_root)
    current_keys = frozenset(edge.key for edge in current)
    approved_edges = tuple(edge for edge in current if edge.key in approved)
    return CompositionReport(
        current=current,
        unapproved=tuple(edge for edge in current if edge.key not in approved),
        stale_approvals=tuple(sorted(approved - current_keys)),
        layer_cycles=_layer_cycles(approved_edges),
    )


def _format_edge(edge: CompositionEdge) -> str:
    return f"{edge.path}: {edge.source_layer} imports {edge.imported} ({edge.target_layer})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the full composition-tier edge graph",
    )
    args = parser.parse_args(argv)

    report = build_report()
    failed = False

    if report.unapproved:
        failed = True
        print("Composition-layering ratchet failed:", file=sys.stderr)
        print("  Unapproved composition-tier edges:", file=sys.stderr)
        for edge in report.unapproved:
            print(f"    - {_format_edge(edge)}", file=sys.stderr)
    if report.stale_approvals:
        failed = True
        print("  Stale approved composition-tier edges:", file=sys.stderr)
        for path, imported in report.stale_approvals:
            print(f"    - {path}: {imported}", file=sys.stderr)
    if ENFORCE_ACYCLIC and report.layer_cycles:
        failed = True
        print("  Composition-tier layer cycles (must be a DAG):", file=sys.stderr)
        for left, right in report.layer_cycles:
            print(f"    - {left} <-> {right}", file=sys.stderr)

    if failed:
        return 1

    print(
        f"Composition-layering ratchet passed: {len(report.current)} approved edges, "
        f"{len(report.layer_cycles)} layer cycle(s) remaining."
    )
    if args.report:
        for edge in report.current:
            print(f"  - {_format_edge(edge)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
