"""Ratchet cross-service imports in the modular monolith.

The import-boundary checker enforces clean layer direction.  This checker is a
more focused service-coupling ratchet: current historical service-to-service
imports are approved explicitly, and new direct imports must be moved behind a
core contract, application use case, or bootstrap composition helper before CI
accepts them.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "quant_platform"

SERVICE_NAMES = frozenset(
    {
        "data_service",
        "execution_service",
        "governance_service",
        "portfolio_service",
        "research_service",
        "signal_service",
    }
)

APPROVED_CROSS_SERVICE_IMPORTS: frozenset[tuple[str, str]] = frozenset()


@dataclass(frozen=True)
class ServiceImportEdge:
    path: str
    imported: str
    source_service: str
    target_service: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.imported)


@dataclass(frozen=True)
class ServiceCouplingReport:
    current: tuple[ServiceImportEdge, ...]
    unapproved: tuple[ServiceImportEdge, ...]
    stale_approvals: tuple[tuple[str, str], ...]


def _service_for_module(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) > 3 and parts[:2] == ["quant_platform", "services"]:
        service = parts[2]
        if service in SERVICE_NAMES:
            return service
    return None


def _module_name(path: Path, source_root: Path) -> str:
    rel = path.relative_to(source_root).with_suffix("")
    return "quant_platform." + ".".join(rel.parts)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def collect_cross_service_imports(
    *,
    root: Path = ROOT,
    source_root: Path | None = None,
) -> tuple[ServiceImportEdge, ...]:
    root = root.resolve()
    source_root = (source_root or root / "src" / "quant_platform").resolve()
    edges: list[ServiceImportEdge] = []
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source_service = _service_for_module(_module_name(path, source_root))
        if source_service is None:
            continue
        rel = path.relative_to(root).as_posix()
        for imported in _imports(path):
            target_service = _service_for_module(imported)
            if target_service is None or target_service == source_service:
                continue
            edges.append(
                ServiceImportEdge(
                    path=rel,
                    imported=imported,
                    source_service=source_service,
                    target_service=target_service,
                )
            )
    return tuple(sorted(set(edges), key=lambda edge: edge.key))


def build_report(
    *,
    root: Path = ROOT,
    source_root: Path | None = None,
    approved: frozenset[tuple[str, str]] = APPROVED_CROSS_SERVICE_IMPORTS,
) -> ServiceCouplingReport:
    current = collect_cross_service_imports(root=root, source_root=source_root)
    current_keys = frozenset(edge.key for edge in current)
    return ServiceCouplingReport(
        current=current,
        unapproved=tuple(edge for edge in current if edge.key not in approved),
        stale_approvals=tuple(sorted(approved - current_keys)),
    )


def _format_edge(edge: ServiceImportEdge) -> str:
    return f"{edge.path}: {edge.source_service} imports {edge.imported} ({edge.target_service})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the approved cross-service import baseline",
    )
    args = parser.parse_args(argv)

    report = build_report()
    if report.unapproved or report.stale_approvals:
        print("Service-coupling ratchet failed:", file=sys.stderr)
        if report.unapproved:
            print("  Unapproved cross-service imports:", file=sys.stderr)
            for edge in report.unapproved:
                print(f"    - {_format_edge(edge)}", file=sys.stderr)
        if report.stale_approvals:
            print("  Stale approved cross-service imports:", file=sys.stderr)
            for path, imported in report.stale_approvals:
                print(f"    - {path}: {imported}", file=sys.stderr)
        return 1

    print(f"Service-coupling ratchet passed: {len(report.current)} approved imports.")
    if args.report:
        for edge in report.current:
            print(f"  - {_format_edge(edge)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
