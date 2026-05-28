"""Ratchet service modules depending on bootstrap/session composition helpers.

Services should depend on core/application contracts and same-service helpers.
Direct service-to-bootstrap/session imports are forbidden because bootstrap is
the composition root and session is the runtime public facade. This
checker catches static imports plus common dynamic import escape hatches.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "quant_platform"

APPROVED_SERVICE_BOOTSTRAP_IMPORTS: frozenset[tuple[str, str]] = frozenset()


@dataclass(frozen=True)
class ServiceBootstrapImportEdge:
    path: str
    imported: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.imported)


@dataclass(frozen=True)
class ServiceBootstrapCouplingReport:
    current: tuple[ServiceBootstrapImportEdge, ...]
    unapproved: tuple[ServiceBootstrapImportEdge, ...]
    stale_approvals: tuple[tuple[str, str], ...]


def _module_name(path: Path, source_root: Path) -> str:
    rel = path.relative_to(source_root).with_suffix("")
    return "quant_platform." + ".".join(rel.parts)


def _is_service_module(module: str) -> bool:
    return module.startswith("quant_platform.services.")


def _is_bootstrap_import(module: str) -> bool:
    return module == "quant_platform.bootstrap" or module.startswith("quant_platform.bootstrap.")


def _is_session_import(module: str) -> bool:
    return module == "quant_platform.session" or module.startswith("quant_platform.session.")


def _is_forbidden_import(module: str) -> bool:
    return _is_bootstrap_import(module) or _is_session_import(module)


def _dynamic_import_target(node: ast.Call) -> str | None:
    func = node.func
    is_import_module = (
        isinstance(func, ast.Name)
        and func.id == "import_module"
        or isinstance(func, ast.Attribute)
        and func.attr == "import_module"
    )
    is_dunder_import = isinstance(func, ast.Name) and func.id == "__import__"
    if not is_import_module and not is_dunder_import:
        return None
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Call):
            target = _dynamic_import_target(node)
            if target is not None:
                modules.append(target)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _is_forbidden_import(node.value)
        ):
            modules.append(node.value)
    return modules


def collect_service_bootstrap_imports(
    *,
    root: Path = ROOT,
    source_root: Path | None = None,
) -> tuple[ServiceBootstrapImportEdge, ...]:
    root = root.resolve()
    source_root = (source_root or root / "src" / "quant_platform").resolve()
    edges: list[ServiceBootstrapImportEdge] = []
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if not _is_service_module(_module_name(path, source_root)):
            continue
        rel = path.relative_to(root).as_posix()
        for imported in _imports(path):
            if _is_forbidden_import(imported):
                edges.append(ServiceBootstrapImportEdge(path=rel, imported=imported))
    return tuple(sorted(set(edges), key=lambda edge: edge.key))


def build_report(
    *,
    root: Path = ROOT,
    source_root: Path | None = None,
    approved: frozenset[tuple[str, str]] = APPROVED_SERVICE_BOOTSTRAP_IMPORTS,
) -> ServiceBootstrapCouplingReport:
    current = collect_service_bootstrap_imports(root=root, source_root=source_root)
    current_keys = frozenset(edge.key for edge in current)
    return ServiceBootstrapCouplingReport(
        current=current,
        unapproved=tuple(edge for edge in current if edge.key not in approved),
        stale_approvals=tuple(sorted(approved - current_keys)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the approved services-to-bootstrap import baseline",
    )
    args = parser.parse_args(argv)

    report = build_report()
    if report.unapproved or report.stale_approvals:
        print("Service-bootstrap coupling ratchet failed:", file=sys.stderr)
        if report.unapproved:
            print("  Unapproved services-to-bootstrap imports:", file=sys.stderr)
            for edge in report.unapproved:
                print(f"    - {edge.path}: {edge.imported}", file=sys.stderr)
        if report.stale_approvals:
            print("  Stale services-to-bootstrap approvals:", file=sys.stderr)
            for path, imported in report.stale_approvals:
                print(f"    - {path}: {imported}", file=sys.stderr)
        return 1

    print(f"Service-bootstrap coupling ratchet passed: {len(report.current)} approved imports.")
    if args.report:
        for edge in report.current:
            print(f"  - {edge.path}: {edge.imported}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
