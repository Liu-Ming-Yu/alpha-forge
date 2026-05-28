"""Enforce architecture import boundaries for the modular monolith.

Every clean-layer and edge-layer violation fails hard. The transitional
escape hatch was retired once the services->infrastructure/session and
edge->service refactors completed; there is no remaining debt to grandfather.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "quant_platform"
PROJECT_PREFIX = "quant_platform."

_STDLIB_AND_EXTERNAL = {
    "",
    "__future__",
}

COMPOSITION_LAYERS = {"bootstrap", "engines", "session", "research"}
EDGE_LAYERS = {"cli", "views"}
INNER_LAYERS = {"core", "services", "application"}


@dataclass(frozen=True)
class BoundaryViolation:
    path: Path
    module: str
    imported: str
    message: str


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT / "src").with_suffix("")
    return ".".join(rel.parts)


def _layer_for(module: str) -> str:
    if module == "quant_platform.session" or module.startswith("quant_platform.session."):
        return "session"
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "quant_platform":
        return "external"
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


def _is_project_import(module: str) -> bool:
    return module == "quant_platform" or module.startswith(PROJECT_PREFIX)


def _violation_for_import(path: Path, imported: str) -> BoundaryViolation | None:
    if imported in _STDLIB_AND_EXTERNAL or not _is_project_import(imported):
        return None
    module = _module_name(path)
    source_layer = _layer_for(module)
    target_layer = _layer_for(imported)

    if source_layer == "core" and target_layer != "core":
        return BoundaryViolation(
            path,
            module,
            imported,
            "core must not import outer layers",
        )
    if source_layer == "application" and target_layer in {
        "infrastructure",
        "bootstrap",
        "research",
        "cli",
        "views",
        "session",
    }:
        return BoundaryViolation(
            path,
            module,
            imported,
            "application use cases must depend on ports/services, not adapters or entrypoints",
        )
    if source_layer == "infrastructure" and target_layer in {
        "application",
        "bootstrap",
        "research",
        "cli",
        "views",
        "session",
    }:
        return BoundaryViolation(
            path,
            module,
            imported,
            "infrastructure adapters must not import application/bootstrap/entrypoints",
        )
    if source_layer == "services" and target_layer in {
        "infrastructure",
        "research",
        "session",
        "cli",
        "views",
    }:
        if imported == "quant_platform.infrastructure.metrics":
            return BoundaryViolation(
                path,
                module,
                imported,
                "services must use quant_platform.telemetry.metrics or injected telemetry ports",
            )
        return BoundaryViolation(
            path,
            module,
            imported,
            "service modules should depend on core ports and same-service helpers",
        )
    if source_layer in EDGE_LAYERS and target_layer in {"infrastructure", "services", "session"}:
        return BoundaryViolation(
            path,
            module,
            imported,
            "entrypoints should call application/bootstrap rather than "
            "storage or service internals",
        )
    return None


def collect_violations() -> list[BoundaryViolation]:
    violations: list[BoundaryViolation] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for imported in _imports(path):
            violation = _violation_for_import(path, imported)
            if violation is not None:
                violations.append(violation)
    return violations


def _format_violation(violation: BoundaryViolation) -> str:
    rel = violation.path.relative_to(ROOT)
    return f"[ERROR] {rel}: {violation.module} imports {violation.imported} ({violation.message})"


def main() -> int:
    violations = collect_violations()
    if violations:
        print("Import boundary violations:")
        for violation in violations:
            print(f"  - {_format_violation(violation)}")
        return 1

    print("Import boundary check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
