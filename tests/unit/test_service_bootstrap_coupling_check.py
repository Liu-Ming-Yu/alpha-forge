from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.check_service_bootstrap_coupling import (
    ServiceBootstrapImportEdge,
    build_report,
    collect_service_bootstrap_imports,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_collect_service_bootstrap_imports_reports_service_edges(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_platform"
    _write(
        source_root / "services" / "governance_service" / "example.py",
        "from quant_platform.bootstrap.governance.repositories import build_repo\n",
    )
    _write(
        source_root / "bootstrap" / "research" / "example.py",
        "from quant_platform.bootstrap.runtime import create_paper_session\n",
    )
    _write(
        source_root / "services" / "research_service" / "local.py",
        "from quant_platform.core.contracts import PerformanceRepository\n",
    )

    edges = collect_service_bootstrap_imports(root=tmp_path, source_root=source_root)

    assert edges == (
        ServiceBootstrapImportEdge(
            path="src/quant_platform/services/governance_service/example.py",
            imported="quant_platform.bootstrap.governance.repositories",
        ),
    )


def test_service_bootstrap_report_blocks_unapproved_edges(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_platform"
    _write(
        source_root / "services" / "signal_service" / "example.py",
        "import quant_platform.bootstrap.runtime\n",
    )

    report = build_report(root=tmp_path, source_root=source_root, approved=frozenset())

    assert [edge.key for edge in report.unapproved] == [
        (
            "src/quant_platform/services/signal_service/example.py",
            "quant_platform.bootstrap.runtime",
        )
    ]


def test_service_bootstrap_report_flags_stale_approvals(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_platform"
    source_root.mkdir(parents=True)

    report = build_report(
        root=tmp_path,
        source_root=source_root,
        approved=frozenset(
            {
                (
                    "src/quant_platform/services/governance_service/missing.py",
                    "quant_platform.bootstrap.runtime",
                )
            }
        ),
    )

    assert report.stale_approvals == (
        (
            "src/quant_platform/services/governance_service/missing.py",
            "quant_platform.bootstrap.runtime",
        ),
    )


def test_collect_service_bootstrap_imports_reports_importlib_edges(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_platform"
    _write(
        source_root / "services" / "research_service" / "example.py",
        "from importlib import import_module\n"
        "runtime = import_module('quant_platform.bootstrap.runtime')\n",
    )

    edges = collect_service_bootstrap_imports(root=tmp_path, source_root=source_root)

    assert edges == (
        ServiceBootstrapImportEdge(
            path="src/quant_platform/services/research_service/example.py",
            imported="quant_platform.bootstrap.runtime",
        ),
    )


def test_collect_service_bootstrap_imports_reports_dunder_import_edges(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_platform"
    _write(
        source_root / "services" / "research_service" / "example.py",
        "__import__('quant_platform.session')\n",
    )

    edges = collect_service_bootstrap_imports(root=tmp_path, source_root=source_root)

    assert edges == (
        ServiceBootstrapImportEdge(
            path="src/quant_platform/services/research_service/example.py",
            imported="quant_platform.session",
        ),
    )


def test_collect_service_bootstrap_imports_reports_literal_targets(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_platform"
    _write(
        source_root / "services" / "research_service" / "example.py",
        "RUNTIME = 'quant_platform.bootstrap.runtime'\n",
    )

    edges = collect_service_bootstrap_imports(root=tmp_path, source_root=source_root)

    assert edges == (
        ServiceBootstrapImportEdge(
            path="src/quant_platform/services/research_service/example.py",
            imported="quant_platform.bootstrap.runtime",
        ),
    )
