from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.check_generated_artifacts import (
    clean_generated_artifacts,
    collect_generated_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_generated_artifact_check_detects_source_caches(tmp_path: Path) -> None:
    pycache = tmp_path / "src" / "quant_platform" / "__pycache__"
    pycache.mkdir(parents=True)
    compiled = pycache / "module.cpython-311.pyc"
    compiled.write_bytes(b"compiled")

    artifacts = collect_generated_artifacts(tmp_path)

    assert [artifact.path.relative_to(tmp_path).as_posix() for artifact in artifacts] == [
        "src/quant_platform/__pycache__",
        "src/quant_platform/__pycache__/module.cpython-311.pyc",
    ]


def test_generated_artifact_clean_removes_detected_caches(tmp_path: Path) -> None:
    cache_dir = tmp_path / "tests" / "unit" / "__pycache__"
    cache_dir.mkdir(parents=True)
    (cache_dir / "test_sample.cpython-311-pytest-8.4.2.pyc").write_bytes(b"compiled")

    clean_generated_artifacts(collect_generated_artifacts(tmp_path))

    assert collect_generated_artifacts(tmp_path) == []
    assert not cache_dir.exists()


def test_generated_artifact_check_ignores_virtualenvs_and_data(tmp_path: Path) -> None:
    ignored_cache = tmp_path / ".venv-verify" / "lib" / "__pycache__"
    ignored_cache.mkdir(parents=True)
    (ignored_cache / "module.pyc").write_bytes(b"compiled")
    data_cache = tmp_path / "data" / "__pycache__"
    data_cache.mkdir(parents=True)

    assert collect_generated_artifacts(tmp_path) == []
