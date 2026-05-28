from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.infrastructure.support.artifact_store import FileSystemArtifactStore


def test_filesystem_artifact_store_round_trips_json(tmp_path: Path) -> None:
    store = FileSystemArtifactStore(tmp_path)

    uri = store.write_json(
        "research/run_1/manifest.json",
        {"passed": True, "metrics": {"ic": 0.04}},
    )

    assert Path(uri).is_file()
    assert store.read_json(uri) == {"passed": True, "metrics": {"ic": 0.04}}


def test_filesystem_artifact_store_lists_matching_json(tmp_path: Path) -> None:
    store = FileSystemArtifactStore(tmp_path)
    store.write_json("research/a/manifest.json", {"run_id": "a"})
    store.write_json("research/b/manifest.json", {"run_id": "b"})
    store.write_json("research/b/other.json", {"run_id": "ignored"})

    rows = store.list_json("research", "*/manifest.json")

    assert [row["run_id"] for row in rows] == ["a", "b"]
    assert all("artifact_root" in row for row in rows)


def test_filesystem_artifact_store_rejects_path_escape(tmp_path: Path) -> None:
    store = FileSystemArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        store.write_json("../outside.json", {"passed": False})
