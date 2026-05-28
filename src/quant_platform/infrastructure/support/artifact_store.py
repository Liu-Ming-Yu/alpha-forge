"""Filesystem-backed artifact store adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path


class FileSystemArtifactStore:
    """JSON artifact store rooted at the configured object-store directory."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        """Return the resolved filesystem root."""
        return self._root

    def read_json(self, uri: str) -> Mapping[str, object]:
        """Read one JSON artifact from an absolute or root-relative path."""
        path = self._resolve(uri)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSON artifact must be an object: {path}")
        return {str(key): value for key, value in payload.items()}

    def write_json(self, uri: str, payload: Mapping[str, object]) -> str:
        """Write one JSON artifact and return its canonical path."""
        path = self._resolve(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return str(path)

    def list_json(self, prefix: str, pattern: str) -> Sequence[Mapping[str, object]]:
        """Return JSON objects below ``prefix`` matching ``pattern``."""
        base = self._resolve(prefix)
        if not base.exists():
            return ()
        rows: list[Mapping[str, object]] = []
        for path in sorted(base.glob(pattern)):
            if not path.is_file():
                continue
            try:
                payload = self.read_json(str(path))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            row = dict(payload)
            row.setdefault("artifact_uri", str(path))
            row.setdefault("artifact_root", str(path.parent))
            rows.append(row)
        return tuple(rows)

    def _resolve(self, uri: str) -> Path:
        raw = Path(uri)
        path = raw if raw.is_absolute() else self._root / raw
        resolved = path.resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise ValueError(f"artifact path escapes object-store root: {uri}")
        return resolved
