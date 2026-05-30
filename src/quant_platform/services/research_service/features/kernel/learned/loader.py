"""JSON serialisation for the ``PCAArtifact``.

Two operator-facing functions:

* :func:`save_pca_artifact` — dump an artifact to disk as JSON.
* :func:`load_pca_artifact` — load and validate an artifact from disk.

Used by the trainer (write) and operator scripts (read). The feature
family itself never touches disk during compute — it consumes an
already-loaded :class:`PCAArtifact` instance.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from quant_platform.services.research_service.features.kernel.learned.artifact import PCAArtifact

if TYPE_CHECKING:
    from pathlib import Path


def save_pca_artifact(artifact: PCAArtifact, path: Path) -> None:
    """Serialise an artifact to ``path`` as pretty-printed JSON.

    The directory must already exist — the loader doesn't create it.
    JSON is the canonical persistence format because it's
    git-diffable for audit + small enough to store in object stores
    next to other research artifacts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_pca_artifact(path: Path) -> PCAArtifact:
    """Load and validate an artifact from ``path``.

    Raises :class:`ValueError` from :meth:`PCAArtifact.from_dict` if
    the payload's schema version doesn't match the build's, or if
    any shape invariant fails in :class:`PCAArtifact.__post_init__`.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PCAArtifact.from_dict(payload)


__all__ = [
    "load_pca_artifact",
    "save_pca_artifact",
]
