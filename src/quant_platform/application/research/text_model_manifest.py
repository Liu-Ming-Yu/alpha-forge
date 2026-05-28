"""Application-level text alpha model manifest helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.application.features.admission import ordered_feature_schema_hash

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


TEXT_MODEL_MANIFEST_SCHEMA_VERSION = "text-alpha-manifest-v2"


@dataclass(frozen=True)
class TextModelManifest:
    """Immutable metadata for a promoted paper text source."""

    model_version: str
    feature_set_version: str
    feature_names: tuple[str, ...]
    weights: dict[str, float]
    provider: str
    llm_model: str
    prompt_version: str
    campaign_manifest: str
    source_data_manifest: str
    extraction_manifest: str
    feature_card_hashes: dict[str, str]
    feature_card_paths: dict[str, str]
    created_at: datetime

    @property
    def feature_schema_hash(self) -> str:
        return ordered_feature_schema_hash(self.feature_names)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": TEXT_MODEL_MANIFEST_SCHEMA_VERSION,
            "signal_type": "text",
            "model_version": self.model_version,
            "feature_set_version": self.feature_set_version,
            "feature_names": list(self.feature_names),
            "feature_schema_hash": self.feature_schema_hash,
            "weights": dict(self.weights),
            "provider": self.provider,
            "llm_model": self.llm_model,
            "prompt_version": self.prompt_version,
            "campaign_manifest": self.campaign_manifest,
            "source_data_manifest": self.source_data_manifest,
            "extraction_manifest": self.extraction_manifest,
            "feature_card_hashes": dict(self.feature_card_hashes),
            "feature_card_paths": dict(self.feature_card_paths),
            "created_at": self.created_at.astimezone(UTC).isoformat(),
        }


def write_text_model_manifest(
    *,
    output_root: Path,
    model_version: str,
    feature_set_version: str,
    feature_names: Sequence[str],
    weights: Mapping[str, float],
    provider: str,
    llm_model: str,
    prompt_version: str,
    campaign_manifest: Path,
    source_data_manifest: Path | None,
    extraction_manifest: Path | None = None,
    feature_card_dir: Path | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Write a governed text model manifest and return its path."""
    names = tuple(str(name) for name in feature_names)
    feature_card_paths: dict[str, str] = {}
    feature_card_hashes: dict[str, str] = {}
    if feature_card_dir is not None:
        for name in names:
            card_path = feature_card_dir / f"{name}.json"
            feature_card_paths[name] = str(card_path)
            feature_card_hashes[name] = _sha256_file(card_path) if card_path.is_file() else ""
    manifest = TextModelManifest(
        model_version=model_version,
        feature_set_version=feature_set_version,
        feature_names=names,
        weights={str(name): float(value) for name, value in weights.items()},
        provider=provider,
        llm_model=llm_model,
        prompt_version=prompt_version,
        campaign_manifest=str(campaign_manifest),
        source_data_manifest=str(source_data_manifest) if source_data_manifest else "",
        extraction_manifest=str(extraction_manifest) if extraction_manifest else "",
        feature_card_hashes=feature_card_hashes,
        feature_card_paths=feature_card_paths,
        created_at=created_at or datetime.now(tz=UTC),
    )
    path = output_root / "models" / "text" / model_version / "text_model_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def read_text_model_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("text model manifest must be a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "TEXT_MODEL_MANIFEST_SCHEMA_VERSION",
    "TextModelManifest",
    "read_text_model_manifest",
    "write_text_model_manifest",
]
