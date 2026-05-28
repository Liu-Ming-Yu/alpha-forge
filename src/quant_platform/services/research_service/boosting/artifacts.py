"""XGBoost sample, manifest, and artifact helpers."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

BoostingDevice = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class BoostingSample:
    """One supervised training row for the XGBoost ranker."""

    as_of: datetime
    instrument_id: uuid.UUID
    features: dict[str, float]
    forward_return: float


@dataclass(frozen=True)
class BoostingTrainConfig:
    """Training parameters for ``train_xgboost_ranker``."""

    model_version: str
    feature_set_version: str
    output_root: Path = Path("data/models/xgboost")
    device: BoostingDevice = "auto"
    require_gpu: bool = False
    validation_fraction: float = 0.20
    purge_days: int = 21
    num_boost_round: int = 100
    early_stopping_rounds: int = 10
    max_depth: int = 4
    eta: float = 0.05
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    min_child_weight: float = 1.0
    random_seed: int = 17
    feature_versions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BoostingManifest:
    """Durable metadata for a trained XGBoost model artifact."""

    model_type: str
    model_version: str
    feature_set_version: str
    booster_path: str
    feature_names: list[str]
    feature_schema_hash: str
    xgboost_version: str
    objective: str
    device: str
    trained_at: str
    metrics_path: str
    booster_sha256: str = ""
    random_seed: int = 17
    metrics: dict[str, object] = field(default_factory=dict)
    feature_versions: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def feature_schema_hash(feature_names: Sequence[str]) -> str:
    """Return a stable hash for the ordered feature list used by XGBoost."""
    payload = json.dumps(list(feature_names), separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def load_samples_async(path: Path) -> list[BoostingSample]:
    """Load supervised ranking samples from JSON without blocking the event loop."""
    import asyncio

    return await asyncio.to_thread(load_samples, path)


def load_samples(path: Path) -> list[BoostingSample]:
    """Load supervised ranking samples from JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("boosting samples file must contain a JSON list")
    samples: list[BoostingSample] = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("each boosting sample must be a JSON object")
        samples.append(
            BoostingSample(
                as_of=datetime.fromisoformat(str(row["as_of"])),
                instrument_id=uuid.UUID(str(row["instrument_id"])),
                features={str(k): float(v) for k, v in dict(row["features"]).items()},
                forward_return=float(row["forward_return"]),
            )
        )
    return samples


def load_manifest(path: Path) -> BoostingManifest:
    """Load and validate a boosting manifest."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    feature_names = list(raw.get("feature_names", []))
    if not feature_names or any(not isinstance(name, str) for name in feature_names):
        raise ValueError("boosting manifest feature_names must be a non-empty string list")
    expected_hash = feature_schema_hash(feature_names)
    if raw.get("feature_schema_hash") != expected_hash:
        raise ValueError("boosting manifest feature_schema_hash does not match feature_names")
    return BoostingManifest(
        model_type=str(raw["model_type"]),
        model_version=str(raw["model_version"]),
        feature_set_version=str(raw["feature_set_version"]),
        booster_path=str(raw["booster_path"]),
        feature_names=feature_names,
        feature_schema_hash=str(raw["feature_schema_hash"]),
        xgboost_version=str(raw["xgboost_version"]),
        objective=str(raw["objective"]),
        device=str(raw["device"]),
        trained_at=str(raw["trained_at"]),
        metrics_path=str(raw["metrics_path"]),
        booster_sha256=str(raw.get("booster_sha256", "")),
        metrics=dict(raw.get("metrics", {})),
        feature_versions={
            str(name): str(version)
            for name, version in dict(raw.get("feature_versions", {})).items()
        },
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
