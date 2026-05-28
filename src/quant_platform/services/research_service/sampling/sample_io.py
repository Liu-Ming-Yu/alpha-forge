"""Supervised research sample loading utilities."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def walk_forward_object_root(object_store_root: Path | str) -> Path:
    """Return the standard durable walk-forward artifact root."""
    return Path(object_store_root) / "research" / "walk_forward"


def load_supervised_samples(path: Path) -> list[SupervisedAlphaSample]:
    """Load shared supervised sample JSON rows.

    The realized-return and label-index fields added in the
    ``backtest-latest-stack-realized-v1`` schema are optional: rows that
    pre-date the change simply lack them and load with ``None`` defaults.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("samples file must contain a JSON list")
    samples: list[SupervisedAlphaSample] = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("each sample row must be a JSON object")
        metadata = _parse_metadata(row.get("metadata"))
        samples.append(
            SupervisedAlphaSample(
                as_of=datetime.fromisoformat(str(row["as_of"])),
                instrument_id=uuid.UUID(str(row["instrument_id"])),
                features={str(k): float(v) for k, v in dict(row["features"]).items()},
                forward_return=float(row["forward_return"]),
                metadata=metadata,
                realized_return_1d=_optional_float(row.get("realized_return_1d")),
                as_of_index=_optional_int(row.get("as_of_index")),
                label_end_index=_optional_int(row.get("label_end_index")),
                label_end_as_of=_optional_datetime(row.get("label_end_as_of")),
            )
        )
    return samples


def _optional_float(raw: object) -> float | None:
    # ``raw`` is ``object`` because it came from ``dict.get(...)`` on a
    # parsed JSON row. We narrow defensively rather than trusting the
    # incoming type — strings like ``"0.01"`` round-trip correctly via
    # ``float(str(...))`` and so do numeric JSON literals.
    if raw is None:
        return None
    return float(str(raw))


def _optional_int(raw: object) -> int | None:
    if raw is None:
        return None
    return int(str(raw))


def _optional_datetime(raw: object) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(str(raw))


def _parse_metadata(raw: object) -> tuple[tuple[str, str], ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        return tuple((str(k), str(v)) for k, v in raw.items())
    if isinstance(raw, list):
        pairs: list[tuple[str, str]] = []
        for item in raw:
            if isinstance(item, list | tuple) and len(item) == 2:
                pairs.append((str(item[0]), str(item[1])))
        return tuple(pairs)
    return ()
