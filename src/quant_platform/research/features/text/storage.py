"""JSONL persistence for extracted records.

Mirrors the pattern used by the mining + auto-promotion layers: one
JSON object per line, schema-versioned, append-friendly, parseable
line-by-line so streaming consumers (jq, grep, head) work without
loading the whole file.

File location
-------------

Default: ``data/parquet/research/text_events/extractions.jsonl``,
relative to the repository root. Override via
``QUANT_TEXT_EXTRACTIONS_PATH``. The loader returns an empty tuple
when the file is missing, so the family's feature compute can run
on a fresh checkout without raising.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.research.features.text.schemas import ExtractedRecord

if TYPE_CHECKING:
    from collections.abc import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[5]

DEFAULT_EXTRACTIONS_PATH: Path = (
    PROJECT_ROOT / "data" / "parquet" / "research" / "text_events" / "extractions.jsonl"
)

ENV_EXTRACTIONS_PATH: str = "QUANT_TEXT_EXTRACTIONS_PATH"


def resolve_extractions_path(*, path: Path | None = None) -> Path:
    """Pick the JSONL path to read/write.

    Resolution order: explicit ``path`` > ``QUANT_TEXT_EXTRACTIONS_PATH``
    env > :data:`DEFAULT_EXTRACTIONS_PATH`.
    """
    if path is not None:
        return path
    env_value = os.environ.get(ENV_EXTRACTIONS_PATH)
    if env_value:
        return Path(env_value)
    return DEFAULT_EXTRACTIONS_PATH


def load_extracted_records(
    *,
    path: Path | None = None,
    strict: bool = False,
) -> tuple[ExtractedRecord, ...]:
    """Read the JSONL store and materialise records.

    Parameters
    ----------
    path:
        Explicit path override.
    strict:
        When ``False`` (default), malformed lines are warned and
        skipped. When ``True``, the first malformed line raises.

    Returns
    -------
    tuple[ExtractedRecord, ...]
        Empty when the resolved path is missing or the file is
        empty.
    """
    resolved = resolve_extractions_path(path=path)
    if not resolved.exists():
        return ()

    records: list[ExtractedRecord] = []
    line_index = 0
    with resolved.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line_index += 1
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                _handle_bad_line(line_index, str(exc), strict)
                continue
            try:
                record = ExtractedRecord.from_payload(payload)
            except (KeyError, ValueError, TypeError) as exc:
                _handle_bad_line(line_index, str(exc), strict)
                continue
            records.append(record)
    return tuple(records)


def append_extracted_record(
    record: ExtractedRecord,
    *,
    path: Path | None = None,
) -> Path:
    """Append a single record to the JSONL store. Returns the resolved path."""
    resolved = resolve_extractions_path(path=path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as fh:
        fh.write(record.to_jsonl_line())
        fh.write("\n")
    return resolved


def append_extracted_records(
    records: Iterable[ExtractedRecord],
    *,
    path: Path | None = None,
) -> tuple[Path, int]:
    """Append many records in one open. Returns ``(path, count)``."""
    resolved = resolve_extractions_path(path=path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with resolved.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_jsonl_line())
            fh.write("\n")
            count += 1
    return resolved, count


def _handle_bad_line(line_index: int, message: str, strict: bool) -> None:
    text = f"extractions JSONL line {line_index}: {message}"
    if strict:
        raise ValueError(text)
    warnings.warn(text, stacklevel=3)


__all__ = [
    "DEFAULT_EXTRACTIONS_PATH",
    "ENV_EXTRACTIONS_PATH",
    "append_extracted_record",
    "append_extracted_records",
    "load_extracted_records",
    "resolve_extractions_path",
]
