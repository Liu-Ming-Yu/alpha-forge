"""Durable registry of auto-promoted formulaic alphas.

The promotion CLI (:mod:`scripts.promote_alphas`) appends one
:class:`PromotedAlphaRecord` per line to a JSONL file. The formulaic
family's package ``__init__`` calls :func:`load_promoted_library` at
import time so the promoted alphas land in the family MANIFEST
alongside the curated starter library — production
``compute_formulaic_features`` sees both sets uniformly.

Why JSONL rather than a database table or a Python module?

* **Append-friendly.** Each promotion run writes new lines; we never
  rewrite the file. Crash-safe across runs.
* **Diffable.** A reviewer can ``git log -p`` the file (when
  checked in) or ``jq``-inspect it ad hoc.
* **Schema-evolution friendly.** ``schema_version`` is pinned per
  line, so old and new records coexist while the loader migrates.

File location
-------------

Default: ``data/parquet/research/alpha_mining/promoted_alphas.jsonl``,
relative to the repository root. Override via
``QUANT_PROMOTED_ALPHAS_PATH`` env var (used by tests + by operators
running on alternative data roots). When the path is unset OR points
at a missing file, :func:`load_promoted_library` returns an empty
tuple — the formulaic family then behaves exactly as if no auto-
promotion has happened yet. That's the bootstrap state and the
default in CI.

Disabling auto-promotion at import time
---------------------------------------

Setting ``QUANT_DISABLE_AUTO_PROMOTED_LIBRARY=1`` forces the loader
to return empty regardless of the file's existence. The existing
formulaic-family tests pin exact feature counts; this flag keeps
them deterministic even on a developer machine that has run the
promotion CLI.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_platform.research.features.formulaic.library import FormulaicAlpha
from quant_platform.research.features.formulaic.serialization import (
    expression_from_dict,
    expression_to_dict,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quant_platform.research.features.formulaic.ast import Expression


PROJECT_ROOT = Path(__file__).resolve().parents[5]

#: Default on-disk location for the promoted-alphas JSONL.
DEFAULT_PROMOTED_PATH: Path = (
    PROJECT_ROOT / "data" / "parquet" / "research" / "alpha_mining" / "promoted_alphas.jsonl"
)

#: Environment variables that influence the loader. ``PATH`` lets an
#: operator point at a non-default file; ``DISABLE`` is the test-
#: friendly "act as if the file doesn't exist" override.
ENV_PROMOTED_PATH: str = "QUANT_PROMOTED_ALPHAS_PATH"
ENV_DISABLE: str = "QUANT_DISABLE_AUTO_PROMOTED_LIBRARY"

#: Schema version pinned into every record. Bumped when the dict
#: shape changes; old records can be migrated by inspecting this
#: field. v1 is the current shape.
RECORD_SCHEMA_VERSION: str = "v1"


@dataclass(frozen=True)
class PromotedAlphaRecord:
    """One row of the promoted-alphas JSONL.

    Attributes
    ----------
    name:
        Content-addressed alpha name (from
        :func:`~.promotion.stable_alpha_id`). Identical expressions
        promoted from different mining runs share the same name.
    expression_payload:
        :func:`~.serialization.expression_to_dict` output for the
        promoted AST. Deserialised at load time via
        :func:`~.serialization.expression_from_dict`.
    description:
        Human-readable provenance string ending up on the generated
        :class:`FeatureSpec`'s description. The promotion CLI
        composes this from a template; default carries the
        mining run's seed, the OOS rank-IC, ICIR, and fold streak.
    promotion_evidence:
        JSON-friendly subset of the evidence used to promote
        (typically the WF metrics). Kept on the record so an
        auditor can re-derive the gate decision later.
    promoted_from_seed:
        Mining-run seed the candidate came from.
    promoted_from_run:
        Optional run identifier — e.g. the JSONL filename. Helps
        track lineage when a single promotion batch draws from
        multiple input files.
    promoted_at:
        UTC ISO timestamp.
    schema_version:
        :data:`RECORD_SCHEMA_VERSION` at write time.
    """

    name: str
    expression_payload: dict[str, Any]
    description: str
    promotion_evidence: dict[str, Any]
    promoted_from_seed: int
    promoted_from_run: str | None = None
    promoted_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = RECORD_SCHEMA_VERSION

    def to_jsonl_line(self) -> str:
        """Return a JSON-serialised single-line representation."""
        return json.dumps(
            {
                "name": self.name,
                "expression_payload": self.expression_payload,
                "description": self.description,
                "promotion_evidence": self.promotion_evidence,
                "promoted_from_seed": self.promoted_from_seed,
                "promoted_from_run": self.promoted_from_run,
                "promoted_at": self.promoted_at,
                "schema_version": self.schema_version,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PromotedAlphaRecord:
        """Construct from a parsed JSONL line. Raises on malformed input."""
        version = payload.get("schema_version", RECORD_SCHEMA_VERSION)
        if version != RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"PromotedAlphaRecord: unsupported schema_version {version!r}; "
                f"this build expects {RECORD_SCHEMA_VERSION!r}"
            )
        name = payload.get("name")
        expression_payload = payload.get("expression_payload")
        description = payload.get("description")
        promotion_evidence = payload.get("promotion_evidence")
        promoted_from_seed = payload.get("promoted_from_seed")
        if (
            not isinstance(name, str)
            or not isinstance(expression_payload, dict)
            or not isinstance(description, str)
            or not isinstance(promotion_evidence, dict)
            or not isinstance(promoted_from_seed, int)
        ):
            raise ValueError(
                f"PromotedAlphaRecord: malformed payload (required fields missing or "
                f"wrong type): {payload!r}"
            )
        return cls(
            name=name,
            expression_payload=expression_payload,
            description=description,
            promotion_evidence=promotion_evidence,
            promoted_from_seed=promoted_from_seed,
            promoted_from_run=payload.get("promoted_from_run"),
            promoted_at=payload.get("promoted_at") or datetime.now(UTC).isoformat(),
            schema_version=version,
        )

    def to_formulaic_alpha(self) -> FormulaicAlpha:
        """Render this record as a :class:`FormulaicAlpha`.

        The expression is deserialised from
        :attr:`expression_payload` via
        :func:`~.serialization.expression_from_dict`. The description
        is passed through unchanged. ``expected_direction`` defaults
        to ``"unknown"`` (the brief's evidence-gated default for
        auto-discovered alphas) and ``larger_is_better=False``.
        """
        expression = expression_from_dict(self.expression_payload)
        return FormulaicAlpha(
            name=self.name,
            expression=expression,
            description=self.description,
            expected_direction="unknown",
            larger_is_better=False,
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def resolve_promoted_path(*, path: Path | None = None) -> Path:
    """Pick the JSONL path to read/write.

    Priority order: explicit ``path`` argument > ``QUANT_PROMOTED_
    ALPHAS_PATH`` env var > :data:`DEFAULT_PROMOTED_PATH`.
    """
    if path is not None:
        return path
    env_path = os.environ.get(ENV_PROMOTED_PATH)
    if env_path:
        return Path(env_path)
    return DEFAULT_PROMOTED_PATH


def load_promoted_library(
    *,
    path: Path | None = None,
    strict: bool = False,
) -> tuple[FormulaicAlpha, ...]:
    """Read the JSONL registry and materialise the promoted alphas.

    Parameters
    ----------
    path:
        Explicit path override. ``None`` (default) consults
        :data:`ENV_PROMOTED_PATH` and falls back to
        :data:`DEFAULT_PROMOTED_PATH`.
    strict:
        When ``False`` (default), malformed lines are warned and
        skipped — useful so one bad line doesn't crash the family
        import. When ``True``, the first malformed line raises.

    Returns
    -------
    tuple[FormulaicAlpha, ...]
        Empty when :data:`ENV_DISABLE` is set, when the resolved
        path is missing, or when the file is empty.

    Notes
    -----
    Duplicate names within the file are de-duplicated, keeping the
    last occurrence (so a re-promotion of the same expression with
    refreshed evidence wins). The dedup is by the record's ``name``
    field, which is content-addressed.
    """
    if os.environ.get(ENV_DISABLE, "").strip() in {"1", "true", "yes"}:
        return ()

    resolved = resolve_promoted_path(path=path)
    if not resolved.exists():
        return ()

    records_by_name: dict[str, PromotedAlphaRecord] = {}
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
                record = PromotedAlphaRecord.from_payload(payload)
            except ValueError as exc:
                _handle_bad_line(line_index, str(exc), strict)
                continue
            records_by_name[record.name] = record  # last-wins dedup

    alphas: list[FormulaicAlpha] = []
    for record in records_by_name.values():
        try:
            alphas.append(record.to_formulaic_alpha())
        except (KeyError, ValueError) as exc:
            _handle_bad_line(0, f"record {record.name!r} could not be materialised: {exc}", strict)
            continue
    return tuple(alphas)


def _handle_bad_line(line_index: int, message: str, strict: bool) -> None:
    text = (
        f"promoted-alphas line {line_index}: {message}"
        if line_index
        else f"promoted-alphas record: {message}"
    )
    if strict:
        raise ValueError(text)
    warnings.warn(text, stacklevel=3)


# ---------------------------------------------------------------------------
# Appender
# ---------------------------------------------------------------------------


def append_promoted_alpha(
    record: PromotedAlphaRecord,
    *,
    path: Path | None = None,
) -> Path:
    """Append a single record to the JSONL registry.

    Creates parent directories as needed. Returns the resolved path
    so the caller can log where the row landed.
    """
    resolved = resolve_promoted_path(path=path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as fh:
        fh.write(record.to_jsonl_line())
        fh.write("\n")
    return resolved


def append_promoted_alphas(
    records: Iterable[PromotedAlphaRecord],
    *,
    path: Path | None = None,
) -> tuple[Path, int]:
    """Append many records in one open. Returns ``(path, count)``."""
    resolved = resolve_promoted_path(path=path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with resolved.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_jsonl_line())
            fh.write("\n")
            count += 1
    return resolved, count


# ---------------------------------------------------------------------------
# Convenience: build a record from an expression + evidence
# ---------------------------------------------------------------------------


def build_record(
    *,
    expression: Expression,
    name: str,
    description: str,
    promotion_evidence: dict[str, Any],
    promoted_from_seed: int,
    promoted_from_run: str | None = None,
) -> PromotedAlphaRecord:
    """Convenience constructor used by the promotion CLI."""
    return PromotedAlphaRecord(
        name=name,
        expression_payload=expression_to_dict(expression),
        description=description,
        promotion_evidence=promotion_evidence,
        promoted_from_seed=promoted_from_seed,
        promoted_from_run=promoted_from_run,
    )


__all__ = [
    "DEFAULT_PROMOTED_PATH",
    "ENV_DISABLE",
    "ENV_PROMOTED_PATH",
    "PromotedAlphaRecord",
    "RECORD_SCHEMA_VERSION",
    "append_promoted_alpha",
    "append_promoted_alphas",
    "build_record",
    "load_promoted_library",
    "resolve_promoted_path",
]
