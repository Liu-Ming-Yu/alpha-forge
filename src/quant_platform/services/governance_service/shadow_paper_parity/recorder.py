"""Shadow-vs-paper parity evidence producer."""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

from quant_platform.core.domain.production import ShadowPaperParityRecord

if TYPE_CHECKING:
    from datetime import datetime

_PARITY_NAMESPACE = uuid.UUID("0f425dfa-733f-4787-a82e-496b4dd1d944")


class ShadowPaperParitySink(Protocol):
    async def save_shadow_paper_parity(self, record: ShadowPaperParityRecord) -> None: ...


@dataclass(frozen=True)
class ShadowPaperParityRecorder:
    """Build and optionally persist shadow-vs-paper parity observations."""

    repository: ShadowPaperParitySink | None = None

    def build_record(
        self,
        *,
        as_of: datetime,
        shadow_targets: object,
        paper_targets: object,
        shadow_order_plan: object,
        paper_order_plan: object,
        instrument_universe: Iterable[object],
        signal_name: str = "text",
        signal_type: str = "text",
        shadow_run_id: object = "",
        paper_run_id: object = "",
        git_commit: str = "",
        config_hash: str = "",
        text_model_manifest_sha256: str = "",
        feature_schema_hash: str = "",
        source_weights: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ShadowPaperParityRecord:
        shadow_weights = _target_weights(shadow_targets)
        paper_weights = _target_weights(paper_targets)
        shadow_sides = _order_sides(shadow_order_plan)
        paper_sides = _order_sides(paper_order_plan)
        universe = _instrument_universe(
            instrument_universe,
            shadow_weights=shadow_weights,
            paper_weights=paper_weights,
            shadow_sides=shadow_sides,
            paper_sides=paper_sides,
        )

        missing = len(universe) if shadow_weights is None or paper_weights is None else 0
        max_diff = _max_target_weight_diff_bps(
            universe,
            shadow_weights or {},
            paper_weights or {},
        )
        side_mismatches = _order_side_mismatches(universe, shadow_sides, paper_sides)
        trading_day = as_of.date()
        base_metadata: dict[str, object] = {
            "shadow_run_id": str(shadow_run_id) if shadow_run_id else "",
            "paper_run_id": str(paper_run_id) if paper_run_id else "",
            "git_commit": git_commit,
            "config_hash": config_hash,
            "text_model_manifest_sha256": text_model_manifest_sha256,
            "feature_schema_hash": feature_schema_hash,
            "source_weights": dict(source_weights or {}),
        }
        if metadata:
            base_metadata.update(dict(metadata))
        return ShadowPaperParityRecord(
            parity_id=uuid.uuid5(
                _PARITY_NAMESPACE,
                f"{signal_type}|{signal_name}|{trading_day.isoformat()}",
            ),
            signal_name=signal_name,
            signal_type=signal_type,
            trading_day=trading_day,
            as_of=as_of,
            instruments_compared=len(universe),
            missing_instruments=missing,
            max_target_weight_diff_bps=max_diff,
            order_side_mismatches=side_mismatches,
            metadata=base_metadata,
        )

    async def record(
        self,
        *,
        as_of: datetime,
        shadow_targets: object,
        paper_targets: object,
        shadow_order_plan: object,
        paper_order_plan: object,
        instrument_universe: Iterable[object],
        signal_name: str = "text",
        signal_type: str = "text",
        shadow_run_id: object = "",
        paper_run_id: object = "",
        git_commit: str = "",
        config_hash: str = "",
        text_model_manifest_sha256: str = "",
        feature_schema_hash: str = "",
        source_weights: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ShadowPaperParityRecord:
        parity_record = self.build_record(
            as_of=as_of,
            shadow_targets=shadow_targets,
            paper_targets=paper_targets,
            shadow_order_plan=shadow_order_plan,
            paper_order_plan=paper_order_plan,
            instrument_universe=instrument_universe,
            signal_name=signal_name,
            signal_type=signal_type,
            shadow_run_id=shadow_run_id,
            paper_run_id=paper_run_id,
            git_commit=git_commit,
            config_hash=config_hash,
            text_model_manifest_sha256=text_model_manifest_sha256,
            feature_schema_hash=feature_schema_hash,
            source_weights=source_weights,
            metadata=metadata,
        )
        if self.repository is not None:
            await self.repository.save_shadow_paper_parity(parity_record)
        return parity_record


def _target_weights(target: object) -> dict[uuid.UUID, Decimal] | None:
    if target is None:
        return None
    raw = getattr(target, "weights", None)
    if raw is None and isinstance(target, Mapping):
        raw = target.get("weights")
        if raw is None:
            raw = target
    if not isinstance(raw, Mapping):
        raise TypeError("target weights must be a mapping or an object with .weights")
    return {_instrument_id(key): _decimal(value) for key, value in raw.items()}


def _order_sides(order_plan: object) -> dict[uuid.UUID, frozenset[str]]:
    raw_items = _order_items(order_plan)
    sides: defaultdict[uuid.UUID, set[str]] = defaultdict(set)
    for item in raw_items:
        instrument_id = _field(item, "instrument_id")
        side = _field(item, "side")
        if instrument_id is None or side is None:
            continue
        side_value = getattr(side, "value", side)
        side_name = str(side_value).lower().strip()
        if side_name:
            sides[_instrument_id(instrument_id)].add(side_name)
    return {instrument_id: frozenset(values) for instrument_id, values in sides.items()}


def _order_items(order_plan: object) -> Sequence[object]:
    if order_plan is None:
        return ()
    for name in ("orders", "intents", "approved", "order_intents"):
        value = getattr(order_plan, name, None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return value
    if isinstance(order_plan, Mapping):
        for name in ("orders", "intents", "approved", "order_intents"):
            value = order_plan.get(name)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return value
        return tuple(order_plan.values())
    if isinstance(order_plan, Sequence) and not isinstance(order_plan, (str, bytes)):
        return order_plan
    raise TypeError("order plan must be a sequence or an object/mapping containing orders")


def _field(item: object, name: str) -> object | None:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _instrument_universe(
    instrument_universe: Iterable[object],
    *,
    shadow_weights: Mapping[uuid.UUID, Decimal] | None,
    paper_weights: Mapping[uuid.UUID, Decimal] | None,
    shadow_sides: Mapping[uuid.UUID, frozenset[str]],
    paper_sides: Mapping[uuid.UUID, frozenset[str]],
) -> tuple[uuid.UUID, ...]:
    universe = {_instrument_id(item) for item in instrument_universe}
    if not universe:
        for mapping in (shadow_weights or {}, paper_weights or {}, shadow_sides, paper_sides):
            universe.update(mapping)
    return tuple(sorted(universe, key=str))


def _max_target_weight_diff_bps(
    universe: Iterable[uuid.UUID],
    shadow_weights: Mapping[uuid.UUID, Decimal],
    paper_weights: Mapping[uuid.UUID, Decimal],
) -> float:
    max_diff = Decimal("0")
    for instrument_id in universe:
        diff = abs(
            shadow_weights.get(instrument_id, Decimal("0"))
            - paper_weights.get(instrument_id, Decimal("0"))
        )
        max_diff = max(max_diff, diff * Decimal("10000"))
    return float(max_diff)


def _order_side_mismatches(
    universe: Iterable[uuid.UUID],
    shadow_sides: Mapping[uuid.UUID, frozenset[str]],
    paper_sides: Mapping[uuid.UUID, frozenset[str]],
) -> int:
    return sum(
        1
        for instrument_id in universe
        if shadow_sides.get(instrument_id, frozenset())
        != paper_sides.get(instrument_id, frozenset())
    )


def _instrument_id(value: object) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _decimal(value: object) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


__all__ = ["ShadowPaperParityRecorder", "ShadowPaperParitySink"]
