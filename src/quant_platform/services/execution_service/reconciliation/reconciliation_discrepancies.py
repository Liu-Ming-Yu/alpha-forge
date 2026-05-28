"""Pure position discrepancy classification for broker reconciliation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.domain.portfolio.positions import AccountSnapshot, PositionSnapshot


class DiscrepancyType(StrEnum):
    POSITION_SIZE_MISMATCH = "position_size_mismatch"
    MISSING_INTERNAL_POSITION = "missing_internal_position"
    EXTRA_INTERNAL_POSITION = "extra_internal_position"
    # Broker reported a position whose conId could not be mapped to any internal
    # instrument_id. This type is set by callers that pre-process broker data
    # and then surface the discrepancy explicitly.
    UNKNOWN_BROKER_CONTRACT = "unknown_broker_contract"


class DiscrepancyResolution(StrEnum):
    AUTO_CORRECTED = "auto_corrected"
    OPERATOR_ACTION_REQUIRED = "operator_action_required"


@dataclass(frozen=True)
class Discrepancy:
    """A single reconciliation discrepancy found during one cycle."""

    discrepancy_id: uuid.UUID
    discrepancy_type: DiscrepancyType
    internal_value: str
    broker_value: str
    resolution: DiscrepancyResolution
    notes: str
    detected_at: datetime
    instrument_id: uuid.UUID | None = None


def classify_position_discrepancies(
    *,
    broker_positions: list[PositionSnapshot],
    internal_snapshot: AccountSnapshot | None,
    detected_at: datetime,
    auto_correct_threshold: int,
) -> list[Discrepancy]:
    """Compare broker and internal positions and classify discrepancies."""

    discrepancies: list[Discrepancy] = []

    broker_map = {p.instrument_id: p for p in broker_positions}
    internal_map: dict[uuid.UUID, PositionSnapshot] = (
        {p.instrument_id: p for p in internal_snapshot.positions} if internal_snapshot else {}
    )

    for instrument_id, broker_pos in broker_map.items():
        internal_pos = internal_map.get(instrument_id)

        if internal_pos is None:
            discrepancies.append(
                Discrepancy(
                    discrepancy_id=uuid.uuid4(),
                    instrument_id=instrument_id,
                    discrepancy_type=DiscrepancyType.MISSING_INTERNAL_POSITION,
                    internal_value="0",
                    broker_value=str(broker_pos.quantity),
                    resolution=DiscrepancyResolution.AUTO_CORRECTED,
                    notes=(
                        "Broker position has no internal counterpart; "
                        "likely a missed fill. Broker state adopted."
                    ),
                    detected_at=detected_at,
                )
            )
            continue

        delta = abs(broker_pos.quantity - internal_pos.quantity)
        if delta == 0:
            continue

        if delta <= auto_correct_threshold:
            discrepancies.append(
                Discrepancy(
                    discrepancy_id=uuid.uuid4(),
                    instrument_id=instrument_id,
                    discrepancy_type=DiscrepancyType.POSITION_SIZE_MISMATCH,
                    internal_value=str(internal_pos.quantity),
                    broker_value=str(broker_pos.quantity),
                    resolution=DiscrepancyResolution.AUTO_CORRECTED,
                    notes=(
                        f"Minor quantity mismatch (delta={delta} <= threshold "
                        f"{auto_correct_threshold}); broker value adopted."
                    ),
                    detected_at=detected_at,
                )
            )
        else:
            discrepancies.append(
                Discrepancy(
                    discrepancy_id=uuid.uuid4(),
                    instrument_id=instrument_id,
                    discrepancy_type=DiscrepancyType.POSITION_SIZE_MISMATCH,
                    internal_value=str(internal_pos.quantity),
                    broker_value=str(broker_pos.quantity),
                    resolution=DiscrepancyResolution.OPERATOR_ACTION_REQUIRED,
                    notes=(
                        f"Position mismatch (delta={delta}) exceeds auto-correct "
                        f"threshold ({auto_correct_threshold}). "
                        "Manual review required before new orders for this instrument."
                    ),
                    detected_at=detected_at,
                )
            )

    for instrument_id, internal_pos in internal_map.items():
        if instrument_id in broker_map:
            continue
        discrepancies.append(
            Discrepancy(
                discrepancy_id=uuid.uuid4(),
                instrument_id=instrument_id,
                discrepancy_type=DiscrepancyType.EXTRA_INTERNAL_POSITION,
                internal_value=str(internal_pos.quantity),
                broker_value="0",
                resolution=DiscrepancyResolution.OPERATOR_ACTION_REQUIRED,
                notes=(
                    "Internal position has no broker counterpart. "
                    "Possible stale state or missed position close. "
                    "Manual review required."
                ),
                detected_at=detected_at,
            )
        )

    return discrepancies
