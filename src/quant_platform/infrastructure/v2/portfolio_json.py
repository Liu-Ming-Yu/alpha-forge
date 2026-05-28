"""JSON adapters for V2 portfolio risk payloads."""

from __future__ import annotations

import json
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.portfolio import StressScenario
from quant_platform.infrastructure.postgres.row_coercion import (
    optional_mapping,
    optional_sequence,
    require_mapping,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def decode_json(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def covariance_to_json(
    covariance: Mapping[tuple[uuid.UUID, uuid.UUID], Decimal],
) -> dict[str, str]:
    return {f"{left}|{right}": str(value) for (left, right), value in covariance.items()}


def json_to_covariance(raw: object) -> dict[tuple[uuid.UUID, uuid.UUID], Decimal]:
    result: dict[tuple[uuid.UUID, uuid.UUID], Decimal] = {}
    for key, value in optional_mapping(decode_json(raw), name="covariance").items():
        left, right = str(key).split("|", 1)
        result[(uuid.UUID(left), uuid.UUID(right))] = Decimal(str(value))
    return result


def factor_exposures_to_json(
    exposures: Mapping[uuid.UUID, Mapping[str, Decimal]],
) -> dict[str, dict[str, str]]:
    return {
        str(instrument_id): {name: str(value) for name, value in factors.items()}
        for instrument_id, factors in exposures.items()
    }


def json_to_factor_exposures(raw: object) -> dict[uuid.UUID, dict[str, Decimal]]:
    return {
        uuid.UUID(str(instrument_id)): {
            str(name): Decimal(str(value))
            for name, value in require_mapping(factors, name="factor_exposures").items()
        }
        for instrument_id, factors in optional_mapping(
            decode_json(raw),
            name="factor_exposures",
        ).items()
    }


def scenarios_to_json(scenarios: tuple[StressScenario, ...]) -> list[dict[str, object]]:
    return [
        {
            "scenario_id": str(scenario.scenario_id),
            "name": scenario.name,
            "shocks": {str(key): str(value) for key, value in scenario.shocks.items()},
        }
        for scenario in scenarios
    ]


def json_to_scenarios(raw: object) -> tuple[StressScenario, ...]:
    result: list[StressScenario] = []
    for raw_item in optional_sequence(decode_json(raw), name="scenarios"):
        item = require_mapping(raw_item, name="scenario")
        shocks: dict[uuid.UUID | str, Decimal] = {}
        for key, value in optional_mapping(item.get("shocks"), name="scenario.shocks").items():
            try:
                parsed_key: uuid.UUID | str = uuid.UUID(str(key))
            except ValueError:
                parsed_key = str(key)
            shocks[parsed_key] = Decimal(str(value))
        result.append(
            StressScenario(
                scenario_id=uuid.UUID(str(item["scenario_id"])),
                name=str(item["name"]),
                shocks=shocks,
            )
        )
    return tuple(result)
