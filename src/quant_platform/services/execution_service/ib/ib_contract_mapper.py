"""IB contract mapping helpers.

This module is part of the IBKR adapter boundary: it translates configured
internal instrument metadata into ibapi contract objects and reverse conId maps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ibapi.contract import Contract

from quant_platform.core.exceptions import BrokerSubmissionError

if TYPE_CHECKING:
    import uuid


def build_con_id_mapping(
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
) -> dict[int, uuid.UUID]:
    """Build the canonical IB conId -> internal instrument_id map."""
    mapping: dict[int, uuid.UUID] = {}
    for instrument_id, spec in instrument_contracts.items():
        con_id = spec.get("con_id")
        if isinstance(con_id, int) and con_id > 0:
            mapping[con_id] = instrument_id
    return mapping


def validate_instrument_mappings(
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
) -> list[str]:
    """Return warnings for instruments missing a usable IB conId."""
    warnings: list[str] = []
    for instrument_id, spec in instrument_contracts.items():
        con_id = spec.get("con_id")
        if not (isinstance(con_id, int) and con_id > 0):
            symbol = spec.get("symbol", "<unknown>")
            warnings.append(
                f"instrument_id={instrument_id} (symbol={symbol!r}) has no con_id; "
                "broker positions and fills for this instrument cannot be matched "
                "to the internal ID"
            )
    return warnings


def resolve_contract(
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    instrument_id: uuid.UUID,
) -> Contract:
    """Resolve an internal instrument_id to an IB Contract, failing closed."""
    spec = instrument_contracts.get(instrument_id)
    if spec is None:
        raise BrokerSubmissionError(f"no IB contract mapping for instrument_id={instrument_id}")
    return contract_from_spec(instrument_id, spec)


def contract_from_spec(instrument_id: uuid.UUID, spec: dict[str, object]) -> Contract:
    symbol = spec.get("symbol")
    exchange = spec.get("exchange")
    if not isinstance(symbol, str) or not symbol:
        raise BrokerSubmissionError(f"invalid contract mapping for {instrument_id}: missing symbol")
    if not isinstance(exchange, str) or not exchange:
        raise BrokerSubmissionError(
            f"invalid contract mapping for {instrument_id}: missing exchange"
        )

    contract = Contract()
    contract.secType = str(spec.get("sec_type", "STK"))
    contract.symbol = symbol
    contract.exchange = exchange
    contract.currency = str(spec.get("currency", "USD"))

    primary_exchange = spec.get("primary_exchange")
    if isinstance(primary_exchange, str) and primary_exchange:
        contract.primaryExchange = primary_exchange

    con_id = spec.get("con_id")
    if isinstance(con_id, int) and con_id > 0:
        contract.conId = con_id

    return contract


def contract_con_id(contract: object) -> int:
    """Return a defensively parsed conId from an ibapi contract-like object."""
    return int(getattr(contract, "conId", 0) or 0)
