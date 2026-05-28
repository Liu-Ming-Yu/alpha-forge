from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest

from quant_platform.application.operator.cli_inputs import (
    instrument_lookup_from_contracts,
    load_instrument_contracts,
    parse_intraday_decision_times,
    parse_vendor_file,
    symbol_by_instrument_from_contracts,
)


def test_load_instrument_contracts_normalizes_uuid_keys(tmp_path) -> None:
    instrument_id = uuid.uuid4()
    contracts_path = tmp_path / "contracts.json"
    contracts_path.write_text(
        json.dumps({str(instrument_id): {"symbol": "aapl", "last_close": 210.5}}),
        encoding="utf-8",
    )

    contracts = load_instrument_contracts(contracts_path)

    assert contracts == {instrument_id: {"symbol": "aapl", "last_close": 210.5}}


def test_load_instrument_contracts_exits_for_invalid_uuid_key(tmp_path) -> None:
    contracts_path = tmp_path / "contracts.json"
    contracts_path.write_text(json.dumps({"not-a-uuid": {"symbol": "AAPL"}}), encoding="utf-8")

    with pytest.raises(SystemExit, match="invalid UUID key"):
        load_instrument_contracts(contracts_path)


def test_contract_lookup_helpers_include_uuid_and_symbol_aliases() -> None:
    first = uuid.uuid4()
    second = uuid.uuid4()
    contracts = {
        first: {"symbol": "aapl"},
        second: {"symbol": "MSFT"},
        uuid.uuid4(): {"exchange": "SMART"},
    }

    lookup = instrument_lookup_from_contracts(contracts)
    symbols = symbol_by_instrument_from_contracts(contracts)

    assert lookup[str(first)] == first
    assert lookup["AAPL"] == first
    assert lookup["MSFT"] == second
    assert symbols == {first: "AAPL", second: "MSFT"}


def test_parse_vendor_file_returns_trimmed_vendor_and_path() -> None:
    vendor, path = parse_vendor_file(" iex =intraday.csv", option_name="--input")

    assert vendor == "iex"
    assert str(path) == "intraday.csv"


@pytest.mark.parametrize("raw", ["iex.csv", "=/tmp/file.csv", "iex= "])
def test_parse_vendor_file_exits_for_malformed_values(raw: str) -> None:
    with pytest.raises(SystemExit, match="vendor=/path/to/file"):
        parse_vendor_file(raw)


def test_parse_intraday_decision_times_expands_daily_times() -> None:
    start = datetime(2025, 1, 2, 14, 0, tzinfo=UTC)
    end = datetime(2025, 1, 3, 16, 0, tzinfo=UTC)

    decision_times = parse_intraday_decision_times(["15:30"], start, end)

    assert decision_times == (
        datetime(2025, 1, 2, 15, 30, tzinfo=UTC),
        datetime(2025, 1, 3, 15, 30, tzinfo=UTC),
    )


def test_parse_intraday_decision_times_accepts_iso_and_deduplicates() -> None:
    start = datetime(2025, 1, 2, 14, 0)
    end = datetime(2025, 1, 2, 16, 0)

    decision_times = parse_intraday_decision_times(
        ["2025-01-02T15:30:00", "15:30"],
        start,
        end,
    )

    assert decision_times == (datetime(2025, 1, 2, 15, 30, tzinfo=UTC),)


@pytest.mark.parametrize(
    ("raw_values", "message"),
    [
        (["25:30"], "ISO datetime or HH:MM"),
        (["09:30"], "decision times are empty"),
    ],
)
def test_parse_intraday_decision_times_exits_for_invalid_or_out_of_window_values(
    raw_values: list[str],
    message: str,
) -> None:
    start = datetime(2025, 1, 2, 14, 0, tzinfo=UTC)
    end = datetime(2025, 1, 2, 16, 0, tzinfo=UTC)

    with pytest.raises(SystemExit, match=message):
        parse_intraday_decision_times(raw_values, start, end)
