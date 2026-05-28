"""Configuration helpers for live IBKR smoke tests."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from tests.live import test_ib_gateway_smoke as live_smoke

if TYPE_CHECKING:
    from pathlib import Path


def test_optional_contracts_loads_contracts_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    instrument_id = uuid.uuid4()
    contracts_path = tmp_path / "contracts.json"
    contracts_path.write_text(
        json.dumps(
            {
                str(instrument_id): {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "con_id": 265598,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("QP__LIVE_IBKR__CONTRACTS_FILE", str(contracts_path))
    monkeypatch.delenv("QP__LIVE_IBKR__TEST_SYMBOL", raising=False)
    monkeypatch.delenv("QP__LIVE_IBKR__TEST_CON_ID", raising=False)

    contracts, historical_instrument_id = live_smoke._optional_contracts()

    assert historical_instrument_id is None
    assert contracts == {
        instrument_id: {
            "symbol": "AAPL",
            "exchange": "SMART",
            "currency": "USD",
            "con_id": 265598,
        }
    }


def test_optional_contracts_adds_historical_contract_to_file_contracts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mapped_id = uuid.uuid4()
    contracts_path = tmp_path / "contracts.json"
    contracts_path.write_text(
        json.dumps(
            {
                str(mapped_id): {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "con_id": 265598,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("QP__LIVE_IBKR__CONTRACTS_FILE", str(contracts_path))
    monkeypatch.setenv("QP__LIVE_IBKR__TEST_SYMBOL", "MSFT")
    monkeypatch.setenv("QP__LIVE_IBKR__TEST_CON_ID", "272093")
    monkeypatch.setenv("QP__LIVE_IBKR__TEST_EXCHANGE", "SMART")
    monkeypatch.setenv("QP__LIVE_IBKR__TEST_CURRENCY", "USD")

    contracts, historical_instrument_id = live_smoke._optional_contracts()

    assert historical_instrument_id is not None
    assert contracts[mapped_id]["con_id"] == 265598
    assert contracts[historical_instrument_id]["symbol"] == "MSFT"
    assert contracts[historical_instrument_id]["con_id"] == 272093


def test_optional_contracts_reuses_file_id_for_duplicate_historical_con_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mapped_id = uuid.uuid4()
    contracts_path = tmp_path / "contracts.json"
    contracts_path.write_text(
        json.dumps(
            {
                str(mapped_id): {
                    "symbol": "AAPL",
                    "exchange": "SMART",
                    "currency": "USD",
                    "con_id": 265598,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("QP__LIVE_IBKR__CONTRACTS_FILE", str(contracts_path))
    monkeypatch.setenv("QP__LIVE_IBKR__TEST_SYMBOL", "AAPL")
    monkeypatch.setenv("QP__LIVE_IBKR__TEST_CON_ID", "265598")
    monkeypatch.setenv("QP__LIVE_IBKR__TEST_EXCHANGE", "SMART")
    monkeypatch.setenv("QP__LIVE_IBKR__TEST_CURRENCY", "USD")

    contracts, historical_instrument_id = live_smoke._optional_contracts()

    assert historical_instrument_id == mapped_id
    assert list(contracts) == [mapped_id]
