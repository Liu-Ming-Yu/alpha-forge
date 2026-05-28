"""Tests for the sector-mapping preflight on session construction."""

from __future__ import annotations

import uuid

import pytest

from quant_platform.config import PlatformSettings, RiskSettings
from quant_platform.core.domain.instruments import AssetClass, Instrument
from quant_platform.services.data_service.reference.contract_master import ContractMaster
from quant_platform.session import _run_sector_mapping_preflight


def _instrument(symbol: str, sector: str | None) -> Instrument:
    return Instrument(
        instrument_id=uuid.uuid4(),
        symbol=symbol,
        exchange="XNAS",
        currency="USD",
        asset_class=AssetClass.EQUITY,
        lot_size=1,
        sector=sector,
        active=True,
    )


def _settings(*, require: bool) -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        risk=RiskSettings(require_sector_mapping=require),
    )


def test_preflight_silent_when_all_mapped() -> None:
    cm = ContractMaster(
        [
            _instrument("AAPL", "Information Technology"),
            _instrument("XOM", "Energy"),
        ]
    )
    _run_sector_mapping_preflight(cm, cm.sector_map(), _settings(require=True))


def test_preflight_raises_when_strict_and_unmapped() -> None:
    cm = ContractMaster(
        [
            _instrument("AAPL", "Information Technology"),
            _instrument("MYSTERY", None),
        ]
    )
    with pytest.raises(ValueError, match="require_sector_mapping=True"):
        _run_sector_mapping_preflight(cm, cm.sector_map(), _settings(require=True))


def test_preflight_warns_but_allows_when_not_strict() -> None:
    cm = ContractMaster(
        [
            _instrument("AAPL", "Information Technology"),
            _instrument("MYSTERY", None),
        ]
    )
    _run_sector_mapping_preflight(cm, cm.sector_map(), _settings(require=False))


def test_preflight_surfaces_truncated_symbol_list_for_large_gaps() -> None:
    missing = [_instrument(f"SYM{i}", None) for i in range(20)]
    cm = ContractMaster(missing + [_instrument("AAPL", "IT")])
    with pytest.raises(ValueError) as excinfo:
        _run_sector_mapping_preflight(cm, cm.sector_map(), _settings(require=True))
    assert "..." in str(excinfo.value)
