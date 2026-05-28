"""End-to-end tests for ``scripts/migrate_bars_split_frequency.py``.

The script is the one-shot batch migration for legacy mixed-frequency bar
files into the daily/intraday split layout. Most of the work happens
inside ``ParquetBarStore._migrate_legacy_partition_if_present`` -- this
test pins the CLI's discovery, dry-run, and ``--apply`` behaviour and
checks idempotency.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.stores.parquet_store_io import BAR_SCHEMA

if TYPE_CHECKING:
    import pytest

_UTC = UTC

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migrate_bars_split_frequency.py"


def _load_script_module():
    """Import ``scripts/migrate_bars_split_frequency.py`` once per test session."""
    if "migrate_bars_split_frequency" in sys.modules:
        return sys.modules["migrate_bars_split_frequency"]
    spec = importlib.util.spec_from_file_location(
        "migrate_bars_split_frequency",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["migrate_bars_split_frequency"] = mod
    spec.loader.exec_module(mod)
    return mod


def _bar(
    instrument_id: uuid.UUID,
    timestamp: datetime,
    bar_seconds: int,
    close: float,
) -> MarketBar:
    px = Decimal(str(close))
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=instrument_id,
        timestamp=timestamp,
        bar_seconds=bar_seconds,
        open=px,
        high=px,
        low=px,
        close=px,
        volume=1_000,
        vwap=None,
        is_complete=True,
    )


def _seed_legacy_partition(
    root: Path,
    instrument_id: uuid.UUID,
    year: int,
    bars: list[MarketBar],
) -> None:
    path = root / "bars" / str(instrument_id) / f"{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "bar_id": [str(b.bar_id) for b in bars],
        "instrument_id": [str(b.instrument_id) for b in bars],
        "timestamp": [b.timestamp for b in bars],
        "bar_seconds": [b.bar_seconds for b in bars],
        "open": [float(b.open) for b in bars],
        "high": [float(b.high) for b in bars],
        "low": [float(b.low) for b in bars],
        "close": [float(b.close) for b in bars],
        "volume": [b.volume for b in bars],
        "vwap": [float(b.vwap) if b.vwap else None for b in bars],
        "is_complete": [b.is_complete for b in bars],
    }
    pq.write_table(pa.table(arrays, schema=BAR_SCHEMA), path)


def test_find_legacy_files_reports_mixed_partitions(tmp_path: Path) -> None:
    mod = _load_script_module()

    inst_a = uuid.uuid4()
    inst_b = uuid.uuid4()
    _seed_legacy_partition(
        tmp_path,
        inst_a,
        2024,
        [
            _bar(inst_a, datetime(2024, 6, 3, tzinfo=_UTC), 86_400, 100.0),
            _bar(inst_a, datetime(2024, 6, 3, 14, 30, tzinfo=_UTC), 60, 100.05),
        ],
    )
    _seed_legacy_partition(
        tmp_path,
        inst_b,
        2025,
        [_bar(inst_b, datetime(2025, 1, 2, tzinfo=_UTC), 86_400, 200.0)],
    )

    legacy = mod._find_legacy_files(tmp_path, only_instrument=None)
    assert len(legacy) == 2

    mixed = [f for f in legacy if f.daily_rows > 0 and f.intraday_rows > 0]
    assert {f.instrument_id for f in mixed} == {inst_a}


def test_main_dry_run_does_not_touch_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_script_module()
    instrument_id = uuid.uuid4()
    _seed_legacy_partition(
        tmp_path,
        instrument_id,
        2024,
        [_bar(instrument_id, datetime(2024, 6, 3, tzinfo=_UTC), 86_400, 100.0)],
    )
    legacy_path = tmp_path / "bars" / str(instrument_id) / "2024.parquet"

    monkeypatch.setattr(sys, "argv", ["migrate_bars_split_frequency.py", "--root", str(tmp_path)])
    rc = mod.main()
    assert rc == 0
    assert legacy_path.exists()
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "1 legacy files" in out


def test_main_apply_migrates_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_script_module()
    instrument_id = uuid.uuid4()
    _seed_legacy_partition(
        tmp_path,
        instrument_id,
        2024,
        [
            _bar(instrument_id, datetime(2024, 6, 3, tzinfo=_UTC), 86_400, 100.0),
            _bar(instrument_id, datetime(2024, 6, 3, 14, 30, tzinfo=_UTC), 60, 100.05),
        ],
    )
    inst_dir = tmp_path / "bars" / str(instrument_id)
    legacy_path = inst_dir / "2024.parquet"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate_bars_split_frequency.py",
            "--root",
            str(tmp_path),
            "--apply",
        ],
    )
    rc = mod.main()
    assert rc == 0

    assert not legacy_path.exists()
    assert (inst_dir / "daily" / "2024.parquet").exists()
    assert (inst_dir / "intraday" / "2024.parquet").exists()

    # Second run finds nothing; still exits clean.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate_bars_split_frequency.py",
            "--root",
            str(tmp_path),
            "--apply",
        ],
    )
    rc = mod.main()
    assert rc == 0


def test_main_can_target_single_instrument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_script_module()
    target = uuid.uuid4()
    other = uuid.uuid4()
    for inst in (target, other):
        _seed_legacy_partition(
            tmp_path,
            inst,
            2024,
            [_bar(inst, datetime(2024, 6, 3, tzinfo=_UTC), 86_400, 100.0)],
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate_bars_split_frequency.py",
            "--root",
            str(tmp_path),
            "--instrument",
            str(target),
            "--apply",
        ],
    )
    rc = mod.main()
    assert rc == 0

    # Target was migrated; other was not touched.
    assert not (tmp_path / "bars" / str(target) / "2024.parquet").exists()
    assert (tmp_path / "bars" / str(target) / "daily" / "2024.parquet").exists()
    assert (tmp_path / "bars" / str(other) / "2024.parquet").exists()
