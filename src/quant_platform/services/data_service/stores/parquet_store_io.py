"""Shared Parquet store schemas and atomic-write helpers."""

from __future__ import annotations

import os
import sys
import threading
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

if sys.platform != "win32":
    import fcntl

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

BAR_SCHEMA = pa.schema(
    [
        ("bar_id", pa.string()),
        ("instrument_id", pa.string()),
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("bar_seconds", pa.int32()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.int64()),
        ("vwap", pa.float64()),
        ("is_complete", pa.bool_()),
    ]
)

CA_SCHEMA = pa.schema(
    [
        ("action_id", pa.string()),
        ("instrument_id", pa.string()),
        ("action_type", pa.string()),
        ("ex_date", pa.date32()),
        ("record_date", pa.date32()),
        ("pay_date", pa.date32()),
        # ratio and cash_amount are persisted as strings so a Decimal can
        # round-trip without precision loss (float64 truncates split ratios
        # like 3.0000001 and exact dividend amounts).
        ("ratio", pa.string()),
        ("cash_amount", pa.string()),
        ("currency", pa.string()),
        ("supersedes_id", pa.string()),
        ("notes", pa.string()),
    ]
)

_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[Path, threading.Lock] = {}


def process_lock(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _PROCESS_LOCKS[resolved] = lock
        return lock


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = process_lock(lock_path)
    with lock, lock_path.open("a+b") as handle:
        if sys.platform != "win32":
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except OSError:
                yield
                return
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        else:
            yield


def atomic_write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
