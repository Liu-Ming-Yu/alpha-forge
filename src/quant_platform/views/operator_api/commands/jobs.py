"""Run CLI commands as tracked subprocess jobs.

Each job shells out to ``python -m quant_platform <argv>`` in a background
thread (sync ``subprocess`` rather than asyncio, so it works regardless of the
API's event-loop policy — Windows Selector loops cannot spawn asyncio
subprocesses). Output is captured line-by-line into a bounded buffer the UI
polls with a cursor for live logs.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

_MAX_LOG_LINES = 5000
_MAX_JOBS = 100
_MAX_CONCURRENT = 6
_TERMINAL = ("succeeded", "failed", "cancelled")


@dataclass
class Job:
    id: str
    path: list[str]
    argv: list[str]
    status: str = "queued"  # queued | running | succeeded | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    # Lines evicted from the front of ``logs`` by the ring buffer. The live-tail
    # cursor is absolute (dropped + len), so trimming never freezes the tail.
    dropped: int = 0
    _process: subprocess.Popen[str] | None = None
    _cancelled: bool = False


class JobStore:
    """Thread-safe, bounded store of command jobs."""

    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.RLock()

    def _active_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status not in _TERMINAL)

    def create(self, path: list[str], argv: list[str]) -> Job:
        with self._lock:
            if self._active_count() >= _MAX_CONCURRENT:
                raise RuntimeError(f"too many concurrent jobs (max {_MAX_CONCURRENT})")
            job = Job(id=uuid.uuid4().hex, path=list(path), argv=list(argv))
            self._jobs[job.id] = job
            self._evict_locked()
            return job

    def _evict_locked(self) -> None:
        while len(self._jobs) > _MAX_JOBS:
            for key, job in list(self._jobs.items()):
                if job.status in _TERMINAL:
                    del self._jobs[key]
                    break
            else:
                break

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def append_log(self, job: Job, line: str) -> None:
        with self._lock:
            job.logs.append(line)
            overflow = len(job.logs) - _MAX_LOG_LINES
            if overflow > 0:
                del job.logs[:overflow]
                job.dropped += overflow

    def mark_running(self, job: Job, process: subprocess.Popen[str]) -> None:
        with self._lock:
            job._process = process
            job.status = "running"
            job.started_at = time.time()

    def mark_finished(self, job: Job, exit_code: int) -> None:
        with self._lock:
            job.exit_code = exit_code
            job.finished_at = time.time()
            if job._cancelled:
                job.status = "cancelled"
            else:
                job.status = "succeeded" if exit_code == 0 else "failed"
            job._process = None

    def mark_failed(self, job: Job, error: str) -> None:
        with self._lock:
            job.status = "failed"
            job.error = error
            job.finished_at = time.time()
            job._process = None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in _TERMINAL:
                return False
            job._cancelled = True
            process = job._process
        if process is not None:
            with contextlib.suppress(Exception):  # best-effort kill
                process.terminate()
        return True

    def snapshot(self, job_id: str, *, since: int = 0) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            data = _job_meta(job)
            # ``since`` is an absolute line index. Map it into the retained
            # window via the dropped count so a ring-buffer trim never strands
            # the cursor (older clients that fell behind the window resync from
            # its start). ``log_cursor`` is the absolute total ever produced.
            total = job.dropped + len(job.logs)
            start = min(max(0, since - job.dropped), len(job.logs))
            data["logs"] = list(job.logs[start:])
            data["log_cursor"] = total
            return data

    def list_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [_job_meta(job) for job in reversed(self._jobs.values())]


def _job_meta(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "path": list(job.path),
        "command": " ".join(job.path),
        "argv": list(job.argv),
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "exit_code": job.exit_code,
        "error": job.error,
        "log_lines": job.dropped + len(job.logs),
    }


def start_job(store: JobStore, job: Job) -> None:
    """Launch ``job`` in a background daemon thread."""
    thread = threading.Thread(target=_run, args=(store, job), daemon=True, name=f"job-{job.id[:8]}")
    thread.start()


def _run(store: JobStore, job: Job) -> None:
    cmd = [sys.executable, "-m", "quant_platform", *job.argv]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        process = subprocess.Popen(  # noqa: S603 - argv built from the typed CLI catalog
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001
        store.mark_failed(job, f"failed to launch: {exc}")
        return
    store.mark_running(job, process)
    store.append_log(job, f"$ python -m quant_platform {' '.join(job.argv)}")
    if process.stdout is not None:
        for line in process.stdout:
            store.append_log(job, line.rstrip("\n"))
    exit_code = process.wait()
    store.mark_finished(job, exit_code)


_STORE: JobStore | None = None


def get_job_store() -> JobStore:
    """Return the process-global job store."""
    global _STORE  # noqa: PLW0603 - intentional process-global singleton
    if _STORE is None:
        _STORE = JobStore()
    return _STORE
