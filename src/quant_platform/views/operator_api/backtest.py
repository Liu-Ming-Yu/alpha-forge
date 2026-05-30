"""Read persisted walk-forward backtest evidence into graphable results.

The latest-stack/campaign runs persist per-arm ``arm_*.json`` evidence under
``{object_store_root}/research/<run-dir>/``. The per-day series are stripped,
but ``folds[]`` is a rich per-fold time series — compounding each fold's
``total_return`` in test-start order yields a real equity curve over a real
date range, alongside per-fold IC / Sharpe / drawdown for the console graphs.
"""

from __future__ import annotations

import json
import statistics
import threading
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings

_MAX_RUNS = 200
_FOLDS_PER_YEAR = 12.0  # ~21-trading-day folds
# backtest.py → operator_api → views → quant_platform → src → <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[4]

# Cache the runs list against a (path, mtime) signature so the polled list
# endpoint re-stats files (cheap) instead of re-reading every JSON each call.
_RunSignature = tuple[tuple[str, int], ...]
_list_cache: tuple[_RunSignature, list[dict[str, Any]]] | None = None
_list_lock = threading.Lock()


def _list_signature(paths: list[Path]) -> _RunSignature:
    out: list[tuple[str, int]] = []
    for path in paths:
        try:
            out.append((str(path), path.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(out)


def _research_root(settings: PlatformSettings) -> Path:
    """Resolve ``<object_store_root>/research``.

    ``object_store_root`` is conventionally relative to the project root. If the
    server's CWD differs (e.g. a dev launcher), fall back to resolving the
    relative root against the repo root so evidence is still found.
    """
    configured = Path(settings.storage.object_store_root)
    primary = configured / "research"
    if primary.is_dir() or configured.is_absolute():
        return primary
    fallback = (_REPO_ROOT / configured / "research").resolve()
    return fallback if fallback.is_dir() else primary


def _num(value: object, default: float = 0.0) -> float:
    if not isinstance(value, (int, float, str, Decimal)):
        return default
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return n if n == n else default  # drop NaN


def _sorted_folds(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    folds = [f for f in (evidence.get("folds") or []) if isinstance(f, dict)]
    return sorted(folds, key=lambda f: str(f.get("test_start") or ""))


def list_backtest_runs(settings: PlatformSettings) -> list[dict[str, Any]]:
    """Summarize every arm evidence file with a fold series, newest first.

    Cached against a (path, mtime) signature so the polled list endpoint does
    cheap stats instead of re-reading every JSON on each request.
    """
    global _list_cache
    root = _research_root(settings)
    if not root.is_dir():
        return []
    paths = sorted(root.glob("*/arm_*.json"))
    signature = _list_signature(paths)
    with _list_lock:
        if _list_cache is not None and _list_cache[0] == signature:
            return _list_cache[1]
    runs: list[dict[str, Any]] = []
    for path in paths:
        try:
            evidence = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001, S112 - skip unreadable/partial files
            continue
        folds = _sorted_folds(evidence)
        if not folds:
            continue
        equity = 1.0
        for fold in folds:
            equity *= 1.0 + _num(fold.get("total_return"))
        metrics = evidence.get("metrics") or {}
        runs.append(
            {
                "id": path.relative_to(root).with_suffix("").as_posix(),
                "arm": evidence.get("arm") or path.stem,
                "category": evidence.get("arm_category"),
                "run_dir": path.parent.name,
                "date_start": str(folds[0].get("test_start") or "")[:10],
                "date_end": str(folds[-1].get("test_end") or "")[:10],
                "n_folds": len(folds),
                "total_return": equity - 1.0,
                "max_drawdown": metrics.get("max_drawdown"),
                "ic_60d": metrics.get("ic_60d"),
                "saved_at": evidence.get("saved_at_utc"),
            }
        )
        if len(runs) >= _MAX_RUNS:
            break
    runs.sort(
        key=lambda r: (str(r.get("date_end") or ""), str(r.get("saved_at") or "")), reverse=True
    )
    with _list_lock:
        _list_cache = (signature, runs)
    return runs


def load_backtest_result(settings: PlatformSettings, run_id: str) -> dict[str, Any] | None:
    """Build the graphable result (equity / drawdown / IC series + metrics)."""
    root = _research_root(settings).resolve()
    try:
        # ``run_id`` is user input: resolving a malformed value (e.g. an embedded
        # null byte) can raise — treat that as "not found", and require the
        # resolved path to stay inside ``root`` (traversal guard).
        target = (root / f"{run_id}.json").resolve()
        if not (target.is_file() and target.is_relative_to(root)):
            return None
        evidence = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    folds = _sorted_folds(evidence)
    points: list[dict[str, Any]] = []
    returns: list[float] = []
    equity = 1.0
    peak = 1.0
    for fold in folds:
        ret = _num(fold.get("total_return"))
        returns.append(ret)
        equity *= 1.0 + ret
        peak = max(peak, equity)
        points.append(
            {
                "date": str(fold.get("test_end") or "")[:10],
                "equity": round(equity, 6),
                "ret": ret,
                "drawdown": round(equity / peak - 1.0, 6),
                "ic": fold.get("mean_ic"),
                "sharpe": fold.get("slippage_adjusted_sharpe"),
                "turnover": fold.get("turnover_avg"),
            }
        )

    metrics = evidence.get("metrics") or {}
    sharpe: float | None = None
    if len(returns) > 1:
        spread = statistics.pstdev(returns)
        if spread > 0:
            sharpe = statistics.mean(returns) / spread * (_FOLDS_PER_YEAR**0.5)
    mean_ic = statistics.mean([_num(f.get("mean_ic")) for f in folds]) if folds else None
    computed_dd = min((p["drawdown"] for p in points), default=None)

    return {
        "id": run_id,
        "arm": evidence.get("arm"),
        "run_id": evidence.get("run_id"),
        "date_start": points[0]["date"] if points else None,
        "date_end": points[-1]["date"] if points else None,
        "points": points,
        "metrics": {
            "total_return": equity - 1.0,
            "sharpe_annualized": sharpe,
            "max_drawdown": metrics.get("max_drawdown", computed_dd),
            "ic_60d": metrics.get("ic_60d"),
            "mean_ic": mean_ic,
            "n_folds": len(folds),
            "fold_negative_ic_streak": metrics.get("fold_negative_ic_streak"),
        },
        "portfolio_config": evidence.get("portfolio_config"),
        "production_candidate": evidence.get("production_candidate"),
    }
