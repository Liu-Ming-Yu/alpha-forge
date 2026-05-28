"""Resolve missing IB ``con_id`` values in a contracts JSON via reqContractDetails.

``scripts/build_universe_curated.py`` emits ``infra/config/universe_300.json``
with every entry's ``con_id`` set to ``0`` because the curated builder has no
broker connectivity. The live/paper bootstrap path
(:mod:`quant_platform.bootstrap.broker.live_broker_wiring`) rejects any non-
positive ``con_id`` (and treats multiple ``0`` values as duplicates) so we need a
one-shot pass against an attached IB Gateway / TWS that resolves the canonical
contract id per symbol and rewrites the file in place.

Usage::

    # Default: paper TWS on 127.0.0.1:7497, rewrite infra/config/universe_300.json
    python scripts/resolve_universe_con_ids.py

    # Dry run: print what would change, do not touch the file
    python scripts/resolve_universe_con_ids.py --dry-run

    # Different file / endpoint
    python scripts/resolve_universe_con_ids.py \\
        --contracts-file infra/config/universe_300.json \\
        --host 127.0.0.1 --port 7497 --client-id 42

Pacing follows the IB market-data subscriber guidance of roughly one historical
/ contract-details request per second. We sleep ``--throttle-seconds`` between
requests; the default 1.0s keeps us comfortably below the 50-msg/sec global
limit even with bursty wrapper traffic.

This script uses ``ibapi`` directly (the same library the production
``execution_service.ib`` adapter wraps).  It supersedes the manual con_id
look-ups that previously seeded ``infra/config/paper_contracts.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    from ibapi.client import EClient
    from ibapi.contract import Contract, ContractDetails
    from ibapi.wrapper import EWrapper
except ImportError as exc:  # pragma: no cover - script-only dep error
    raise SystemExit(
        "ibapi is required; install it into the active virtualenv "
        "(it is already a runtime dep of quant_platform's execution service)."
    ) from exc


# IB pacing: keep contract-details requests at ~1/sec to stay well under the
# global 50-msg/sec ceiling and avoid the 100 simultaneous market-data lines
# cap (contractDetails does not consume a line but errs on the side of safety).
DEFAULT_THROTTLE_SECONDS = 1.0
# Per-request timeout; reqContractDetails should respond within a couple of
# seconds even for fuzzy SMART routing, so 15s is generous.
DEFAULT_REQUEST_TIMEOUT = 15.0
# Connection handshake timeout (matches IBGatewayConnectionLifecycleMixin).
DEFAULT_CONNECT_TIMEOUT = 30.0


class _ContractDetailsClient(EWrapper, EClient):
    """Minimal EClient+EWrapper for reqContractDetails roundtrips.

    The wrapper accumulates ``contractDetails`` callbacks per ``reqId`` and
    signals completion via a per-request ``threading.Event`` populated on
    either ``contractDetailsEnd`` or an unrecoverable error code.
    """

    def __init__(self) -> None:
        EClient.__init__(self, wrapper=self)

        self._lock = threading.Lock()
        self._results: dict[int, list[int]] = {}
        self._errors: dict[int, tuple[int, str]] = {}
        self._done: dict[int, threading.Event] = {}

        self._connect_event = threading.Event()
        self._connect_error: str | None = None
        self._next_req_id = 1

    # ---- connection handshake ----------------------------------------------

    def nextValidId(self, orderId: int) -> None:  # noqa: N802, N803 (IB API name)
        # Use a request-id space well above any IB-allocated orderId to avoid
        # accidental collisions with the order lifecycle.
        self._next_req_id = max(self._next_req_id, orderId + 10_000)
        self._connect_event.set()

    def connectAck(self) -> None:  # noqa: N802
        # Default behavior is fine; just here to silence "no callback" noise.
        return

    # ---- contract details --------------------------------------------------

    def begin(self) -> int:
        with self._lock:
            req_id = self._next_req_id
            self._next_req_id += 1
            self._results[req_id] = []
            self._done[req_id] = threading.Event()
        return req_id

    def contractDetails(  # noqa: N802 (IB API name)
        self,
        reqId: int,  # noqa: N803
        contractDetails: ContractDetails,  # noqa: N803
    ) -> None:
        con_id = int(getattr(contractDetails.contract, "conId", 0) or 0)
        if con_id > 0:
            with self._lock:
                bucket = self._results.get(reqId)
                if bucket is not None:
                    bucket.append(con_id)

    def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802, N803
        with self._lock:
            event = self._done.get(reqId)
        if event is not None:
            event.set()

    # ---- errors ------------------------------------------------------------

    def error(self, reqId: int, *args: object, **_kwargs: object) -> None:  # noqa: N802, N803
        # ibapi's error signature changed in the protobuf-era release; the
        # production wrapper (ib_wrapper/ib_wrapper_error_callbacks.py) has
        # the same normalization. Layouts we accept:
        #   (code, message)
        #   (code, message, advancedOrderRejectJson)
        #   (errorTime, code, message, advancedOrderRejectJson)
        code: int | None = None
        message = ""
        if len(args) == 2:
            raw_code, raw_message = args
        elif len(args) == 3:
            first, second, third = args
            # Protobuf signature: (errorTime, code, message)
            try:
                int(second)  # type: ignore[arg-type]
                second_is_code = True
            except (TypeError, ValueError):
                second_is_code = False
            if second_is_code and isinstance(third, str):
                raw_code, raw_message = second, third
            else:
                raw_code, raw_message = first, second
        elif len(args) >= 4:
            raw_code, raw_message = args[1], args[2]
        else:
            return
        try:
            code = int(raw_code)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        message = str(raw_message) if raw_message is not None else ""

        # Connection-scoped errors carry reqId == -1; capture them so the
        # caller can surface a useful diagnostic.
        if reqId == -1:
            # 2104/2106/2158 are "market data farm OK" info messages.
            if code not in (2104, 2106, 2158, 2107, 2108, 2119):
                self._connect_error = f"IB error {code}: {message}"
            return

        # Per-request hard failures: 200 (no security definition), 321 (bad
        # parameter), 354 (no market data subscription) — treat as "no result"
        # and unblock the waiter.
        with self._lock:
            if reqId in self._done:
                self._errors[reqId] = (code, message)
                self._done[reqId].set()


def _build_contract(spec: dict[str, Any]) -> Contract:
    """Translate a contracts-file entry into an ibapi ``Contract``."""
    contract = Contract()
    contract.symbol = str(spec["symbol"])
    contract.secType = str(spec.get("sec_type", "STK"))
    contract.currency = str(spec.get("currency", "USD"))
    contract.exchange = str(spec.get("exchange", "SMART"))
    primary = spec.get("primary_exchange")
    if isinstance(primary, str) and primary:
        contract.primaryExchange = primary
    return contract


def _issue_request(
    client: _ContractDetailsClient,
    contract: Contract,
    timeout: float,
) -> tuple[list[int], tuple[int, str] | None, bool]:
    """Single reqContractDetails roundtrip. Returns (results, error, timed_out)."""
    req_id = client.begin()
    client.reqContractDetails(req_id, contract)
    event = client._done[req_id]  # noqa: SLF001 (internal coordination)
    timed_out = not event.wait(timeout=timeout)
    with client._lock:  # noqa: SLF001
        results = client._results.pop(req_id, [])  # noqa: SLF001
        err = client._errors.pop(req_id, None)  # noqa: SLF001
        client._done.pop(req_id, None)  # noqa: SLF001
    return results, err, timed_out


def _resolve_one(
    client: _ContractDetailsClient,
    spec: dict[str, Any],
    timeout: float,
) -> tuple[int, str]:
    """Issue a single reqContractDetails; return (con_id, note).

    ``con_id`` is ``0`` on any failure; ``note`` describes the outcome.

    On IB error 200 ("No security definition") we retry once without
    ``primaryExchange`` — the curated builder's primary-venue guesses are
    best-effort and a stale tag (e.g. an issuer that relisted) trips this
    error even though SMART routing alone resolves cleanly.
    """
    contract = _build_contract(spec)
    results, err, timed_out = _issue_request(client, contract, timeout)
    if timed_out:
        return 0, f"timeout after {timeout:.0f}s"

    if err is not None and err[0] == 200 and contract.primaryExchange:
        # Retry without the primary-exchange hint.
        contract.primaryExchange = ""
        results, err, timed_out = _issue_request(client, contract, timeout)
        if timed_out:
            return 0, f"timeout after {timeout:.0f}s (no-primary retry)"
        if not err and results:
            note = "ok (no-primary fallback)"
            if len(results) > 1:
                note = f"ambiguous ({len(results)} matches, no-primary fallback)"
            return results[0], note

    if err is not None:
        code, message = err
        return 0, f"IB error {code}: {message}"
    if not results:
        return 0, "no contractDetails returned"
    if len(results) > 1:
        # SMART routing can return multiple listings for ambiguous symbols
        # (e.g. ADRs or dual-listed names); pick the first and flag it.
        return results[0], f"ambiguous ({len(results)} matches); picked {results[0]}"
    return results[0], "ok"


def _atomic_write_json(path: Path, data: dict[str, dict[str, Any]]) -> None:
    """Write JSON to ``path`` atomically via a sibling .tmp file + os.replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def resolve_universe(
    contracts_path: Path,
    *,
    host: str,
    port: int,
    client_id: int,
    throttle_seconds: float,
    request_timeout: float,
    connect_timeout: float,
    dry_run: bool,
) -> int:
    """Resolve missing con_ids in ``contracts_path``. Returns process exit code."""

    raw = contracts_path.read_text(encoding="utf-8")
    contracts: dict[str, dict[str, Any]] = json.loads(raw)
    if not contracts:
        print(f"No entries in {contracts_path}; nothing to resolve.", file=sys.stderr)
        return 1

    targets = [
        (uid, spec)
        for uid, spec in contracts.items()
        if not (isinstance(spec.get("con_id"), int) and spec["con_id"] > 0)
    ]
    pre_resolved = len(contracts) - len(targets)
    print(
        f"Loaded {len(contracts)} contracts from {contracts_path} "
        f"({pre_resolved} already have con_id; {len(targets)} to resolve)."
    )
    if not targets:
        print("Nothing to do.")
        return 0

    client = _ContractDetailsClient()
    print(f"Connecting to IB at {host}:{port} (client_id={client_id})...")
    client.connect(host, port, client_id)

    reader = threading.Thread(target=client.run, name="ib-contract-reader", daemon=True)
    reader.start()
    try:
        if not client._connect_event.wait(timeout=connect_timeout):  # noqa: SLF001
            err = client._connect_error or "no nextValidId within timeout"  # noqa: SLF001
            print(f"IB handshake failed: {err}", file=sys.stderr)
            return 2

        ok = 0
        skipped: list[tuple[str, str, str]] = []  # (uid, symbol, note)
        ambiguous: list[tuple[str, str, str]] = []
        for idx, (uid, spec) in enumerate(targets, start=1):
            symbol = str(spec.get("symbol", "?"))
            con_id, note = _resolve_one(client, spec, request_timeout)
            if con_id > 0:
                ok += 1
                spec["con_id"] = con_id
                tag = "OK"
                if note.startswith("ambiguous"):
                    ambiguous.append((uid, symbol, note))
                    tag = "AMB"
                print(f"  [{idx:>3}/{len(targets)}] {symbol:<6} {tag:>3} con_id={con_id}  ({note})")
            else:
                skipped.append((uid, symbol, note))
                print(f"  [{idx:>3}/{len(targets)}] {symbol:<6} SKIP  ({note})")

            # Throttle between requests (skip the post-sleep on the final one).
            if idx < len(targets):
                time.sleep(throttle_seconds)
    finally:
        # Best-effort cleanup; if the socket is already torn down the
        # disconnect() may raise, but we don't want that to mask the real result.
        with contextlib.suppress(Exception):
            client.disconnect()

    print()
    print(f"Resolved: {ok}/{len(targets)}")
    if ambiguous:
        print(f"Ambiguous (picked first match): {len(ambiguous)}")
        for _uid, sym, note in ambiguous:
            print(f"  {sym}: {note}")
    if skipped:
        print(f"Skipped: {len(skipped)}")
        for _uid, sym, note in skipped:
            print(f"  {sym}: {note}")

    if dry_run:
        print("\n--dry-run: contracts file NOT modified.")
        return 0

    _atomic_write_json(contracts_path, contracts)
    print(f"\nWrote {contracts_path} (atomic .tmp + rename).")

    remaining = sum(
        1
        for spec in contracts.values()
        if not (isinstance(spec.get("con_id"), int) and spec["con_id"] > 0)
    )
    if remaining:
        print(
            f"WARNING: {remaining} entries still have con_id <= 0; "
            "live/paper validation will fail until those are resolved.",
            file=sys.stderr,
        )
        return 3
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve missing IB con_ids in a contracts JSON via reqContractDetails."
    )
    parser.add_argument(
        "--contracts-file",
        default="infra/config/universe_300.json",
        help="Path to the contracts JSON (default: infra/config/universe_300.json).",
    )
    parser.add_argument("--host", default=os.environ.get("QP__BROKER__HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("QP__BROKER__PORT", "7497")))
    parser.add_argument(
        "--client-id",
        type=int,
        default=int(os.environ.get("QP__BROKER__CLIENT_ID", "77")),
        help="IB API client id; pick something distinct from running sessions.",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=DEFAULT_THROTTLE_SECONDS,
        help="Pause between reqContractDetails calls (IB pacing).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Per-symbol timeout waiting for contractDetailsEnd.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT,
        help="Handshake timeout (nextValidId).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved con_ids and summary but do not rewrite the file.",
    )
    args = parser.parse_args()

    contracts_path = Path(args.contracts_file)
    if not contracts_path.is_file():
        raise SystemExit(f"contracts file not found: {contracts_path}")

    exit_code = resolve_universe(
        contracts_path,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        throttle_seconds=args.throttle_seconds,
        request_timeout=args.request_timeout,
        connect_timeout=args.connect_timeout,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
