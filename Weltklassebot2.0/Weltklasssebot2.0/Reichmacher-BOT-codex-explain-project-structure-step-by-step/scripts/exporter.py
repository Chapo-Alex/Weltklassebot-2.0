"""Expose Weltklassebot metrics via a Prometheus scrape endpoint."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from threading import Event
from types import FrameType

from core.metrics import LAT, get_registry, start_http_server

_HEARTBEAT_INTERVAL = 30.0


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _log(level: str, message: str) -> None:
    clean = message.replace("\n", " ")
    sys.stderr.write(f"{_timestamp()} level={level} msg=\"{clean}\"\n")
    sys.stderr.flush()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--addr",
        default="0.0.0.0",
        help="Address to bind the exporter to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port to expose Prometheus metrics on",
    )
    parser.add_argument(
        "--grace-ms",
        type=int,
        default=500,
        help="Grace period in milliseconds before shutting down",
    )
    return parser.parse_args(argv)


def _start_server(port: int, addr: str, registry: object | None) -> None:
    try:
        start_http_server(port, addr=addr, registry=registry)
    except TypeError as exc:
        if "addr" not in str(exc):
            raise
        start_http_server(port, registry=registry)


def _install_signal_handlers(stop_event: Event) -> None:
    def _handle(signum: int, frame: FrameType | None) -> None:
        del frame
        _log("INFO", f"received signal signum={signum}")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(ValueError):
            signal.signal(sig, _handle)


def _serve(args: argparse.Namespace) -> int:
    registry = get_registry()
    try:
        _start_server(args.port, args.addr, registry)
    except Exception as exc:  # noqa: BLE001 - need clean exit path
        _log(
            "ERROR",
            f"failed to start exporter addr={args.addr} port={args.port} error={exc}",
        )
        return 2

    stop_event = Event()
    _install_signal_handlers(stop_event)
    _log("INFO", f"exporter listening addr={args.addr} port={args.port}")

    heartbeat_enabled = os.getenv("WELTKLASSE_EXPORTER_TICK") == "1"
    if heartbeat_enabled:
        while not stop_event.wait(_HEARTBEAT_INTERVAL):
            LAT.labels(stage="exporter_heartbeat").observe(0.0)
    else:
        stop_event.wait()

    grace_seconds = max(args.grace_ms, 0) / 1000.0
    if grace_seconds:
        time.sleep(grace_seconds)
    _log("INFO", "exporter stopped")
    return 0


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return _serve(args)


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))


if __name__ == "__main__":
    main()
