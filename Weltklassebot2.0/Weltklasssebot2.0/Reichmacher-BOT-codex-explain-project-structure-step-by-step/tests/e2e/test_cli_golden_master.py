from __future__ import annotations

import json
import sys
from io import StringIO
from typing import Any

from scripts.run_backtest_cli import main as run_cli


def _run_once(monkeypatch) -> dict[str, Any]:
    buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    status = run_cli(
        [
            "--seed",
            "1337",
            "--candles",
            "60",
            "--strategy",
            "breakout",
            "--params",
            "{}",
        ]
    )
    assert status == 0
    payload = buffer.getvalue()
    if not payload:
        raise AssertionError("CLI did not emit payload")
    return json.loads(payload)


def test_cli_digest_stable(monkeypatch) -> None:
    first = _run_once(monkeypatch)
    second = _run_once(monkeypatch)

    assert first == second
    assert isinstance(first["sha256"], str)
    assert len(first["sha256"]) == 64
    assert first["lines"] == 7

    expected_sha = "2ab36c04592da2ab762bb2a627b057fde110fbd7d701020f9e402bd1dd272bc8"
    assert first["sha256"] == expected_sha
