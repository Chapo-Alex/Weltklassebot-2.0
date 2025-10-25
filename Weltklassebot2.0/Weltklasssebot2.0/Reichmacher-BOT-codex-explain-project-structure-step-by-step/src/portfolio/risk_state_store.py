"""Filesystem-backed state persistence for the risk manager."""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from state import JsonlStore


@dataclass(slots=True)
class JsonlStateStore:
    """Persist risk state snapshots and audit entries atomically."""

    dir: Path
    state_name: str = "risk_state.json"
    audit_name: str = "risk_audit.jsonl"
    rotate_lines: int | None = 100_000
    rotate_mb: int | None = 64
    fsync: bool = False
    _audit_store: JsonlStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.dir = Path(self.dir)
        self._audit_store = JsonlStore(
            self.audit_path,
            rotate_lines=self.rotate_lines,
            rotate_mb=self.rotate_mb,
            fsync=self.fsync,
        )

    @property
    def state_path(self) -> Path:
        return self.dir / self.state_name

    @property
    def audit_path(self) -> Path:
        return self.dir / self.audit_name

    def load_state(self) -> dict[str, Any]:
        path = self.state_path
        if not path.exists():
            return {"state": "RUNNING", "since": 0.0, "meta": {}}
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            msg = "invalid risk state payload"
            raise ValueError(msg)
        return cast(dict[str, Any], data)

    def save_state(self, state: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        fd, tmp_path = tempfile.mkstemp(dir=self.dir, prefix=".state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.state_path)
        finally:
            with suppress(FileNotFoundError):
                os.remove(tmp_path)

    def append_audit(self, event: dict[str, Any]) -> None:
        self._audit_store.append(event)

    @staticmethod
    def now() -> float:
        return time.time()


__all__ = ["JsonlStateStore"]
