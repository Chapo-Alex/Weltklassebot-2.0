"""Rotating JSONL append-only store with optional fsync protection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class JsonlStore:
    """Append JSON serialised records to disk with rotation safeguards."""

    __slots__ = (
        "_path",
        "_rotate_lines",
        "_rotate_bytes",
        "_fsync",
        "_line_count",
        "_bytes_written",
        "_index_interval",
        "_index_path",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        rotate_lines: int | None = 100_000,
        rotate_mb: int | None = 64,
        fsync: bool = False,
        index_interval: int | None = 1_000,
    ) -> None:
        self._path = Path(path)
        self._rotate_lines = self._normalise_threshold(rotate_lines)
        self._rotate_bytes = self._normalise_threshold(rotate_mb, multiplier=1024 * 1024)
        self._fsync = fsync
        self._line_count = 0
        self._bytes_written = 0
        self._index_interval = self._normalise_threshold(index_interval)
        self._index_path = self._path.with_name(f"{self._path.name}.index")
        self._refresh_counters()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: Any) -> None:
        line = self._serialise(record)
        payload = line.encode("utf-8")
        self._ensure_parent()
        self._maybe_rotate()
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            if self._fsync:
                os.fsync(handle.fileno())
        self._line_count += 1
        self._bytes_written += len(payload)
        self._maybe_checkpoint_index()

    # Internal helpers -------------------------------------------------
    def _ensure_parent(self) -> None:
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)

    def _maybe_rotate(self) -> None:
        if not self._path.exists():
            self._line_count = 0
            self._bytes_written = 0
            return
        if self._rotate_lines is not None and self._line_count >= self._rotate_lines:
            self._rotate()
            return
        if self._rotate_bytes is not None and self._bytes_written >= self._rotate_bytes:
            self._rotate()

    def _rotate(self) -> None:
        if not self._path.exists():
            self._line_count = 0
            self._bytes_written = 0
            return
        if self._index_path.exists():
            self._index_path.unlink(missing_ok=True)
        index = 1
        while self._rotation_path(index).exists():
            index += 1
        for current in range(index, 0, -1):
            src = self._path if current == 1 else self._rotation_path(current - 1)
            if not src.exists():
                continue
            os.replace(src, self._rotation_path(current))
        self._line_count = 0
        self._bytes_written = 0

    def _rotation_path(self, index: int) -> Path:
        return self._path.with_name(f"{self._path.name}.{index}")

    def _refresh_counters(self) -> None:
        if not self._path.exists():
            self._line_count = 0
            self._bytes_written = 0
            return
        self._bytes_written = self._path.stat().st_size
        if self._index_path.exists():
            try:
                with self._index_path.open("r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                self._line_count = int(meta.get("lines", 0))
                indexed_bytes = int(meta.get("bytes", self._bytes_written))
                if indexed_bytes != self._bytes_written:
                    raise ValueError("byte_mismatch")
                return
            except (OSError, ValueError, json.JSONDecodeError):
                # fall back to a full recount when the index is invalid
                pass
        if self._rotate_lines is None:
            self._line_count = 0
            return
        with self._path.open("r", encoding="utf-8") as handle:
            self._line_count = sum(1 for _ in handle)

    @staticmethod
    def _serialise(record: Any) -> str:
        if isinstance(record, bytes):
            text = record.decode("utf-8")
        elif isinstance(record, str):
            text = record
        else:
            text = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        if not text.endswith("\n"):
            text += "\n"
        return text

    @staticmethod
    def _normalise_threshold(
        value: int | None, *, multiplier: int | None = None
    ) -> int | None:
        if value is None:
            return None
        if value <= 0:
            return None
        if multiplier is None:
            return int(value)
        return int(value) * multiplier

    def _maybe_checkpoint_index(self) -> None:
        if self._index_interval is None:
            return
        if self._line_count % self._index_interval != 0:
            return
        payload = json.dumps(
            {"lines": self._line_count, "bytes": self._bytes_written},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        atomic_write(self._index_path, payload, tmp_suffix=".idx.tmp")


def atomic_write(path: str | Path, data: bytes | str, *, tmp_suffix: str = ".tmp") -> None:
    """Persist ``data`` to ``path`` using an atomic rename."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        payload = data.encode("utf-8")
    else:
        payload = data
    fd, tmp_path = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=tmp_suffix,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


__all__ = ["JsonlStore", "atomic_write"]
