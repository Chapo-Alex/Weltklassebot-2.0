from __future__ import annotations

import json
from pathlib import Path

from state import JsonlStore


def _load_records(path: Path) -> list[dict[str, int]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_rotation_preserves_order(tmp_path: Path) -> None:
    audit_path = tmp_path / "risk_audit.jsonl"
    store = JsonlStore(audit_path, rotate_lines=2, rotate_mb=None, fsync=False)

    for idx in range(6):
        store.append({"idx": idx})

    current = _load_records(audit_path)
    rotated_first = _load_records(audit_path.with_name("risk_audit.jsonl.1"))
    rotated_second = _load_records(audit_path.with_name("risk_audit.jsonl.2"))

    assert [entry["idx"] for entry in current] == [4, 5]
    assert [entry["idx"] for entry in rotated_first] == [2, 3]
    assert [entry["idx"] for entry in rotated_second] == [0, 1]
    assert not audit_path.with_name("risk_audit.jsonl.3").exists()

    reconstructed = [*rotated_second, *rotated_first, *current]
    assert [entry["idx"] for entry in reconstructed] == list(range(6))
