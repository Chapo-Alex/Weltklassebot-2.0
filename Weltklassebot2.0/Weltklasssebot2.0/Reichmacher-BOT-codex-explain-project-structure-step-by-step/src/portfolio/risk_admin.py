"""Administrative helpers for mutating risk manager state files."""

from __future__ import annotations

from typing import Any

from .risk_state_store import JsonlStateStore


def _write(
    store: JsonlStateStore,
    state: str,
    actor: str,
    reason: str,
    minutes: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"state": state, "since": store.now(), "meta": {}}
    if minutes is not None:
        payload["meta"]["cooldown_minutes"] = minutes
    store.save_state(payload)
    store.append_audit(
        {"ts": store.now(), "actor": actor, "action": state.lower(), "reason": reason}
    )
    return payload


def get_state(store: JsonlStateStore) -> dict[str, Any]:
    return store.load_state()


def force_cooldown(
    store: JsonlStateStore, minutes: int, actor: str, reason: str
) -> dict[str, Any]:
    return _write(store, "COOLDOWN", actor, reason, minutes)


def halt(store: JsonlStateStore, actor: str, reason: str) -> dict[str, Any]:
    return _write(store, "HALTED", actor, reason)


def resume(store: JsonlStateStore, actor: str, reason: str) -> dict[str, Any]:
    return _write(store, "RUNNING", actor, reason)


__all__ = ["force_cooldown", "get_state", "halt", "resume"]
