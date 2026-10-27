"""Small persistent state file for restart-safe agent behaviour.

The agent must survive its own restart.  We persist only durable facts:
the operator task, the last chat message seen, the selected CO2 source, and
short-lived action/notification guards.  Command queues are intentionally NOT
persisted: after reconnect the agent re-reads telemetry and decides again.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def _default_state() -> dict[str, Any]:
    return {
        "version": 1,
        "last_message_id": 0,
        "handled_message_ids": [],
        "task": None,
        "operator_requests": {},
        "last_actions": {},
        "failed_until": {},
        "co2_override": None,
        "notes": {},
    }


def _as_epoch(value: Any) -> float:
    """Convert a persisted timestamp to float, tolerating malformed values."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _action_entry(actions: Any, key: str) -> dict[str, Any]:
    """Return a valid action entry from a possibly malformed mapping."""
    if not isinstance(actions, dict):
        return {}
    entry = actions.get(key)
    return entry if isinstance(entry, dict) else {}


class StateStore:
    """JSON state with atomic replacement and permissive migration."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.data = _default_state()
        self.load_error: str | None = None
        self.corrupt_path: Path | None = None
        self.load()

    def load(self) -> None:
        self.load_error = None
        self.corrupt_path = None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            # A corrupt state must never prevent the agent from starting.
            self.data = _default_state()
            self.load_error = f"{type(exc).__name__}: {exc}"
            self._preserve_corrupt()
            return
        if not isinstance(raw, dict):
            self.data = _default_state()
            self.load_error = "invalid top-level state value"
            self._preserve_corrupt()
            return
        merged = _default_state()
        merged.update(raw)
        if not isinstance(merged.get("handled_message_ids"), list):
            merged["handled_message_ids"] = []
        if not isinstance(merged.get("operator_requests"), dict):
            merged["operator_requests"] = {}
        if not isinstance(merged.get("last_actions"), dict):
            merged["last_actions"] = {}
        if not isinstance(merged.get("failed_until"), dict):
            merged["failed_until"] = {}
        if not isinstance(merged.get("notes"), dict):
            merged["notes"] = {}
        self.data = merged

    def _preserve_corrupt(self) -> None:
        """Keep a private copy of an unreadable state file for recovery."""
        target = self.path.with_name(
            f"{self.path.name}.corrupt-{time.time_ns()}"
        )
        try:
            shutil.copy2(self.path, target)
            try:
                os.chmod(target, 0o600)
            except OSError:
                pass
        except OSError:
            return
        self.corrupt_path = target

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        payload = json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True)
        temp.write_text(payload + "\n", encoding="utf-8")
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        os.replace(temp, self.path)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any, *, save: bool = True) -> None:
        self.data[key] = value
        if save:
            self.save()

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    # --- task ---------------------------------------------------------

    @property
    def task(self) -> dict[str, Any] | None:
        task = self.data.get("task")
        return task if isinstance(task, dict) else None

    def set_task(self, task: dict[str, Any] | None) -> None:
        self.data["task"] = copy.deepcopy(task) if task else None
        self.save()

    def update_task(self, **changes: Any) -> dict[str, Any]:
        task = copy.deepcopy(self.task or {})
        task.update(changes)
        self.data["task"] = task
        self.save()
        return task

    def clear_task(self) -> None:
        self.data["task"] = None
        self.save()

    # --- chat ---------------------------------------------------------

    @property
    def last_message_id(self) -> int:
        try:
            return max(0, int(self.data.get("last_message_id") or 0))
        except (TypeError, ValueError):
            return 0

    def mark_message(self, message_id: int) -> None:
        if message_id <= 0:
            return
        self.data["last_message_id"] = max(self.last_message_id, int(message_id))
        handled = [
            int(value)
            for value in self.data.get("handled_message_ids") or []
            if isinstance(value, (int, float, str)) and str(value).lstrip("-").isdigit()
        ]
        if message_id not in handled:
            handled.append(int(message_id))
        self.data["handled_message_ids"] = handled[-300:]
        self.save()

    def seen_message(self, message_id: int) -> bool:
        if message_id <= self.last_message_id:
            return True
        handled = {int(value) for value in self.data.get("handled_message_ids") or []}
        return int(message_id) in handled

    # --- action / notification guards --------------------------------

    def action_allowed(self, key: str, cooldown_s: float, *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        entry = _action_entry(self.data.get("last_actions"), key)
        last = _as_epoch(entry.get("at"))
        return last <= 0 or (now - last) >= max(0.0, cooldown_s)

    def mark_action(
        self,
        key: str,
        *,
        success: bool | None = None,
        now: float | None = None,
        failure_cooldown_s: float = 0.0,
    ) -> None:
        now = time.time() if now is None else now
        actions = self.data.get("last_actions")
        if not isinstance(actions, dict):
            actions = {}
            self.data["last_actions"] = actions
        actions[key] = {"at": float(now), "success": success}
        failures = self.data.get("failed_until")
        if not isinstance(failures, dict):
            failures = {}
            self.data["failed_until"] = failures
        if success is True:
            failures.pop(str(key), None)
        elif success is False and failure_cooldown_s > 0:
            failures[str(key)] = float(now) + max(0.0, float(failure_cooldown_s))
            self._prune_failed_until()
        # Keep the file small and bounded.
        if len(actions) > 200:
            ordered = sorted(
                actions.items(),
                key=lambda item: (
                    _as_epoch(item[1].get("at"))
                    if isinstance(item[1], dict)
                    else 0.0
                ),
                reverse=True,
            )[:200]
            self.data["last_actions"] = dict(ordered)
        self.save()

    def mark_action_failure(
        self,
        key: str,
        cooldown_s: float,
        *,
        now: float | None = None,
    ) -> None:
        """Remember that ``key`` recently failed verification."""
        now = time.time() if now is None else now
        failures = self.data.get("failed_until")
        if not isinstance(failures, dict):
            failures = {}
            self.data["failed_until"] = failures
        failures[str(key)] = float(now) + max(0.0, float(cooldown_s))
        self._prune_failed_until()
        self.save()

    def action_failed_recently(self, key: str, *, now: float | None = None) -> bool:
        """Return True while the failure cooldown for ``key`` is still active."""
        now = time.time() if now is None else now
        failures = self.data.get("failed_until")
        if not isinstance(failures, dict):
            return False
        return _as_epoch(failures.get(str(key))) > now

    def action_succeeded(self, key: str) -> bool:
        """Return True only when ``key`` was last recorded as successful."""
        entry = _action_entry(self.data.get("last_actions"), str(key))
        return bool(entry.get("success") is True)

    def failure_remaining(self, key: str, *, now: float | None = None) -> float:
        """Seconds left before a failed action may be retried."""
        now = time.time() if now is None else now
        failures = self.data.get("failed_until")
        if not isinstance(failures, dict):
            return 0.0
        return max(0.0, _as_epoch(failures.get(str(key))) - now)

    def _prune_failed_until(self) -> None:
        failures = self.data.get("failed_until")
        if not isinstance(failures, dict):
            self.data["failed_until"] = {}
            return
        if len(failures) <= 200:
            return
        ordered = sorted(
            failures.items(),
            key=lambda item: _as_epoch(item[1]),
            reverse=True,
        )[:200]
        self.data["failed_until"] = dict(ordered)

    def cooldown_remaining(self, key: str, cooldown_s: float, *, now: float | None = None) -> float:
        now = time.time() if now is None else now
        entry = _action_entry(self.data.get("last_actions"), key)
        last = _as_epoch(entry.get("at"))
        return max(0.0, cooldown_s - (now - last)) if last else 0.0

    def should_notify(
        self,
        key: str,
        text: str,
        *,
        min_interval_s: float = 900.0,
        now: float | None = None,
    ) -> bool:
        """Return true when this exact operator notification is due again."""
        now = time.time() if now is None else now
        requests = self.data.get("operator_requests")
        if not isinstance(requests, dict):
            requests = {}
            self.data["operator_requests"] = requests
        entry = requests.get(key)
        entry = entry if isinstance(entry, dict) else {}
        last = _as_epoch(entry.get("at"))
        previous = str(entry.get("text") or "")
        if previous == text and last and (now - last) < min_interval_s:
            return False
        requests[key] = {"at": float(now), "text": text}
        if len(requests) > 80:
            ordered = sorted(
                requests.items(),
                key=lambda item: (
                    _as_epoch(item[1].get("at"))
                    if isinstance(item[1], dict)
                    else 0.0
                ),
                reverse=True,
            )[:80]
            self.data["operator_requests"] = dict(ordered)
        self.save()
        return True

    # --- CO2 primary override ----------------------------------------

    @property
    def co2_override(self) -> dict[str, Any] | None:
        value = self.data.get("co2_override")
        return value if isinstance(value, dict) else None

    def set_co2_override(self, value: dict[str, Any] | None) -> None:
        self.data["co2_override"] = copy.deepcopy(value) if value else None
        self.save()
