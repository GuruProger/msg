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
        "co2_override": None,
        "notes": {},
    }


class StateStore:
    """JSON state with atomic replacement and permissive migration."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.data = _default_state()
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            # A corrupt state must never prevent the agent from starting.
            self.data = _default_state()
            return
        if not isinstance(raw, dict):
            self.data = _default_state()
            return
        merged = _default_state()
        merged.update(raw)
        if not isinstance(merged.get("handled_message_ids"), list):
            merged["handled_message_ids"] = []
        if not isinstance(merged.get("operator_requests"), dict):
            merged["operator_requests"] = {}
        if not isinstance(merged.get("last_actions"), dict):
            merged["last_actions"] = {}
        if not isinstance(merged.get("notes"), dict):
            merged["notes"] = {}
        self.data = merged

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
        entry = (self.data.get("last_actions") or {}).get(key) or {}
        try:
            last = float(entry.get("at") or 0.0)
        except (TypeError, ValueError):
            last = 0.0
        return last <= 0 or (now - last) >= max(0.0, cooldown_s)

    def mark_action(
        self,
        key: str,
        *,
        success: bool | None = None,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        actions = self.data.setdefault("last_actions", {})
        actions[key] = {"at": float(now), "success": success}
        # Keep the file small and bounded.
        if len(actions) > 200:
            ordered = sorted(
                actions.items(),
                key=lambda item: float((item[1] or {}).get("at") or 0.0),
                reverse=True,
            )[:200]
            self.data["last_actions"] = dict(ordered)
        self.save()

    def cooldown_remaining(self, key: str, cooldown_s: float, *, now: float | None = None) -> float:
        now = time.time() if now is None else now
        entry = (self.data.get("last_actions") or {}).get(key) or {}
        try:
            last = float(entry.get("at") or 0.0)
        except (TypeError, ValueError):
            last = 0.0
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
        requests = self.data.setdefault("operator_requests", {})
        entry = requests.get(key) or {}
        try:
            last = float(entry.get("at") or 0.0)
            previous = str(entry.get("text") or "")
        except (TypeError, ValueError):
            last, previous = 0.0, ""
        if previous == text and last and (now - last) < min_interval_s:
            return False
        requests[key] = {"at": float(now), "text": text}
        if len(requests) > 80:
            ordered = sorted(
                requests.items(),
                key=lambda item: float((item[1] or {}).get("at") or 0.0),
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
