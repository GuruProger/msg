"""Crash-safe JSON Lines logging for the autonomous Dome agent.

The format is defined by ``03_Требования_к_агенту.md``.  Every record has
``ts``, ``run_id``, ``seq`` and ``kind``.  The log is flushed after each line,
so a crash still leaves a usable audit trail.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import redact_mapping

VALID_KINDS = {
    "start",
    "stop",
    "connection",
    "observation",
    "decision",
    "action",
    "verification",
    "message",
    "llm_call",
    "error",
}


class AgentLog:
    """Append-only JSONL writer with redaction and sequence numbers."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        run_id: str | None = None,
        secrets: tuple[str, ...] = (),
    ) -> None:
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._secrets = tuple(secret for secret in secrets if secret)
        self.run_id = run_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        self.path = self._unique_path(self.run_id)
        self.run_id = self.path.stem.removeprefix("agent_")
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        self.seq = 0
        self.closed = False
        self.dome_time: str | None = None

    def _unique_path(self, run_id: str) -> Path:
        candidate = self.log_dir / f"agent_{run_id}.jsonl"
        if not candidate.exists():
            return candidate
        counter = 2
        while True:
            candidate = self.log_dir / f"agent_{run_id}-{counter}.jsonl"
            if not candidate.exists():
                return candidate
            counter += 1

    def set_dome_time(self, value: str | None) -> None:
        self.dome_time = value

    def write(
        self,
        kind: str,
        *,
        ref: int | None = None,
        dome_time: str | None = None,
        **fields: Any,
    ) -> int:
        if self.closed:
            raise ValueError("agent log is closed")
        if kind not in VALID_KINDS:
            raise ValueError(f"unknown log kind: {kind!r}")
        self.seq += 1
        record: dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "seq": self.seq,
            "kind": kind,
        }
        effective_dome_time = dome_time if dome_time is not None else self.dome_time
        if effective_dome_time:
            record["dome_time"] = effective_dome_time
        if ref is not None:
            record["ref"] = ref
        record.update(redact_mapping(fields, extra_secrets=self._secrets))
        line = json.dumps(record, ensure_ascii=False, default=str)
        self._file.write(line + "\n")
        self._file.flush()
        return self.seq

    def flush(self) -> None:
        if not self.closed:
            self._file.flush()

    def close(self) -> None:
        if self.closed:
            return
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except OSError:
            pass
        self._file.close()
        self.closed = True

    def __enter__(self) -> "AgentLog":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
