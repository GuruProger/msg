"""Append-only audit log for the advisory panel conversation.

The agent already writes ``logs/agent_*.jsonl``.  The panel is a separate
process and used to keep its chat only in memory, which made an appeal harder.
This writer stores the minimum needed to reconstruct a conversation: question,
answer, engine, verdict and cited sources.  It never stores the LLM prompt or
telemetry context.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class PanelLog:
    def __init__(
        self,
        log_dir: str | Path,
        *,
        redact: Callable[[Any], Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        path = self.log_dir / f"panel_{self.run_id}.jsonl"
        counter = 2
        while True:
            try:
                path.touch(exist_ok=False)
                break
            except FileExistsError:
                path = self.log_dir / f"panel_{self.run_id}-{counter}.jsonl"
                counter += 1
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        self.path = path
        self._redact = redact
        self._file = None
        self.seq = 0
        self.closed = False

    def _open(self):
        if self._file is None:
            self._file = self.path.open("a", encoding="utf-8", buffering=1)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        return self._file

    def write(self, kind: str, **fields: Any) -> int:
        if self.closed:
            return 0
        self.seq += 1
        record: dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "seq": self.seq,
            "kind": str(kind),
        }
        record.update(fields)
        if self._redact is not None:
            try:
                record = self._redact(record)
            except Exception:
                pass
        handle = self._open()
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        return self.seq

    def close(self) -> None:
        if self.closed:
            return
        if self._file is not None:
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            except OSError:
                pass
            self._file.close()
            self._file = None
        self.closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
