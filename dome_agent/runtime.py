"""Local runtime overrides shared by a running agent and the local panel.

The panel and the agent are separate processes.  The panel cannot mutate an
``AgentConfig`` that already lives in the agent's memory, so the two exchange
a very small, explicitly validated JSON document:

    {"mode": "auto", "dome_url": "wss://dome.example:443/ws",
     "updated_at": "2026-10-27T12:00:00+03:00", "requested_by": "panel"}

The file contains no secrets and is written atomically with mode ``0600``.
Only values that are already valid for startup are accepted.  A scored run
still rejects ``yolo`` in :class:`~dome_agent.config.AgentConfig`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

MODES: tuple[str, ...] = ("normal", "auto", "yolo")
DEFAULT_RUNTIME_NAME = "agent_runtime.json"
STATUS_SUFFIX = "_status"
MAX_RUNTIME_BYTES = 64 * 1024


class RuntimeConfigError(ValueError):
    """The runtime override document is malformed or unsafe."""


def normalize_mode(value: Any) -> str:
    """Return a known agent mode or raise :class:`RuntimeConfigError`."""
    mode = str(value or "").strip().casefold()
    if mode not in MODES:
        raise RuntimeConfigError(
            f"AGENT_MODE must be normal, auto or yolo, got {value!r}"
        )
    return mode


def normalize_dome_url(value: Any) -> str:
    """Validate a Dome WebSocket URL, including an optional port and path."""
    url = str(value or "").strip().rstrip("/")
    if not url:
        raise RuntimeConfigError("DOME_URL must not be empty")
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise RuntimeConfigError(f"DOME_URL is not a valid URL: {value!r}") from exc
    if parts.scheme.casefold() not in {"ws", "wss"}:
        raise RuntimeConfigError(
            f"DOME_URL must be ws://host:port or wss://host, got {value!r}"
        )
    if not parts.hostname:
        raise RuntimeConfigError(f"DOME_URL has no host: {value!r}")
    if parts.username or parts.password:
        raise RuntimeConfigError("DOME_URL must not contain credentials")
    if parts.query or parts.fragment:
        raise RuntimeConfigError("DOME_URL must not contain a query or fragment")
    try:
        port = parts.port
    except ValueError as exc:
        raise RuntimeConfigError(f"DOME_URL has an invalid port: {value!r}") from exc
    if port is not None and not 1 <= port <= 65535:
        raise RuntimeConfigError(f"DOME_URL port is out of range: {value!r}")
    return url


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """Validated overrides requested by the panel."""

    mode: str | None = None
    dome_url: str | None = None
    updated_at: str | None = None
    requested_by: str = "panel"
    ack_url: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "dome_url": self.dome_url,
            "updated_at": self.updated_at,
            "requested_by": self.requested_by,
        }


def runtime_file_for(log_dir: str | Path, value: str | Path | None = None) -> Path:
    """Resolve the runtime file, keeping it next to the agent logs."""
    if value is not None and str(value).strip():
        return Path(value).expanduser()
    return Path(log_dir).expanduser() / DEFAULT_RUNTIME_NAME


def status_path(runtime_file: str | Path) -> Path:
    """Status file written by the agent next to the runtime overrides."""
    path = Path(runtime_file).expanduser()
    return path.with_name(f"{path.stem}{STATUS_SUFFIX}{path.suffix or '.json'}")


def read_runtime(
    path: str | Path,
    *,
    strict: bool = True,
) -> RuntimeState | None:
    """Read validated overrides.

    A missing file means "no overrides".  With ``strict=False`` an unreadable
    or malformed file is also treated as absent, so a live agent can keep its
    last known-good configuration and report the error.
    """
    candidate = Path(path).expanduser()
    try:
        if candidate.stat().st_size > MAX_RUNTIME_BYTES:
            raise RuntimeConfigError(f"runtime file is too large: {candidate}")
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        if strict:
            raise RuntimeConfigError(f"cannot read runtime file {candidate}: {exc}") from exc
        return None
    if not isinstance(raw, Mapping):
        if strict:
            raise RuntimeConfigError("runtime file must contain a JSON object")
        return None
    try:
        mode = normalize_mode(raw["mode"]) if raw.get("mode") not in (None, "") else None
        dome_url = (
            normalize_dome_url(raw["dome_url"])
            if raw.get("dome_url") not in (None, "")
            else None
        )
    except RuntimeConfigError:
        if strict:
            raise
        return None
    updated_at = raw.get("updated_at")
    updated_text = str(updated_at).strip()[:64] if updated_at else None
    requested_by = str(raw.get("requested_by") or "panel").strip()[:40] or "panel"
    return RuntimeState(
        mode=mode,
        dome_url=dome_url,
        updated_at=updated_text,
        requested_by=requested_by,
        ack_url=bool(raw.get("ack_url")),
    )


def write_runtime(
    path: str | Path,
    *,
    mode: str | None = None,
    dome_url: str | None = None,
    requested_by: str = "panel",
    ack_url: bool = False,
) -> RuntimeState:
    """Merge and atomically persist validated overrides."""
    candidate = Path(path).expanduser()
    current = read_runtime(candidate, strict=False)
    next_mode = normalize_mode(mode) if mode is not None else (current.mode if current else None)
    next_url = (
        normalize_dome_url(dome_url)
        if dome_url is not None
        else (current.dome_url if current else None)
    )
    state = RuntimeState(
        mode=next_mode,
        dome_url=next_url,
        updated_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
        requested_by=str(requested_by or "panel").strip()[:40] or "panel",
        ack_url=bool(ack_url),
    )
    payload = {
        "mode": state.mode,
        "dome_url": state.dome_url,
        "updated_at": state.updated_at,
        "requested_by": state.requested_by,
        "ack_url": state.ack_url,
    }
    write_json_atomic(candidate, payload)
    return state


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a small local JSON document without exposing partial content."""
    candidate = Path(path).expanduser()
    candidate.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{candidate.name}.",
        suffix=".tmp",
        dir=str(candidate.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, candidate)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_status(path: str | Path) -> dict[str, Any] | None:
    """Read the agent heartbeat; unreadable status is not an error."""
    candidate = Path(path).expanduser()
    try:
        if candidate.stat().st_size > MAX_RUNTIME_BYTES:
            return None
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(raw) if isinstance(raw, Mapping) else None
