"""Read-only FastAPI journal API for the local web panel.

The API is deliberately a viewer: it only reads the agent JSONL log and the
persistent state file.  It never sends Dome commands, mutates state, or writes
logs.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

MAX_TAIL_BYTES: Final = 4_000_000
STALE_AFTER_S: Final = 120.0
MAX_ALERTS: Final = 20
TIMELINE_LIMIT: Final = 60
RUN_ID_RE: Final = re.compile(r"[A-Za-z0-9_.-]+")
SECRET_WORDS: Final = ("key", "token", "secret", "password")
METRIC_KEYS: Final = ("co2_ppm", "soc_pct", "printer", "cnc", "fume", "power_w")
TIMELINE_KINDS: Final = frozenset(
    {
        "start",
        "stop",
        "connection",
        "observation",
        "decision",
        "action",
        "action_blocked",
        "action_reconcile",
        "verification",
        "message",
        "llm_call",
        "error",
    }
)
SECRET_ENV_NAMES: Final = ("DOME_KEY", "DEEPSEEK_API_KEY", "DEEPSEEK_KEY")


@dataclass(frozen=True, slots=True)
class _LogSnapshot:
    """Bounded view of the newest agent log file."""

    log_path: Path | None
    records: tuple[dict[str, Any], ...]
    truncated: bool
    read_error: bool


def _as_int(value: Any) -> int | None:
    """Return an integer for an int-like JSON value, otherwise ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> float | None:
    """Return a finite float for a numeric JSON value, otherwise ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_text(value: Any, limit: int = 500) -> str:
    """Convert an arbitrary JSON value into bounded display text."""
    if value is None:
        return ""
    return str(value).strip()[:limit]


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO timestamp, tolerating a trailing ``Z`` and naive values."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _record_age_s(record: Mapping[str, Any] | None, now: datetime) -> float | None:
    """Return the age of a record in seconds when its timestamp is valid."""
    if record is None:
        return None
    parsed = _parse_timestamp(record.get("ts"))
    if parsed is None:
        return None
    return round(max(0.0, (now - parsed).total_seconds()), 1)


def _is_secret_name(name: str) -> bool:
    """Return True when a field name contains a configured secret word."""
    folded = name.casefold()
    return any(word in folded for word in SECRET_WORDS)


def _env_secrets() -> frozenset[str]:
    """Collect literal secret values available in the process environment."""
    values = {
        value
        for name in SECRET_ENV_NAMES
        if (value := os.getenv(name))
    }
    return frozenset(values)


def _collect_secret_values(*payloads: Any) -> frozenset[str]:
    """Find literal values stored under secret-looking keys in JSON data."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if _is_secret_name(str(key)) and isinstance(item, str) and item:
                    found.add(item)
                walk(item)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item)

    for payload in payloads:
        walk(payload)
    return frozenset(found) | _env_secrets()


def _scrub_text(text: str, secrets: frozenset[str]) -> str:
    """Replace known literal secret values in a string."""
    result = text
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result


def _sanitize(value: Any, secrets: frozenset[str]) -> Any:
    """Recursively drop secret-named fields and scrub literal secrets."""
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if _is_secret_name(name):
                continue
            cleaned[name] = _sanitize(item, secrets)
        return cleaned
    if isinstance(value, list | tuple):
        return [_sanitize(item, secrets) for item in value]
    if isinstance(value, str):
        return _scrub_text(value, secrets)
    return value


def _latest_log_file(log_dir: Path) -> Path | None:
    """Return the newest ``agent_*.jsonl`` file by modification time."""
    candidates = [path for path in log_dir.glob("agent_*.jsonl") if path.is_file()]
    if not candidates:
        return None
    ranked: list[tuple[int, str, Path]] = []
    for path in candidates:
        try:
            ranked.append((path.stat().st_mtime_ns, path.name, path))
        except OSError:
            continue
    if not ranked:
        return None
    return max(ranked)[2]


def _read_log_file(path: Path) -> tuple[tuple[dict[str, Any], ...], bool]:
    """Read a bounded tail, reusing parsed records until the file changes."""
    stat = path.stat()
    return _read_log_file_cached(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=2)
def _read_log_file_cached(
    path_text: str,
    size: int,
    mtime_ns: int,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    """Cache the parsed tail for the current size/mtime of one log file."""
    return _read_log_file_uncached(Path(path_text), size)


def _read_log_file_uncached(
    path: Path,
    size: int,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    """Read a bounded tail of a JSONL log, ignoring corrupt/partial lines."""
    with path.open("rb") as handle:
        if size > MAX_TAIL_BYTES:
            handle.seek(size - MAX_TAIL_BYTES)
            data = handle.read()
            first_newline = data.find(b"\n")
            data = b"" if first_newline < 0 else data[first_newline + 1 :]
        else:
            data = handle.read()

    records: list[dict[str, Any]] = []
    for raw_line in data.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return tuple(records), size > MAX_TAIL_BYTES


def _snapshot(log_dir: Path, state_file: Path) -> _LogSnapshot:
    """Load the newest log tail and record any read failure."""
    log_path: Path | None = None
    records: tuple[dict[str, Any], ...] = ()
    truncated = False
    read_error = False
    try:
        log_path = _latest_log_file(log_dir)
    except OSError:
        read_error = True
    if log_path is not None:
        try:
            records, truncated = _read_log_file(log_path)
        except OSError:
            read_error = True
            records = ()
            truncated = False
    return _LogSnapshot(
        log_path=log_path,
        records=records,
        truncated=truncated,
        read_error=read_error,
    )


def _health_payload(snapshot: _LogSnapshot, state_file: Path) -> dict[str, Any]:
    """Build the health object shared by the health and summary endpoints."""
    last = snapshot.records[-1] if snapshot.records else None
    age = _record_age_s(last, datetime.now(UTC))
    if snapshot.read_error:
        status = "error"
    elif snapshot.log_path is None or last is None:
        status = "waiting"
    elif last.get("kind") == "stop":
        status = "stopped"
    elif age is not None and age > STALE_AFTER_S:
        status = "stale"
    else:
        status = "ok"

    return {
        "status": status,
        "has_log": snapshot.log_path is not None,
        "has_state": state_file.is_file(),
        "log_file": snapshot.log_path.name if snapshot.log_path is not None else None,
        "state_file": str(state_file),
        "last_record_age_s": age,
        "records_loaded": len(snapshot.records),
    }


def _empty_state() -> dict[str, Any]:
    """Return the public state shape when no usable state file exists."""
    return {
        "task": None,
        "last_message_id": 0,
        "co2_override": None,
        "last_actions": {},
        "failed_until": {},
        "notes": {},
        "updated_at": None,
    }


def _file_mtime_iso(path: Path) -> str | None:
    """Return the state file modification time as an ISO timestamp."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(
            timespec="seconds"
        )
    except OSError:
        return None


def _load_state(path: Path) -> dict[str, Any]:
    """Load, whitelist, and sanitize the persistent agent state file."""
    payload = _empty_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return payload
    if not isinstance(raw, dict):
        return payload

    task = raw.get("task")
    payload["task"] = task if isinstance(task, dict) else None
    payload["last_message_id"] = max(0, _as_int(raw.get("last_message_id")) or 0)
    co2_override = raw.get("co2_override")
    payload["co2_override"] = co2_override if isinstance(co2_override, dict) else None
    for key in ("last_actions", "failed_until", "notes"):
        value = raw.get(key)
        payload[key] = value if isinstance(value, dict) else {}
    updated_at = raw.get("updated_at")
    payload["updated_at"] = (
        updated_at if isinstance(updated_at, str) and updated_at else _file_mtime_iso(path)
    )
    return _sanitize(payload, _env_secrets())


def _last_record(
    records: tuple[dict[str, Any], ...], kind: str
) -> dict[str, Any] | None:
    """Return the most recent record of ``kind``."""
    for record in reversed(records):
        if record.get("kind") == kind:
            return record
    return None


def _safe_config(raw: Any) -> dict[str, Any]:
    """Project a start record config onto the dashboard allowlist."""
    config = raw if isinstance(raw, Mapping) else {}

    def pick(*names: str) -> Any:
        for name in names:
            if name in config:
                return config[name]
        return None

    return {
        "mode": pick("mode"),
        "allow_control": pick("allow_control", "control_enabled"),
        "dry_run": pick("dry_run"),
        "dome_url": pick("dome_url"),
        "name": pick("name", "agent"),
        "policy": pick("policy", "policy_file"),
        "llm": pick("llm", "llm_enabled"),
        "run_seconds": pick("run_seconds"),
        "observation_interval": pick("observation_interval", "observation_interval_s"),
        "min_send_gap": pick("min_send_gap", "min_send_gap_s"),
    }


def _normalize_metrics(raw: Any) -> dict[str, Any]:
    """Return exactly the public metric keys, using null for missing values."""
    metrics = raw if isinstance(raw, Mapping) else {}
    return {key: metrics.get(key) for key in METRIC_KEYS}


def _string_list(raw: Any) -> list[str]:
    """Normalize an arbitrary list into a list of display strings."""
    if not isinstance(raw, list):
        return []
    return [_as_text(item) for item in raw if item is not None]


def _verification(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a verification record for the public API."""
    return {
        "ts": record.get("ts"),
        "seq": _as_int(record.get("seq")),
        "ref": _as_int(record.get("ref")),
        "expected": record.get("expected"),
        "observed": record.get("observed"),
        "ok": record.get("ok"),
    }


def _action(
    record: Mapping[str, Any],
    verification: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize an action record and attach its verification when present."""
    request = record.get("request")
    response = record.get("response")
    response_map = response if isinstance(response, Mapping) else {}
    return {
        "ts": record.get("ts"),
        "seq": _as_int(record.get("seq")),
        "request": request if isinstance(request, Mapping) else {},
        "response": {
            "success": response_map.get("success"),
            "error": response_map.get("error"),
        },
        "ref": _as_int(record.get("ref")),
        "verification": verification,
    }


def _message(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a chat message record."""
    sender = record.get("sender")
    if isinstance(sender, Mapping):
        role = sender.get("role")
        name = sender.get("name")
    else:
        role = record.get("role")
        name = sender
    return {
        "ts": record.get("ts"),
        "seq": _as_int(record.get("seq")),
        "direction": record.get("direction"),
        "sender": {"role": role, "name": name},
        "to": record.get("to"),
        "text": record.get("text"),
    }


def _request_label(record: Mapping[str, Any]) -> str:
    """Build a short human label for an action request."""
    request = record.get("request")
    if not isinstance(request, Mapping):
        return ""
    node = _as_text(request.get("node_id") or request.get("node"))
    action = _as_text(request.get("action") or request.get("type"))
    if node and action:
        return f"{node}.{action}"
    return action or node


def _build_alerts(records: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Extract recent operational alerts, deduped and capped."""
    candidates: list[dict[str, Any]] = []
    for record in records:
        kind = record.get("kind")
        if kind == "error":
            where = _as_text(record.get("where"), 120)
            detail = _as_text(record.get("error") or "unknown error", 500)
            candidates.append(
                {
                    "level": "critical",
                    "title": "Agent error",
                    "detail": f"{where}: {detail}" if where else detail,
                    "ts": record.get("ts"),
                }
            )
        elif kind == "action":
            response = record.get("response")
            if isinstance(response, Mapping) and response.get("success") is False:
                label = _request_label(record) or "Action"
                error = _as_text(response.get("error"), 300)
                candidates.append(
                    {
                        "level": "warning",
                        "title": "Action failed",
                        "detail": f"{label}: {error}" if error else label,
                        "ts": record.get("ts"),
                    }
                )
        elif kind == "action_blocked":
            label = _request_label(record)
            reason = _as_text(record.get("reason") or "blocked", 500)
            candidates.append(
                {
                    "level": "warning",
                    "title": "Action blocked",
                    "detail": f"{label}: {reason}" if label else reason,
                    "ts": record.get("ts"),
                }
            )
        elif kind == "observation":
            anomalies = record.get("anomalies")
            if isinstance(anomalies, list):
                for anomaly in anomalies:
                    detail = _as_text(anomaly, 500)
                    if detail:
                        candidates.append(
                            {
                                "level": "warning",
                                "title": "Observation anomaly",
                                "detail": detail,
                                "ts": record.get("ts"),
                            }
                        )

    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for alert in candidates:
        key = (
            _as_text(alert.get("level")),
            _as_text(alert.get("title")),
            _as_text(alert.get("detail")),
        )
        deduped.pop(key, None)
        deduped[key] = alert
    return list(deduped.values())[-MAX_ALERTS:]


def _timeline_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one log record for the dashboard activity timeline."""
    raw_kind = _as_text(record.get("kind"), 80)
    kind = raw_kind if raw_kind in TIMELINE_KINDS else "unknown"
    status = "info"
    title = "Unknown"
    detail = ""

    if kind == "start":
        title = "Start"
        status = "success"
        detail = " ".join(
            part
            for part in (
                _as_text(record.get("agent"), 120),
                _as_text(record.get("version"), 80),
            )
            if part
        )
    elif kind == "stop":
        title = "Stop"
        detail = _as_text(record.get("reason"), 300)
    elif kind == "connection":
        title = "Connection"
        detail = " ".join(
            part
            for part in (
                _as_text(record.get("event"), 80),
                _as_text(record.get("role"), 80),
            )
            if part
        )
    elif kind == "observation":
        title = "Observation"
        detail = _as_text(record.get("summary"), 500)
        if record.get("anomalies"):
            status = "warning"
    elif kind == "decision":
        title = "Decision"
        detail = _as_text(record.get("situation"), 500)
    elif kind == "action":
        title = "Action"
        detail = _request_label(record)
        response = record.get("response")
        if isinstance(response, Mapping) and response.get("success") is True:
            status = "success"
        else:
            status = "error"
    elif kind == "action_blocked":
        title = "Action blocked"
        detail = _as_text(record.get("reason"), 500)
        status = "warning"
    elif kind == "action_reconcile":
        title = "Action reconciliation"
        detail = _as_text(record.get("expected"), 500)
        status = "warning"
    elif kind == "verification":
        title = "Verification"
        detail = _as_text(record.get("expected"), 500)
        status = "success" if record.get("ok") is True else "error"
    elif kind == "message":
        title = "Message"
        detail = _as_text(record.get("text"), 500)
    elif kind == "llm_call":
        title = "LLM call"
        detail = " ".join(
            part
            for part in (
                _as_text(record.get("model"), 120),
                _as_text(record.get("purpose"), 300),
            )
            if part
        )
        status = "success" if record.get("ok") is not False else "error"
    elif kind == "error":
        title = "Error"
        detail = _as_text(record.get("error"), 500)
        status = "error"
    else:
        detail = raw_kind

    return {
        "ts": record.get("ts"),
        "seq": _as_int(record.get("seq")),
        "kind": kind,
        "title": title,
        "detail": detail,
        "status": status,
        "ref": _as_int(record.get("ref")),
    }


def _build_summary(
    records: tuple[dict[str, Any], ...],
    health: dict[str, Any],
    state: dict[str, Any],
    log_path: Path | None,
    *,
    demo: bool,
) -> dict[str, Any]:
    """Assemble the complete dashboard summary from loaded records."""
    start = _last_record(records, "start")
    last = records[-1] if records else None

    run_id: Any = start.get("run_id") if start is not None else None
    if not isinstance(run_id, str) or not run_id:
        if last is not None and isinstance(last.get("run_id"), str):
            run_id = last.get("run_id")
        elif log_path is not None:
            run_id = log_path.stem.removeprefix("agent_")

    models_raw = start.get("models") if start is not None else None
    models = (
        [_as_text(item, 120) for item in models_raw if item is not None]
        if isinstance(models_raw, list)
        else []
    )

    role: str | None = None
    team: int | None = None
    for record in reversed(records):
        if record.get("kind") != "connection":
            continue
        record_role = record.get("role")
        record_team = _as_int(record.get("team"))
        if record_role is not None or record_team is not None:
            role = record_role if isinstance(record_role, str) else None
            team = record_team
            break

    stopped = last is not None and last.get("kind") == "stop"
    stop_reason = None
    if stopped and last is not None and isinstance(last.get("reason"), str):
        stop_reason = last.get("reason")

    observation_record = _last_record(records, "observation")
    observation = None
    if observation_record is not None:
        observation = {
            "ts": observation_record.get("ts"),
            "dome_time": observation_record.get("dome_time"),
            "summary": observation_record.get("summary"),
            "metrics": _normalize_metrics(observation_record.get("metrics")),
            "metric_sources": {
                "co2_ppm": (observation_record.get("metrics") or {}).get("co2_source"),
            },
            "anomalies": _string_list(observation_record.get("anomalies")),
        }
    metrics = (
        observation["metrics"] if observation is not None else _normalize_metrics(None)
    )

    decision_record = _last_record(records, "decision")
    decision = None
    if decision_record is not None:
        evidence = decision_record.get("evidence")
        plan = decision_record.get("plan")
        decision = {
            "ts": decision_record.get("ts"),
            "situation": decision_record.get("situation"),
            "cause": decision_record.get("cause"),
            "confidence": _as_float(decision_record.get("confidence")),
            "plan": plan if isinstance(plan, list) else [],
            "source": decision_record.get("source"),
            "evidence": evidence if isinstance(evidence, list) else [],
            "ref": _as_int(decision_record.get("ref")),
        }

    verifications = [
        _verification(record) for record in records if record.get("kind") == "verification"
    ]
    verifications_by_ref = {
        item["ref"]: item for item in verifications if item.get("ref") is not None
    }
    actions = [
        _action(record, verifications_by_ref.get(_as_int(record.get("seq"))))
        for record in records
        if record.get("kind") == "action"
    ]
    messages = [
        _message(record) for record in records if record.get("kind") == "message"
    ]

    counts = {
        "records": len(records),
        "decisions": sum(1 for record in records if record.get("kind") == "decision"),
        "actions": len(actions),
        "verifications": len(verifications),
        "messages": len(messages),
        "errors": sum(1 for record in records if record.get("kind") == "error"),
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "demo": demo,
        "health": health,
        "run": {
            "run_id": run_id if isinstance(run_id, str) else None,
            "file": str(log_path) if log_path is not None else None,
            "started_at": start.get("ts") if start is not None else None,
            "last_ts": last.get("ts") if last is not None else None,
            "stopped": stopped,
            "stop_reason": stop_reason,
            "agent": {
                "name": start.get("agent") if start is not None else None,
                "version": start.get("version") if start is not None else None,
                "models": models,
            },
            "config": _safe_config(start.get("config") if start is not None else None),
            "role": role,
            "team": team,
        },
        "state": state,
        "observation": observation,
        "decision": decision,
        "alerts": _build_alerts(records),
        "actions": actions,
        "verifications": verifications,
        "messages": messages,
        "timeline": [_timeline_entry(record) for record in records[-TIMELINE_LIMIT:]],
        "counts": counts,
        "metrics": metrics,
    }


def _demo_summary() -> dict[str, Any]:
    """Return a realistic summary generated entirely in memory."""
    now = datetime.now(UTC)
    run_id = now.astimezone().strftime("%Y%m%d-%H%M%S")

    def stamp(seconds_ago: int) -> str:
        return (now - timedelta(seconds=seconds_ago)).isoformat(timespec="milliseconds")

    records: tuple[dict[str, Any], ...] = (
        {
            "ts": stamp(180),
            "run_id": run_id,
            "seq": 1,
            "kind": "start",
            "agent": "team3-agent",
            "version": "abc1234",
            "models": ["deepseek-chat"],
            "config": {
                "mode": "normal",
                "allow_control": True,
                "dry_run": False,
                "dome_url": "wss://dome.example/ws",
                "name": "team3-agent",
                "policy": "dome_policy.json",
                "llm": True,
                "run_seconds": 0,
                "observation_interval": 45,
                "min_send_gap": 0.35,
            },
        },
        {
            "ts": stamp(176),
            "run_id": run_id,
            "seq": 2,
            "kind": "connection",
            "event": "auth",
            "role": "team",
            "team": 3,
        },
        {
            "ts": stamp(90),
            "run_id": run_id,
            "seq": 3,
            "kind": "observation",
            "dome_time": "14:02",
            "summary": "Смена стабильна: печать идёт, CO2 в норме.",
            "metrics": {
                "co2_ppm": 810,
                "soc_pct": 67,
                "printer": "PRINTING",
                "cnc": "IDLE",
                "fume": "ON",
                "power_w": 420,
            },
            "anomalies": ["CO2 растёт быстрее обычного"],
        },
        {
            "ts": stamp(78),
            "run_id": run_id,
            "seq": 4,
            "kind": "decision",
            "situation": "Печать продолжается, вентиляция работает",
            "cause": "Параметры в допустимых границах",
            "confidence": 0.86,
            "plan": ["наблюдать за CO2", "проверить завершение печати"],
            "source": "mixed",
            "evidence": [{"node": "climate_sensor_02", "param": "co2_ppm", "value": 790}],
            "ref": 120,
        },
        {
            "ts": stamp(66),
            "run_id": run_id,
            "seq": 5,
            "kind": "action",
            "request": {
                "type": "control",
                "node_id": "smart_panel_01",
                "action": "line_off",
                "value": 8,
            },
            "response": {"success": True, "error": None},
            "ref": 120,
        },
        {
            "ts": stamp(62),
            "run_id": run_id,
            "seq": 6,
            "kind": "verification",
            "ref": 5,
            "expected": "smart_panel_01.line 8 = OFF",
            "observed": {"line": 8, "state": "OFF"},
            "ok": True,
        },
        {
            "ts": stamp(40),
            "run_id": run_id,
            "seq": 7,
            "kind": "message",
            "direction": "in",
            "sender": {"role": "operator", "name": "Оператор"},
            "to": None,
            "text": "Проверь завершение печати и доложи.",
        },
        {
            "ts": stamp(25),
            "run_id": run_id,
            "seq": 8,
            "kind": "action_blocked",
            "request": {
                "type": "control",
                "node_id": "printer_3d_01",
                "action": "start_job",
                "value": "part.gcode",
            },
            "reason": "policy requires operator confirmation",
            "ref": 120,
        },
        {
            "ts": stamp(10),
            "run_id": run_id,
            "seq": 9,
            "kind": "error",
            "where": "llm_advice",
            "error": "DeepSeekTimeout: advisory call timed out",
        },
        {
            "ts": stamp(2),
            "run_id": run_id,
            "seq": 10,
            "kind": "message",
            "direction": "out",
            "sender": {"role": "agent", "name": "team3-agent"},
            "to": "operator",
            "text": "Печать идёт штатно, CO2 810 ppm.",
        },
    )

    health = {
        "status": "ok",
        "has_log": True,
        "has_state": True,
        "log_file": f"agent_{run_id}.jsonl",
        "state_file": "logs/agent_state.json",
        "last_record_age_s": 2.0,
        "records_loaded": len(records),
    }
    state = {
        "task": {"text": "Напечатать демонстрационную деталь", "status": "in_progress"},
        "last_message_id": 7,
        "co2_override": None,
        "last_actions": {},
        "failed_until": {},
        "notes": {"demo": "sample data"},
        "updated_at": stamp(30),
    }
    return _build_summary(
        records,
        health,
        state,
        Path("logs") / f"agent_{run_id}.jsonl",
        demo=True,
    )


def create_app(
    log_dir: str | Path | None = None,
    state_file: str | Path | None = None,
) -> FastAPI:
    """Create the read-only dashboard application.

    Explicit paths are intended for tests.  When a path is omitted it is
    resolved from the environment on every request, so a long-running server
    follows ``AGENT_LOG_DIR``/``AGENT_STATE_FILE`` changes.
    """
    app = FastAPI(title="Dome Dashboard", version="1.0.0")
    static_dir = Path(__file__).resolve().parent / "static"

    def resolve_paths() -> tuple[Path, Path]:
        if log_dir is not None:
            resolved_log = Path(log_dir).expanduser()
        else:
            resolved_log = Path(os.getenv("AGENT_LOG_DIR") or "logs").expanduser()
        if state_file is not None:
            resolved_state = Path(state_file).expanduser()
        else:
            env_state = os.getenv("AGENT_STATE_FILE")
            resolved_state = (
                Path(env_state).expanduser()
                if env_state and env_state.strip()
                else resolved_log / "agent_state.json"
            )
        return resolved_log, resolved_state

    @app.middleware("http")
    async def no_store(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        """Return the latest run health without exposing log contents."""
        resolved_log, resolved_state = resolve_paths()
        snapshot = _snapshot(resolved_log, resolved_state)
        return _health_payload(snapshot, resolved_state)

    @app.get("/api/summary")
    def api_summary() -> dict[str, Any]:
        """Return the complete dashboard summary for the newest run."""
        resolved_log, resolved_state = resolve_paths()
        snapshot = _snapshot(resolved_log, resolved_state)
        health = _health_payload(snapshot, resolved_state)
        state = _load_state(resolved_state)
        summary = _build_summary(
            snapshot.records,
            health,
            state,
            snapshot.log_path,
            demo=False,
        )
        secrets = _collect_secret_values(snapshot.records)
        return _sanitize(summary, secrets)

    @app.get("/api/logs")
    def api_logs(
        kind: str | None = None,
        limit: int = Query(200, ge=1, le=1000),
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Return the newest matching raw log records in chronological order."""
        resolved_log, _ = resolve_paths()
        selected: Path | None = None
        if run_id is not None and run_id.strip():
            if run_id in {".", ".."} or RUN_ID_RE.fullmatch(run_id) is None:
                raise HTTPException(status_code=400, detail="invalid run_id")
            candidate = resolved_log / f"agent_{run_id}.jsonl"
            if candidate.is_file():
                selected = candidate
            else:
                try:
                    matches = [
                        path
                        for path in resolved_log.glob("agent_*.jsonl")
                        if path.is_file()
                        and path.stem.removeprefix("agent_") == run_id
                    ]
                except OSError as exc:
                    raise HTTPException(
                        status_code=500, detail="unable to read log directory"
                    ) from exc
                if matches:
                    selected = max(matches, key=lambda path: path.stat().st_mtime_ns)
        else:
            try:
                selected = _latest_log_file(resolved_log)
            except OSError as exc:
                raise HTTPException(
                    status_code=500, detail="unable to read log directory"
                ) from exc

        if selected is None:
            return {
                "run_id": run_id if run_id and run_id.strip() else None,
                "file": None,
                "records": [],
                "truncated": False,
                "count": 0,
            }

        try:
            all_records, truncated = _read_log_file(selected)
        except OSError as exc:
            raise HTTPException(status_code=500, detail="unable to read log file") from exc

        filtered = (
            [record for record in all_records if record.get("kind") == kind]
            if kind
            else list(all_records)
        )
        window = filtered[-limit:]
        secrets = _collect_secret_values(window)
        selected_run_id = (
            run_id.strip()
            if run_id and run_id.strip()
            else selected.stem.removeprefix("agent_")
        )
        return {
            "run_id": selected_run_id,
            "file": str(selected),
            "records": _sanitize(window, secrets),
            "truncated": truncated,
            "count": len(window),
        }

    @app.get("/api/state")
    def api_state() -> dict[str, Any]:
        """Return the sanitized persistent agent state."""
        _, resolved_state = resolve_paths()
        return _load_state(resolved_state)

    @app.get("/api/demo")
    def api_demo() -> dict[str, Any]:
        """Return a complete sample summary without touching disk or network."""
        return _sanitize(_demo_summary(), _env_secrets())

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the dashboard frontend when it has been installed."""
        index_path = static_dir / "index.html"
        if not index_path.is_file():
            raise HTTPException(
                status_code=503, detail="dashboard frontend is not installed"
            )
        return FileResponse(index_path, media_type="text/html")

    app.mount(
        "/static",
        StaticFiles(directory=str(static_dir), check_dir=False),
        name="static",
    )
    return app


app = create_app()
