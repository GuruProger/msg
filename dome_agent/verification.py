"""Explicit expected effects for Dome control actions.

The agent must be able to prove that a command changed telemetry, not merely
that the API accepted it.  This module keeps that contract independent from the
transport layer and from the LLM: an :class:`ExpectedEffect` describes what
should be observed, how to check it, and how long to wait.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .analysis import as_bool, as_float, normalize_interlocks

DEFAULT_TIMEOUT_S = 6.0
_DIFF_METADATA_KEYS = {
    "accepted",
    "action",
    "node_id",
    "request_id",
    "success",
    "timestamp",
    "updated_at",
    "verified",
}


def _metadata_only_mapping(value: Mapping[Any, Any]) -> bool:
    keys = {str(key).casefold() for key in value}
    return bool(keys) and keys <= _DIFF_METADATA_KEYS


def meaningful_diff(value: Any) -> bool:
    """Return True only when a diff contains a non-metadata change.

    A command acknowledgement may include ``node_id`` or ``action``; that is
    not proof of a physical effect.  Known actions use explicit predicates;
    this fallback is only for actions without a predicate.
    """
    if not isinstance(value, Mapping):
        return False
    for key, change in value.items():
        if str(key).casefold() in _DIFF_METADATA_KEYS:
            continue
        if isinstance(change, Mapping) and "before" in change and "after" in change:
            old = change.get("before")
            new = change.get("after")
            if old == new:
                continue
            if (
                isinstance(old, Mapping)
                and isinstance(new, Mapping)
                and _metadata_only_mapping(old)
                and _metadata_only_mapping(new)
            ):
                continue
            return True
        if isinstance(change, Mapping):
            if meaningful_diff(change):
                return True
            continue
        return True
    return False


def _upper(value: Any) -> str:
    return str(value or "").strip().upper()


def _line_entry(node: Mapping[str, Any], number: Any) -> Mapping[str, Any] | None:
    try:
        wanted = int(number)
    except (TypeError, ValueError):
        return None
    for line in node.get("lines") or []:
        if not isinstance(line, Mapping):
            continue
        try:
            if int(line.get("line")) == wanted:
                return line
        except (TypeError, ValueError):
            continue
    return None


def _action_confirmed(
    node_id: str,
    action: str,
    value: Any,
    node: Any,
) -> bool | None:
    if not isinstance(node, Mapping) or not node:
        return False
    name = str(action or "").strip().casefold()
    state = _upper(node.get("state"))
    if name in {"line_on", "line_off"}:
        line = _line_entry(node, value)
        if line is None:
            return False
        expected = "ON" if name == "line_on" else "OFF"
        return _upper(line.get("state")) == expected
    if name == "reset_protection":
        line = _line_entry(node, value)
        if line is None:
            return False
        return _upper(line.get("protection_state")) == "ARMED"
    if name == "start_job":
        if state not in {"PRINTING", "RUNNING"}:
            return False
        expected = str(value or "").strip()
        current = str(node.get("file_name") or "").strip()
        return not expected or not current or current.casefold() == expected.casefold()
    if name == "resume":
        hold = _upper(node.get("hold_level"))
        return (
            not as_bool(node.get("paused"))
            and hold not in {"MANDATORY", "SAFETY", "ALARM"}
            and state not in {"PAUSED", "ALARM", "ERROR"}
        )
    if name == "home":
        return as_bool(node.get("homed"))
    if name == "turn_on":
        if "enabled" in node:
            return as_bool(node.get("enabled"))
        return state in {"ON", "RUNNING", "ACTIVE", "AVAILABLE"}
    if name == "turn_off":
        if "enabled" in node:
            return not as_bool(node.get("enabled"))
        if state:
            return state in {"OFF", "STOPPED", "DISABLED"}
        return None
    if name == "stop_job":
        return state in {"IDLE", "STOPPED", "COMPLETED", "CANCELLED", "CANCELED"} and not as_bool(
            node.get("paused")
        )
    if name == "reconnect":
        online = node.get("online")
        if online is None:
            return None
        return as_bool(online) and state not in {"UNKNOWN", "OFFLINE"}
    if name == "set_mode":
        return _upper(node.get("mode")) == _upper(value)
    if name == "set_primary_sensor":
        return str(node.get("primary_co2_sensor") or "").strip() == str(value or "").strip()
    if name == "reset_interlock":
        entry = normalize_interlocks(node.get("interlocks")).get(str(value))
        if not isinstance(entry, Mapping):
            return False
        return _upper(entry.get("state")) not in {"TRIPPED", "BLOCKED", "ALARM"}
    if name == "reset_error":
        return not as_bool(node.get("error")) and not node.get("error_code")
    if name == "set_fan":
        expected = _upper(value)
        fan_state = _upper(node.get("fan_state"))
        current = as_float(node.get("fan_current_a"))
        if expected in {"ON", "1", "TRUE"}:
            return fan_state == "ON" and current is not None and current > 0.01
        if expected in {"OFF", "0", "FALSE"}:
            return fan_state == "OFF" and current is not None and current <= 0.01
        return False
    if name == "set_data_trust":
        trust = node.get("data_trust")
        if not isinstance(trust, Mapping) or not isinstance(value, Mapping):
            return None
        suspect = str(value.get("node") or "")
        param = str(value.get("param") or "")
        expected = _upper(value.get("trust"))
        found = trust.get(f"{suspect}.{param}")
        if found is None:
            nested = trust.get(suspect)
            if isinstance(nested, Mapping):
                found = nested.get(param)
        if found is None:
            return None
        return _upper(found) == expected
    return None


def _observed(node_id: str, action: str, value: Any, node: Any) -> dict[str, Any] | None:
    if not isinstance(node, Mapping):
        return None
    name = str(action or "").strip().casefold()
    if name in {"line_on", "line_off", "reset_protection"}:
        line = _line_entry(node, value)
        if line is None:
            return None
        return {
            "line": line.get("line"),
            "state": line.get("state"),
            "protection_state": line.get("protection_state"),
            "current_a": line.get("current_a"),
            "power_w": line.get("power_w"),
        }
    if name in {"start_job", "resume", "home", "stop_job"}:
        keys = ("state", "paused", "homed", "file_name", "progress_pct", "hold_level", "hold_reason")
    elif name == "turn_off":
        keys = ("state", "enabled", "power_w", "fan_rpm", "fan_current_a")
    elif name == "reconnect":
        keys = ("state", "online", "status")
    elif name in {"turn_on", "set_mode"}:
        keys = ("state", "enabled", "mode", "airflow_pct")
    elif name == "set_fan":
        keys = ("fan_state", "fan_mode", "fan_current_a", "inverter_temp_c")
    elif name == "reset_error":
        keys = ("state", "error", "error_code")
    elif name == "set_primary_sensor":
        keys = ("primary_co2_sensor",)
    elif name == "reset_interlock":
        entry = normalize_interlocks(node.get("interlocks")).get(str(value))
        return {"interlock": value, "entry": entry}
    elif name == "set_data_trust":
        return {"data_trust": node.get("data_trust")}
    else:
        keys = ("state",)
    return {key: node.get(key) for key in keys if key in node}


def _description(node_id: str, action: str, value: Any, *, has_value: bool) -> str:
    value_text = "" if not has_value else f" {value!r}"
    if action in {"line_on", "line_off"}:
        return f"{node_id}.line {value} = {'ON' if action == 'line_on' else 'OFF'}"
    if action == "start_job":
        return f"{node_id}.state changes to PRINTING/RUNNING for {value!r}"
    if action == "resume":
        return f"{node_id} is no longer paused"
    if action == "home":
        return f"{node_id}.homed or state becomes operational"
    if action == "set_primary_sensor":
        return f"{node_id}.primary_co2_sensor = {value!r}"
    if action == "set_data_trust":
        return f"{node_id} data trust is applied"
    if action == "reset_interlock":
        return f"{node_id}.{value} is no longer tripped"
    if action == "reset_protection":
        return f"{node_id}.line {value} protection = ARMED"
    return f"{node_id}.{action}{value_text} is reflected in telemetry"


@dataclass(frozen=True, slots=True, eq=False)
class ExpectedEffect:
    """The telemetry effect that must be observed after a control action."""

    description: str
    check: Callable[[Mapping[str, Any]], bool | None]
    observe: Callable[[Mapping[str, Any]], Any]
    timeout_s: float = DEFAULT_TIMEOUT_S


def expected_effect(
    node_id: str,
    action: str,
    value: Any = None,
    *,
    has_value: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ExpectedEffect:
    """Build an explicit effect contract for a known Dome action.

    ``check`` returns ``True`` when the effect is proven, ``False`` when it is
    not yet observed, and ``None`` when the action has no known predicate (the
    caller may then fall back to the API-provided telemetry diff).
    """
    name = str(action or "").strip()
    normalized = name.casefold()

    def check(node: Mapping[str, Any]) -> bool | None:
        return _action_confirmed(node_id, name, value, node)

    def observe(node: Mapping[str, Any]) -> Any:
        return _observed(node_id, name, value, node)

    return ExpectedEffect(
        description=_description(node_id, normalized, value, has_value=has_value),
        check=check,
        observe=observe,
        timeout_s=max(0.1, float(timeout_s)),
    )
