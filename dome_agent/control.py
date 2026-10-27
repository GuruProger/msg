"""Structured model control for the Dome agent.

The model may propose or execute a very small set of Dome operations through
:class:`DomeTool`.  There is intentionally no shell access: the only tools are
``control``, ``query``, ``telemetry``, ``catalog`` and ``send_message``.

``normal`` mode treats model commands as proposals until the operator confirms
them.  ``yolo`` mode executes them immediately.  A hard denylist is enforced in
both modes and cannot be overridden by the operator or by ``force``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from dome_control.agent_api import DomeAgentAPI

from .rules import NO_VALUE

HARD_DENY_ACTIONS: frozenset[str] = frozenset(
    {
        "block_automation",
        "block_interlock",
        "set_thresholds",
        "unlock_door",
        "remove_part",
        "shutdown",
        "reboot",
    }
)

CONTROL_TOOLS: frozenset[str] = frozenset(
    {"control", "query", "telemetry", "catalog", "send_message"}
)

MAX_COMMANDS = 6
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

# User-approved protective actions that need no confirmation in normal mode.
_PROTECTIVE_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("smart_panel_01", "line_off"),
        ("solar_inverter_01", "set_fan"),
        ("supply_ventilation_01", "turn_on"),
        ("fume_extraction_01", "turn_on"),
        ("supply_ventilation_01", "set_mode"),
    }
)


def _text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    return str(value).strip()[:limit]


def _as_float(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool) or value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result or result in {float("inf"), float("-inf")}:
        return default
    return result


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return default


def extract_json(content: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response."""
    text = _FENCE_RE.sub("", content or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def is_auto_protective(action: Any) -> bool:
    """Return True for the user-approved protective actions.

    ``action`` may be an :class:`~dome_agent.rules.ActionSpec` or a
    :class:`ModelCommand`.
    """
    node_id = str(getattr(action, "node_id", "") or "").strip()
    name = str(getattr(action, "action", "") or "").strip()
    if name == "turn_off":
        return True
    if (node_id, name) not in _PROTECTIVE_PAIRS:
        return False
    if name == "set_mode":
        return str(getattr(action, "value", "") or "").strip().upper() == "FRESH_AIR"
    if name == "set_fan":
        return str(getattr(action, "value", "") or "").strip().upper() in {"ON", "1", "TRUE"}
    return True


def _command_signature(command: "ModelCommand") -> str:
    value = "<missing>" if command.value is NO_VALUE else command.value
    return json.dumps(
        [command.tool, command.node_id, command.action, command.query, value, command.params],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


@dataclass(frozen=True, slots=True)
class ModelCommand:
    """One validated tool call proposed by the model."""

    tool: str
    node_id: str = ""
    action: str = ""
    value: Any = NO_VALUE
    query: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    to: str = "operator"
    reason: str = ""
    expect: Any | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        tool = str(self.tool or "").strip().casefold()
        if tool not in CONTROL_TOOLS:
            raise ValueError(f"unsupported control tool: {self.tool!r}")
        object.__setattr__(self, "tool", tool)
        object.__setattr__(self, "node_id", str(self.node_id or "").strip())
        object.__setattr__(self, "action", str(self.action or "").strip())
        object.__setattr__(self, "query", str(self.query or "").strip())
        if not isinstance(self.params, Mapping):
            object.__setattr__(self, "params", {})
        else:
            object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "reason", _text(self.reason, 500))
        object.__setattr__(self, "text", _text(self.text, 2000))
        object.__setattr__(self, "to", _text(self.to, 80) or "operator")
        if tool == "control":
            if not self.node_id:
                raise ValueError("control command requires node_id")
            if not self.action:
                raise ValueError("control command requires action")
        if tool == "query" and (not self.node_id or not self.query):
            raise ValueError("query command requires node_id and query")
        if tool == "send_message" and not self.text:
            raise ValueError("send_message requires text")

    @property
    def signature(self) -> str:
        return _command_signature(self)

    @classmethod
    def from_action(cls, action: Any, *, reason: str = "") -> "ModelCommand":
        return cls(
            tool="control",
            node_id=str(getattr(action, "node_id", "")),
            action=str(getattr(action, "action", "")),
            value=getattr(action, "value", NO_VALUE),
            reason=reason or str(getattr(action, "reason", "")),
            expect=getattr(action, "expect", None),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "node_id": self.node_id,
            "action": self.action,
            "value": None if self.value is NO_VALUE else self.value,
            "query": self.query,
            "params": self.params,
            "text": self.text,
            "to": self.to,
            "reason": self.reason,
            "expect": getattr(self.expect, "description", None),
        }


@dataclass(frozen=True, slots=True)
class ModelDecision:
    analysis: str
    operator_message: str
    commands: tuple[ModelCommand, ...]
    risk: str = "medium"
    confidence: float | None = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            "operator_message": self.operator_message,
            "commands": [command.as_dict() for command in self.commands],
            "risk": self.risk,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
        }

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(command.signature for command in self.commands)


def parse_model_decision(content: str) -> ModelDecision | None:
    """Parse and validate a strict model decision.

    Unknown tools and hard-denied actions are dropped and reported as warnings.
    The caller must not execute a dropped command.
    """
    raw = extract_json(content)
    if raw is None:
        return None
    analysis = _text(raw.get("analysis") or raw.get("situation"), 1200)
    operator_message = _text(raw.get("operator_message"), 1800)
    commands_raw = raw.get("commands")
    warnings: list[str] = []
    commands: list[ModelCommand] = []
    if commands_raw is None:
        commands_raw = []
    if not isinstance(commands_raw, list):
        warnings.append("commands must be a list")
        commands_raw = []
    for item in commands_raw[:MAX_COMMANDS]:
        if not isinstance(item, Mapping):
            warnings.append("command entry is not an object")
            continue
        tool = _text(item.get("tool") or "control", 30).casefold()
        action = _text(item.get("action"), 80)
        if tool == "control" and action.casefold() in HARD_DENY_ACTIONS:
            warnings.append(f"hard-denied action dropped: {action}")
            continue
        params = item.get("params")
        if not isinstance(params, Mapping):
            params = {}
        value = item.get("value", NO_VALUE)
        if value is None:
            value = NO_VALUE
        try:
            commands.append(
                ModelCommand(
                    tool=tool,
                    node_id=_text(item.get("node_id"), 120),
                    action=action,
                    value=value,
                    query=_text(item.get("query"), 200),
                    params=dict(params),
                    text=_text(item.get("text"), 2000),
                    to=_text(item.get("to"), 80) or "operator",
                    reason=_text(item.get("reason"), 500),
                )
            )
        except ValueError as exc:
            warnings.append(str(exc))
    if not analysis and not operator_message and not commands:
        return None
    risk = _text(raw.get("risk"), 20).casefold()
    if risk not in {"low", "medium", "high"}:
        risk = "medium"
    confidence = _as_float(raw.get("confidence"))
    if confidence is not None:
        confidence = min(1.0, max(0.0, confidence))
    return ModelDecision(
        analysis=analysis,
        operator_message=operator_message,
        commands=tuple(commands),
        risk=risk,
        confidence=confidence,
        warnings=tuple(warnings),
    )


@dataclass(slots=True)
class PendingProposal:
    """A normal-mode proposal waiting for operator confirmation."""

    commands: tuple[ModelCommand, ...]
    message: str
    created_at: float
    source: str = "llm"
    force: bool = False
    awaiting_force: bool = False
    analysis: str = ""
    risk: str = "medium"

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(command.signature for command in self.commands)

    def expired(self, now: float, ttl_s: float = 300.0) -> bool:
        return (now - self.created_at) >= ttl_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "message": self.message,
            "commands": [command.as_dict() for command in self.commands],
            "force": self.force,
            "awaiting_force": self.awaiting_force,
            "analysis": self.analysis,
            "risk": self.risk,
            "age_s": None,
        }


class DomeTool:
    """Restricted Dome API executor used by the model.

    This class never imports ``subprocess`` and never accepts a shell command.
    """

    def __init__(self, api: DomeAgentAPI) -> None:
        self.api = api

    async def execute(
        self,
        command: ModelCommand,
        *,
        force: bool = False,
        verify: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if command.tool == "control":
            action = command.action.casefold()
            if action in HARD_DENY_ACTIONS:
                return {
                    "ok": False,
                    "blocked": True,
                    "forceable": False,
                    "error": f"action {command.action!r} is hard-denied",
                    "command": command.as_dict(),
                }
            try:
                allow_without_before = is_auto_protective(command)
                if command.value is NO_VALUE:
                    result = await self.api.control_device(
                        command.node_id,
                        command.action,
                        dry_run=dry_run,
                        force=force,
                        verify=verify,
                        allow_without_before=allow_without_before,
                    )
                else:
                    result = await self.api.control_device(
                        command.node_id,
                        command.action,
                        command.value,
                        dry_run=dry_run,
                        force=force,
                        verify=verify,
                        allow_without_before=allow_without_before,
                    )
            except Exception as exc:
                return {
                    "ok": False,
                    "blocked": False,
                    "forceable": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "command": command.as_dict(),
                }
            success = bool(
                result.get("success")
                or result.get("accepted")
                or result.get("dry_run")
            )
            raw_response = result.get("response")
            response_error = (
                raw_response.get("error") if isinstance(raw_response, Mapping) else None
            )
            return {
                "ok": success,
                "blocked": bool(result.get("blocked")),
                "forceable": False,
                "error": result.get("error") or response_error,
                "command": command.as_dict(),
                "result": {
                    "success": success,
                    "accepted": bool(result.get("accepted", success)),
                    "error": result.get("error") or response_error,
                    "dry_run": bool(result.get("dry_run")),
                    "verify_error": result.get("verify_error"),
                    "unknown": bool(result.get("unknown")),
                    "before_error": result.get("before_error"),
                    "after": result.get("after"),
                    "diff": result.get("diff"),
                },
            }
        if command.tool == "query":
            try:
                result = await self.api.query_device(
                    command.node_id,
                    command.query,
                    command.params or None,
                    force=force,
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "blocked": False,
                    "forceable": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "command": command.as_dict(),
                }
            return {"ok": True, "command": command.as_dict(), "result": result}
        if command.tool == "telemetry":
            try:
                result = await self.api.get_telemetry(command.node_id or None)
            except Exception as exc:
                return {
                    "ok": False,
                    "blocked": False,
                    "forceable": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "command": command.as_dict(),
                }
            return {"ok": True, "command": command.as_dict(), "result": result}
        if command.tool == "catalog":
            try:
                result = await self.api.catalog()
            except Exception as exc:
                return {
                    "ok": False,
                    "blocked": False,
                    "forceable": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "command": command.as_dict(),
                }
            return {"ok": True, "command": command.as_dict(), "result": result}
        if command.tool == "send_message":
            try:
                result = await self.api.post_message(
                    command.text,
                    to=command.to,
                    name=None,
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "blocked": False,
                    "forceable": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "command": command.as_dict(),
                }
            return {"ok": True, "command": command.as_dict(), "result": result}
        return {
            "ok": False,
            "blocked": False,
            "forceable": False,
            "error": f"unsupported tool {command.tool!r}",
            "command": command.as_dict(),
        }


def compact_catalog(catalog: Iterable[Mapping[str, Any]], *, limit: int = 40) -> list[dict[str, Any]]:
    """Return a compact catalog safe for an LLM context window."""
    result: list[dict[str, Any]] = []
    for node in catalog:
        if not isinstance(node, Mapping):
            continue
        actions: list[str] = []
        for item in node.get("controls") or []:
            if isinstance(item, Mapping):
                name = item.get("action") or item.get("name")
            else:
                name = item
            if name:
                actions.append(str(name))
        for item in node.get("service_actions") or []:
            if isinstance(item, Mapping):
                name = item.get("action") or item.get("name")
            else:
                name = item
            if name:
                actions.append(str(name))
        result.append(
            {
                "node_id": str(node.get("node_id") or ""),
                "title": str(node.get("title") or node.get("name") or "")[:120],
                "actions": sorted(set(actions))[:40],
            }
        )
        if len(result) >= limit:
            break
    return result
