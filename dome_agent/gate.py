"""Fail-closed command gate for the autonomous Dome agent.

The rule engine decides *what* should happen.  The gate decides whether that
action is allowed to leave the process.  It is deliberately independent from
the LLM and from the transport layer: every control action passes through it.

Modes
-----
safe (default)
    Read-only unless ``AGENT_ALLOW_CONTROL=1``.  When control is enabled, only
    node/action pairs listed in ``dome_policy.json`` are allowed.
yolo
    Explicit unsafe mode (``AGENT_YOLO=1``).  The policy and cooldowns are
    bypassed.  Transport authentication, rate limiting, and audit logging stay
    in place because they are protocol/audit guarantees, not safety policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dome_agent.rules import ActionSpec


class GateError(RuntimeError):
    """Raised when the policy cannot be loaded or is malformed."""


@dataclass(frozen=True, slots=True)
class GateDecision:
    allowed: bool
    reason: str
    unsafe: bool = False


class CommandGate:
    """Policy-backed command gate."""

    def __init__(
        self,
        policy_file: Path,
        *,
        control_enabled: bool = False,
        yolo: bool = False,
    ) -> None:
        self.policy_file = Path(policy_file)
        self.control_enabled = bool(control_enabled)
        self.yolo = bool(yolo)
        self._policy: dict[str, Any] = {}
        if self.control_enabled and not self.yolo:
            self._policy = self._load_policy()

    def check(self, action: ActionSpec, *, dry_run: bool = False) -> GateDecision:
        """Return whether ``action`` may be executed."""
        if dry_run:
            return GateDecision(True, "dry-run: no command is sent")
        if self.yolo:
            return GateDecision(True, "YOLO: policy and cooldowns bypassed", unsafe=True)
        if not self.control_enabled:
            return GateDecision(
                False,
                "control disabled: set AGENT_ALLOW_CONTROL=1 to send commands",
            )

        node_id = str(action.node_id)
        action_name = str(action.action)
        deny_actions = {str(value) for value in self._policy.get("deny_actions", [])}
        deny_nodes = {str(value) for value in self._policy.get("deny_nodes", [])}
        if action_name in deny_actions:
            return GateDecision(False, f"action {action_name!r} is denied by policy")
        if node_id in deny_nodes:
            return GateDecision(False, f"node {node_id!r} is denied by policy")

        allow = self._policy.get("allow")
        if not isinstance(allow, Mapping) or node_id not in allow:
            return GateDecision(False, f"node {node_id!r} is not in the policy allowlist")

        rule = allow[node_id]
        if isinstance(rule, Mapping):
            actions = rule.get("actions", [])
            values = rule.get("values", {})
        else:
            actions = rule
            values = {}
        if action_name not in {str(value) for value in actions}:
            return GateDecision(
                False,
                f"action {action_name!r} is not allowed for node {node_id!r}",
            )

        if isinstance(values, Mapping) and action_name in values:
            allowed_values = values[action_name]
            if isinstance(allowed_values, list):
                if action.value not in allowed_values:
                    return GateDecision(
                        False,
                        f"value {action.value!r} is not allowed for {node_id}.{action_name}",
                    )

        return GateDecision(True, f"allowed by policy: {node_id}.{action_name}")

    def allowed_summary(self) -> dict[str, Any]:
        """Return a compact policy summary for the model context."""
        if not self._policy:
            return {}
        allow = self._policy.get("allow")
        if not isinstance(allow, Mapping):
            return {}
        summary: dict[str, Any] = {}
        for node_id, rule in allow.items():
            if isinstance(rule, Mapping):
                summary[str(node_id)] = {
                    "actions": list(rule.get("actions") or []),
                    "values": dict(rule.get("values") or {}),
                }
            else:
                summary[str(node_id)] = {"actions": list(rule)}
        return {
            "default": "deny",
            "allow": summary,
            "deny_actions": list(self._policy.get("deny_actions") or []),
            "deny_nodes": list(self._policy.get("deny_nodes") or []),
        }

    def _load_policy(self) -> dict[str, Any]:
        if not self.policy_file.exists():
            raise GateError(f"policy file not found: {self.policy_file}")
        try:
            data = json.loads(self.policy_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GateError(f"cannot read policy {self.policy_file}: {exc}") from exc
        if not isinstance(data, dict):
            raise GateError(f"policy {self.policy_file} must contain a JSON object")
        if data.get("default", "deny") != "deny":
            raise GateError("policy.default must be 'deny'")
        if not isinstance(data.get("allow"), Mapping):
            raise GateError("policy.allow must be an object")
        return data
