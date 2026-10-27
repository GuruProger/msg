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

from .config import COMPETITION_DENY_ACTIONS, file_sha256
from .rules import ActionSpec


class GateError(RuntimeError):
    """Raised when the policy cannot be loaded or is malformed."""


def load_policy(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a policy file."""
    policy_file = Path(path)
    if not policy_file.exists():
        raise GateError(f"policy file not found: {policy_file}")
    try:
        data = json.loads(policy_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read policy {policy_file}: {exc}") from exc
    if not isinstance(data, dict):
        raise GateError(f"policy {policy_file} must contain a JSON object")
    if data.get("default", "deny") != "deny":
        raise GateError("policy.default must be 'deny'")
    if not isinstance(data.get("allow"), Mapping):
        raise GateError("policy.allow must be an object")
    return data


def policy_digest(path: str | Path) -> str | None:
    """SHA-256 of the exact policy bytes, for the start log."""
    return file_sha256(path)


def _allowed_pairs(policy: Mapping[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    allow = policy.get("allow")
    if not isinstance(allow, Mapping):
        return pairs
    for node_id, rule in allow.items():
        if isinstance(rule, Mapping):
            actions = rule.get("actions", [])
        else:
            actions = rule
        if isinstance(actions, (str, bytes)) or not isinstance(actions, (list, tuple, set)):
            continue
        for action in actions:
            pairs.add((str(node_id), str(action)))
    return pairs


def validate_competition_policy(policy: Mapping[str, Any]) -> list[str]:
    """Return reasons why ``policy`` is not acceptable for a scored run."""
    errors: list[str] = []
    if policy.get("default", "deny") != "deny":
        errors.append("policy.default must be 'deny'")
    allow = policy.get("allow")
    if not isinstance(allow, Mapping):
        errors.append("policy.allow must be an object")
        allow = {}

    denied = {str(value) for value in policy.get("deny_actions", [])}
    required = set(COMPETITION_DENY_ACTIONS)
    missing = sorted(required - denied)
    if missing:
        errors.append(
            "policy.deny_actions must list: " + ", ".join(missing)
        )

    allowed_actions = {action for _, action in _allowed_pairs(policy)}
    forbidden = sorted(allowed_actions & required)
    if forbidden:
        errors.append(
            "policy must not allow hazardous actions: " + ", ".join(forbidden)
        )
    return errors


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
        competition: bool = False,
    ) -> None:
        self.policy_file = Path(policy_file)
        self.control_enabled = bool(control_enabled)
        self.yolo = bool(yolo)
        self.competition = bool(competition)
        self._policy: dict[str, Any] = {}
        if self.competition or (self.control_enabled and not self.yolo):
            self._policy = self._load_policy()
        if self.competition and self.yolo:
            raise GateError("competition mode forbids YOLO")
        if self.competition:
            errors = validate_competition_policy(self._policy)
            if errors:
                raise GateError("unsafe competition policy: " + "; ".join(errors))

    def apply_runtime(self, *, yolo: bool, control_enabled: bool) -> None:
        """Apply a validated runtime mode without rebuilding the gate.

        A live mode change must never turn a scored run into YOLO, and turning
        YOLO off must load the policy if it was bypassed at construction time.
        """
        if self.competition and yolo:
            raise GateError("competition mode forbids YOLO")
        self.yolo = bool(yolo)
        self.control_enabled = bool(control_enabled)
        if not self.yolo and self.control_enabled and not self._policy:
            self._policy = self._load_policy()

    def check(self, action: ActionSpec, *, dry_run: bool = False) -> GateDecision:
        """Return whether ``action`` may be executed."""
        if dry_run:
            return GateDecision(True, "dry-run: no command is sent")
        if str(action.action) == "stop_job":
            return GateDecision(
                False,
                "action 'stop_job' is not allowed: the Dome API has no stop_job; "
                "a confirmed line_off is the only stop path",
            )
        if self.competition and str(action.action) in COMPETITION_DENY_ACTIONS:
            return GateDecision(
                False,
                f"action {str(action.action)!r} is disabled by the competition profile",
            )
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
        return load_policy(self.policy_file)
