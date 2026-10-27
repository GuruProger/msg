"""Fail-closed competition profile tests.  No network, no live dome."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dome_agent.config import AgentConfig
from dome_agent.gate import (
    CommandGate,
    GateError,
    load_policy,
    validate_competition_policy,
)
from dome_agent.rules import ActionSpec

REPO = Path(__file__).resolve().parents[1]
COMPETITION_POLICY = REPO / "dome_policy.competition.json"
DEV_POLICY = REPO / "dome_policy.json"


def action(node: str, name: str, value=1) -> ActionSpec:
    return ActionSpec(key=f"{node}:{name}", node_id=node, action=name, value=value, reason="test")


class CompetitionPolicyTests(unittest.TestCase):
    def test_competition_policy_is_safe(self):
        policy = load_policy(COMPETITION_POLICY)
        self.assertEqual(validate_competition_policy(policy), [])

    def test_development_policy_is_rejected_for_scored_run(self):
        policy = load_policy(DEV_POLICY)
        errors = validate_competition_policy(policy)
        self.assertTrue(errors)
        joined = " ".join(errors)
        for name in (
            "reset_interlock",
            "reset_protection",
            "set_data_trust",
            "set_primary_sensor",
        ):
            self.assertIn(name, joined)

    def test_gate_denies_hazardous_and_force_actions(self):
        gate = CommandGate(
            COMPETITION_POLICY,
            control_enabled=True,
            competition=True,
        )
        for name in (
            "force",
            "reset_interlock",
            "reset_protection",
            "set_data_trust",
            "set_primary_sensor",
        ):
            decision = gate.check(action("smart_panel_01", name))
            self.assertFalse(decision.allowed, name)
            self.assertIn("competition", decision.reason)

    def test_gate_allows_declared_safe_action(self):
        gate = CommandGate(
            COMPETITION_POLICY,
            control_enabled=True,
            competition=True,
        )
        decision = gate.check(action("smart_panel_01", "line_off", 8))
        self.assertTrue(decision.allowed, decision.reason)

    def test_gate_rejects_yolo_in_competition(self):
        with self.assertRaises(GateError):
            CommandGate(
                COMPETITION_POLICY,
                control_enabled=True,
                competition=True,
                yolo=True,
            )


class CompetitionConfigTests(unittest.TestCase):
    def base(self, **overrides):
        values = {
            "dome_url": "ws://127.0.0.1:1",
            "dome_key": "test",
            "policy_file": COMPETITION_POLICY,
            "competition": True,
            "mode": "auto",
            "control_enabled": True,
            "llm_enabled": False,
        }
        values.update(overrides)
        return AgentConfig(**values)

    def test_rules_only_profile_is_accepted(self):
        config = self.base()
        self.assertTrue(config.competition)
        self.assertFalse(config.competition_profile()["llm_enabled"])
        self.assertEqual(
            config.competition_profile()["policy_sha256"],
            __import__("hashlib").sha256(COMPETITION_POLICY.read_bytes()).hexdigest(),
        )

    def test_yolo_is_rejected(self):
        with self.assertRaises(ValueError):
            self.base(mode="yolo")
        with self.assertRaises(ValueError):
            self.base(yolo_requested=True)

    def test_non_allowlisted_model_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.base(
                llm_enabled=True,
                deepseek_api_key="key",
                deepseek_model="deepseek-chat",
            )
        self.assertIn("deepseek-chat", str(ctx.exception))

    def test_allowlisted_model_is_accepted(self):
        config = self.base(
            llm_enabled=True,
            deepseek_api_key="key",
            deepseek_model="deepseek-v4-flash",
        )
        self.assertTrue(config.competition_profile()["llm_enabled"])

    def test_non_deepseek_base_url_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.base(
                llm_enabled=True,
                deepseek_api_key="key",
                deepseek_model="deepseek-v4-flash",
                deepseek_base_url="https://example.com",
            )
        self.assertIn("DeepSeek", str(ctx.exception))

    def test_unsafe_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            self.base(policy_file=DEV_POLICY)

    def test_missing_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.base(policy_file=Path(tmp) / "missing.json")

    def test_env_yolo_is_rejected_even_with_normal_mode_override(self):
        env = {
            "AGENT_COMPETITION": "1",
            "AGENT_YOLO": "1",
            "DOME_URL": "ws://127.0.0.1:1",
            "DOME_KEY": "test",
            "AGENT_POLICY_FILE": str(COMPETITION_POLICY),
            "AGENT_LLM": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaises(ValueError) as ctx:
                AgentConfig.from_env(env_file=None, overrides={"mode": "normal"})
        self.assertIn("yolo", str(ctx.exception).casefold())

    def test_safe_dict_exposes_only_non_secret_profile(self):
        config = self.base()
        safe = config.safe_dict()
        self.assertTrue(safe["competition"])
        self.assertEqual(safe["competition_models"], ["deepseek-v4-flash"])
        self.assertNotIn("dome_key", safe)
        self.assertNotIn("deepseek_api_key", safe)

    def test_competition_policy_file_is_json_object(self):
        data = json.loads(COMPETITION_POLICY.read_text(encoding="utf-8"))
        self.assertEqual(data["default"], "deny")
        self.assertIn("dome_automation_01", data["deny_nodes"])


if __name__ == "__main__":
    unittest.main()
