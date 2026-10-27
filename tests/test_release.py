"""Release pre-flight tests.  They do not call the network."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from dome_agent import release

BASE_ENV = {
    "AGENT_COMPETITION": "1",
    "AGENT_MODE": "auto",
    "AGENT_YOLO": "0",
    "AGENT_LLM": "0",
    "AGENT_POLICY_FILE": "dome_policy.competition.json",
    "DOME_URL": "ws://127.0.0.1:1",
    "DOME_KEY": "test",
    "DOME_KEY_FILE": "",
}


class ReleaseCheckTests(unittest.TestCase):
    def test_competition_profile_passes_preflight(self):
        with mock.patch.dict(os.environ, BASE_ENV, clear=False):
            report = release.check_all(run_tests=False)
        failed = [item for item in report["checks"] if item["required"] and not item["ok"]]
        self.assertEqual(failed, [])
        self.assertTrue(report["ok"])

    def test_competition_yolo_fails_preflight(self):
        env = dict(BASE_ENV, AGENT_YOLO="1")
        with mock.patch.dict(os.environ, env, clear=False):
            report = release.check_all(run_tests=False)
        self.assertFalse(report["ok"])
        names = {item["name"] for item in report["checks"] if item["required"] and not item["ok"]}
        self.assertIn("competition:no-yolo", names)

    def test_unsafe_policy_fails_preflight(self):
        env = dict(BASE_ENV, AGENT_POLICY_FILE="dome_policy.json")
        with mock.patch.dict(os.environ, env, clear=False):
            report = release.check_all(run_tests=False)
        self.assertFalse(report["ok"])
        names = {item["name"] for item in report["checks"] if item["required"] and not item["ok"]}
        self.assertIn("policy:competition-safe", names)

    def test_probe_skips_without_key(self):
        env = {"DEEPSEEK_API_KEY": "", "DEEPSEEK_MODEL": "deepseek-v4-flash"}
        with mock.patch.dict(os.environ, env, clear=False):
            report = release.probe_model()
        self.assertEqual(report["status"], "skipped")
        self.assertIn("AGENT_LLM=0", report["reason"])


if __name__ == "__main__":
    unittest.main()
