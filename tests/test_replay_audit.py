"""Offline replay, audit archive and panel log tests."""
from __future__ import annotations

import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dome_agent.audit import write_archive
from dome_agent.replay import INJECTIONS, load_day, replay, replay_day
from dome_web.panel_log import PanelLog

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "Dome_RAG-main" / "Logs"
POLICY = REPO / "dome_policy.competition.json"


@unittest.skipUnless((LOGS / "2026-10-14/telemetry.jsonl").is_file(), "historical logs are absent")
class ReplayTests(unittest.TestCase):
    def test_load_day_counts_frames_and_injected_anomalies(self):
        clean = load_day(LOGS / "2026-10-14")
        self.assertGreater(len(clean.frames), 100)
        self.assertEqual(clean.invalid_lines, 0)
        self.assertFalse(any(frame.anomalies for frame in clean.frames))

        injected = load_day(
            LOGS / "2026-10-14",
            inject=INJECTIONS,
        )
        self.assertGreaterEqual(injected.invalid_lines, 1)
        seen = {name for frame in injected.frames for name in frame.anomalies}
        self.assertTrue({"stale", "duplicate", "out_of_order"} <= seen, seen)

    def test_replay_runs_rules_and_gate(self):
        day = load_day(LOGS / "2026-10-14")
        report = replay_day(day, policy_file=POLICY, competition=True)
        self.assertGreater(report["decisions"], 0)
        self.assertGreater(report["action_requests"], 0)
        self.assertEqual(report["policy_blocked"], 0, report["blocked"])
        self.assertTrue(report["actions"])

    def test_replay_reports_policy_blocks(self):
        policy = {
            "version": 1,
            "default": "deny",
            "deny_nodes": [],
            "deny_actions": [
                "force",
                "reset_interlock",
                "reset_protection",
                "set_data_trust",
                "set_primary_sensor",
            ],
            "allow": {
                "smart_panel_01": {
                    "actions": ["line_on"],
                    "values": {"line_on": [1, 2, 3, 4, 5, 6, 7, 8]},
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            report = replay_day(
                load_day(LOGS / "2026-10-13"),
                policy_file=path,
                competition=True,
            )
        self.assertGreater(report["policy_blocked"], 0, report["blocked"])
        self.assertTrue(any("not allowed" in reason or "not in the policy" in reason for reason in report["blocked"]))

    def test_replay_all_days_is_json_serializable(self):
        report = replay(
            LOGS,
            policy_file=POLICY,
            competition=True,
            days=["2026-10-14"],
        )
        json.dumps(report)
        self.assertEqual(report["totals"]["days"], 1)
        self.assertGreater(report["totals"]["frames"], 0)


class AuditTests(unittest.TestCase):
    def test_archive_contains_manifest_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs"
            log_dir.mkdir()
            log = log_dir / "agent_20260101-000000.jsonl"
            log.write_text('{"kind":"start"}\n', encoding="utf-8")
            state = root / "state.json"
            state.write_text('{"ok":true}\n', encoding="utf-8")
            policy = root / "policy.json"
            policy.write_text('{"default":"deny","allow":{}}\n', encoding="utf-8")
            out = root / "audit.tar.gz"
            with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "super-secret"}, clear=False):
                path = write_archive(
                    out,
                    log_dir=log_dir,
                    state_file=state,
                    policy_file=policy,
                    kb_dir=root / "missing-kb",
                )
            self.assertTrue(path.is_file())
            with tarfile.open(path, "r:gz") as archive:
                names = set(archive.getnames())
                manifest = json.loads(
                    archive.extractfile("MANIFEST.json").read().decode("utf-8")
                )
            self.assertIn("logs/agent.jsonl", names)
            self.assertIn("state/agent_state.json", names)
            self.assertIn("config/dome_policy.json", names)
            self.assertTrue(manifest["files"]["logs/agent.jsonl"]["sha256"])
            self.assertNotIn("super-secret", json.dumps(manifest))
            self.assertNotIn("DEEPSEEK_API_KEY", manifest["env"])


class PanelLogTests(unittest.TestCase):
    def test_panel_conversation_writes_jsonl(self):
        import asyncio
        from dome_web.service import DashboardService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = DashboardService(root, demo=True, env_file="absent.env")
            try:
                result = asyncio.run(service.conversation.ask("Что с батареей?"))
                self.assertTrue(result["ok"])
            finally:
                service.close()
            logs = list((root / "logs").glob("panel_*.jsonl"))
            self.assertTrue(logs, list((root / "logs").iterdir()))
            records = [
                json.loads(line)
                for line in logs[0].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["kind"], "chat")
            self.assertIn("батаре", records[0]["question"])

    def test_panel_log_is_append_only_and_redacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = PanelLog(tmp, redact=lambda row: {**row, "answer": str(row.get("answer", "")).replace("key", "[REDACTED]")})
            try:
                log.write("chat", question="q", answer="key")
                log.write("chat", question="q2", answer="ok")
            finally:
                log.close()
            records = [
                json.loads(line)
                for line in log.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["seq"] for record in records], [1, 2])
            self.assertEqual(records[0]["answer"], "[REDACTED]")

    def test_panel_log_picks_a_free_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = PanelLog(tmp, run_id="same")
            first.close()
            second = PanelLog(tmp, run_id="same")
            try:
                self.assertNotEqual(first.path, second.path)
            finally:
                second.close()


if __name__ == "__main__":
    unittest.main()
