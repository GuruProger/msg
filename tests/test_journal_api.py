"""Tests for the read-only journal API."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from dome_web.journal_api import create_app

RUN_ID = "20261027-120000"
DOME_SECRET = "dome-secret-value"
DEEPSEEK_SECRET = "deepseek-secret-value"
TOKEN_SECRET = "token-secret-value"


class DashboardTests(unittest.TestCase):
    """Exercise the dashboard against temporary log and state files."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log_dir = self.root / "logs"
        self.log_dir.mkdir()
        self.state_file = self.log_dir / "agent_state.json"
        self.app = create_app(log_dir=self.log_dir, state_file=self.state_file)
        self.client = TestClient(self.app)

    @staticmethod
    def _ts(seconds_ago: float = 0.0) -> str:
        return (datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat(
            timespec="milliseconds"
        )

    def _record(self, kind: str, seq: int, seconds_ago: float, **fields: object) -> dict:
        record = {
            "ts": self._ts(seconds_ago),
            "run_id": RUN_ID,
            "seq": seq,
            "kind": kind,
        }
        record.update(fields)
        return record

    def _write_log(self, records: list[dict], name: str = f"agent_{RUN_ID}.jsonl") -> Path:
        path = self.log_dir / name
        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )
        return path

    def _default_records(self) -> list[dict]:
        return [
            self._record(
                "start",
                1,
                100,
                agent="team3-agent",
                version="abc1234",
                models=["deepseek-chat"],
                config={
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
                    "dome_key": DOME_SECRET,
                    "deepseek_api_key": DEEPSEEK_SECRET,
                    "token": TOKEN_SECRET,
                },
            ),
            self._record(
                "connection",
                2,
                98,
                event="auth",
                role="team",
                team=3,
            ),
            self._record(
                "observation",
                3,
                90,
                dome_time="14:02",
                summary="Печать идёт, CO2 в норме.",
                metrics={
                    "co2_ppm": 810,
                    "soc_pct": 67,
                    "printer": "PRINTING",
                    "cnc": "IDLE",
                    "fume": "ON",
                    "power_w": 420,
                },
                anomalies=["CO2 растёт быстрее обычного"],
            ),
            self._record(
                "decision",
                4,
                80,
                situation="Продолжаем печать",
                cause="Параметры в норме",
                confidence=0.86,
                plan=["наблюдать за CO2", "проверить печать"],
                source="mixed",
                evidence=[{"node": "climate_sensor_02", "param": "co2_ppm", "value": 790}],
                ref=120,
            ),
            self._record(
                "action",
                5,
                70,
                request={
                    "type": "control",
                    "node_id": "smart_panel_01",
                    "action": "line_off",
                    "value": 8,
                },
                response={"success": True, "error": None},
                ref=120,
            ),
            self._record(
                "verification",
                6,
                65,
                ref=5,
                expected="smart_panel_01.line 8 = OFF",
                observed={"line": 8, "state": "OFF"},
                ok=True,
            ),
            self._record(
                "message",
                7,
                50,
                direction="in",
                sender="Оператор",
                role="operator",
                team=3,
                to=None,
                text="Проверь печать и доложи.",
            ),
            self._record(
                "action_blocked",
                8,
                40,
                request={
                    "type": "control",
                    "node_id": "printer_3d_01",
                    "action": "start_job",
                    "value": "part.gcode",
                },
                reason="policy requires confirmation",
                ref=120,
            ),
            self._record(
                "error",
                9,
                20,
                where="llm_advice",
                error="DeepSeekTimeout: advisory call timed out",
            ),
        ]

    def _write_state(self, payload: dict) -> None:
        self.state_file.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def test_summary_contains_run_metrics_actions_and_alerts(self) -> None:
        self._write_log(self._default_records())
        self._write_state(
            {
                "task": {"text": "Напечатать деталь", "status": "in_progress"},
                "last_message_id": 8,
                "co2_override": None,
                "last_actions": {"smart_panel_01:line_off": {"at": 1.0, "success": True}},
                "failed_until": {},
                "notes": {"shift": "night"},
            }
        )

        response = self.client.get("/api/summary")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        data = response.json()
        self.assertFalse(data["demo"])
        self.assertEqual(data["health"]["status"], "ok")
        self.assertEqual(data["health"]["log_file"], f"agent_{RUN_ID}.jsonl")

        run = data["run"]
        self.assertEqual(run["run_id"], RUN_ID)
        self.assertTrue(run["file"].endswith(f"agent_{RUN_ID}.jsonl"))
        self.assertIsNotNone(run["started_at"])
        self.assertFalse(run["stopped"])
        self.assertIsNone(run["stop_reason"])
        self.assertEqual(run["agent"]["name"], "team3-agent")
        self.assertEqual(run["agent"]["version"], "abc1234")
        self.assertEqual(run["agent"]["models"], ["deepseek-chat"])
        self.assertEqual(run["config"]["mode"], "normal")
        self.assertTrue(run["config"]["allow_control"])
        self.assertFalse(run["config"]["dry_run"])
        self.assertEqual(run["role"], "team")
        self.assertEqual(run["team"], 3)

        self.assertEqual(data["state"]["last_message_id"], 8)
        self.assertEqual(data["state"]["task"]["status"], "in_progress")

        self.assertEqual(data["observation"]["dome_time"], "14:02")
        self.assertEqual(data["metrics"]["co2_ppm"], 810)
        self.assertEqual(data["metrics"]["fume"], "ON")
        self.assertEqual(data["decision"]["confidence"], 0.86)
        self.assertEqual(data["decision"]["ref"], 120)

        self.assertEqual(len(data["actions"]), 1)
        action = data["actions"][0]
        self.assertEqual(action["ref"], 120)
        self.assertEqual(action["response"]["success"], True)
        self.assertIsNotNone(action["verification"])
        self.assertEqual(action["verification"]["ref"], 5)
        self.assertTrue(action["verification"]["ok"])
        self.assertEqual(len(data["verifications"]), 1)

        self.assertEqual(len(data["messages"]), 1)
        self.assertEqual(data["messages"][0]["sender"]["role"], "operator")
        self.assertEqual(data["messages"][0]["sender"]["name"], "Оператор")

        self.assertEqual(
            data["counts"],
            {
                "records": 9,
                "decisions": 1,
                "actions": 1,
                "verifications": 1,
                "messages": 1,
                "errors": 1,
            },
        )

        alert_titles = {alert["title"] for alert in data["alerts"]}
        self.assertIn("Agent error", alert_titles)
        self.assertIn("Action blocked", alert_titles)
        self.assertIn("Observation anomaly", alert_titles)
        self.assertLessEqual(len(data["alerts"]), 20)

        self.assertEqual(len(data["timeline"]), 9)
        kinds = [item["kind"] for item in data["timeline"]]
        self.assertEqual(
            kinds,
            [
                "start",
                "connection",
                "observation",
                "decision",
                "action",
                "verification",
                "message",
                "action_blocked",
                "error",
            ],
        )

    def test_secrets_are_never_returned(self) -> None:
        self._write_log(self._default_records())

        summary = self.client.get("/api/summary")
        logs = self.client.get("/api/logs")

        self.assertEqual(summary.status_code, 200)
        self.assertEqual(logs.status_code, 200)
        for secret in (DOME_SECRET, DEEPSEEK_SECRET, TOKEN_SECRET):
            self.assertNotIn(secret, summary.text)
            self.assertNotIn(secret, logs.text)

        config = summary.json()["run"]["config"]
        self.assertEqual(config["mode"], "normal")
        self.assertEqual(config["dome_url"], "wss://dome.example/ws")
        self.assertEqual(config["name"], "team3-agent")
        for forbidden in ("dome_key", "deepseek_api_key", "token", "secret", "password"):
            self.assertNotIn(forbidden, config)

        for record in logs.json()["records"]:
            if record.get("kind") == "start":
                self.assertNotIn("dome_key", record["config"])
                self.assertNotIn("deepseek_api_key", record["config"])
                self.assertNotIn("token", record["config"])

    def test_logs_kind_filter_limit_and_chronological_order(self) -> None:
        records = [
            self._record("decision", 1, 40, situation="first"),
            self._record("message", 2, 30, direction="in", text="hello"),
            self._record("decision", 3, 20, situation="second"),
            self._record("decision", 4, 10, situation="third"),
        ]
        self._write_log(records)

        response = self.client.get(
            "/api/logs", params={"kind": "decision", "limit": 2}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["run_id"], RUN_ID)
        self.assertTrue(data["file"].endswith(f"agent_{RUN_ID}.jsonl"))
        self.assertEqual(data["count"], 2)
        self.assertEqual([item["seq"] for item in data["records"]], [3, 4])
        timestamps = [item["ts"] for item in data["records"]]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertFalse(data["truncated"])

        unfiltered = self.client.get("/api/logs", params={"limit": 2}).json()
        self.assertEqual([item["seq"] for item in unfiltered["records"]], [3, 4])

    def test_logs_reject_run_id_traversal(self) -> None:
        self._write_log(self._default_records())

        response = self.client.get(
            "/api/logs", params={"run_id": "../../etc/passwd"}
        )

        self.assertEqual(response.status_code, 400)

    def test_empty_log_directory_is_waiting(self) -> None:
        health = self.client.get("/api/health")
        summary = self.client.get("/api/summary")

        self.assertEqual(health.status_code, 200)
        health_data = health.json()
        self.assertEqual(health_data["status"], "waiting")
        self.assertFalse(health_data["has_log"])
        self.assertFalse(health_data["has_state"])
        self.assertIsNone(health_data["log_file"])
        self.assertIsNone(health_data["last_record_age_s"])
        self.assertEqual(health_data["records_loaded"], 0)

        self.assertEqual(summary.status_code, 200)
        data = summary.json()
        self.assertEqual(data["health"]["status"], "waiting")
        self.assertIsNone(data["run"]["run_id"])
        self.assertIsNone(data["run"]["file"])
        self.assertIsNone(data["run"]["started_at"])
        self.assertIsNone(data["run"]["last_ts"])
        self.assertEqual(data["run"]["agent"], {"name": None, "version": None, "models": []})
        self.assertEqual(set(data["run"]["config"]), {
            "mode",
            "allow_control",
            "dry_run",
            "dome_url",
            "name",
            "policy",
            "llm",
            "run_seconds",
            "observation_interval",
            "min_send_gap",
        })
        self.assertTrue(all(value is None for value in data["run"]["config"].values()))
        self.assertIsNone(data["observation"])
        self.assertIsNone(data["decision"])
        self.assertEqual(data["alerts"], [])
        self.assertEqual(data["actions"], [])
        self.assertEqual(data["verifications"], [])
        self.assertEqual(data["messages"], [])
        self.assertEqual(data["timeline"], [])
        self.assertEqual(data["counts"]["records"], 0)
        self.assertTrue(all(value is None for value in data["metrics"].values()))

    def test_corrupt_and_partial_lines_are_tolerated(self) -> None:
        start = self._record("start", 1, 30, agent="team3-agent", version="abc1234")
        observation = self._record(
            "observation",
            2,
            10,
            summary="still running",
            metrics={"co2_ppm": 700, "soc_pct": 55},
            anomalies=[],
        )
        path = self.log_dir / f"agent_{RUN_ID}.jsonl"
        path.write_text(
            json.dumps(start)
            + "\n{this is not json\n"
            + json.dumps(observation)
            + '\n{"ts": "2026-10-27T12:00:00+03:00", "kind":',
            encoding="utf-8",
        )

        health = self.client.get("/api/health").json()
        summary = self.client.get("/api/summary").json()

        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["records_loaded"], 2)
        self.assertEqual(summary["counts"]["records"], 2)
        self.assertEqual(summary["observation"]["summary"], "still running")
        self.assertEqual(summary["metrics"]["co2_ppm"], 700)
        self.assertEqual(summary["run"]["agent"]["name"], "team3-agent")

    def test_log_cache_refreshes_when_file_grows(self) -> None:
        path = self._write_log(
            [self._record("start", 1, 30, agent="team3-agent", version="abc1234")]
        )
        first = self.client.get("/api/health").json()
        self.assertEqual(first["records_loaded"], 1)

        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    self._record(
                        "observation",
                        2,
                        5,
                        summary="new observation",
                        metrics={"co2_ppm": 700},
                        anomalies=[],
                    )
                )
                + "\n"
            )

        second = self.client.get("/api/summary").json()
        self.assertEqual(second["health"]["records_loaded"], 2)
        self.assertEqual(second["counts"]["records"], 2)
        self.assertEqual(second["observation"]["summary"], "new observation")

    def test_demo_is_complete_and_populated(self) -> None:
        response = self.client.get("/api/demo")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        data = response.json()
        self.assertTrue(data["demo"])
        self.assertEqual(data["health"]["status"], "ok")
        self.assertTrue(data["health"]["has_log"])
        self.assertIsNotNone(data["run"]["run_id"])
        self.assertIsNotNone(data["run"]["agent"]["name"])
        self.assertIsNotNone(data["run"]["config"]["dome_url"])
        self.assertIsNotNone(data["state"]["task"])
        self.assertEqual(data["metrics"]["co2_ppm"], 810)
        self.assertGreaterEqual(len(data["alerts"]), 1)
        self.assertGreaterEqual(len(data["actions"]), 1)
        self.assertGreaterEqual(len(data["verifications"]), 1)
        self.assertGreaterEqual(len(data["messages"]), 1)
        self.assertGreaterEqual(len(data["timeline"]), 5)
        self.assertGreater(data["counts"]["records"], 0)

    def test_state_is_sanitized_and_missing_state_is_tolerated(self) -> None:
        self._write_state(
            {
                "task": {
                    "text": "Напечатать деталь",
                    "status": "in_progress",
                    "api_key": "STATE_API_KEY",
                },
                "last_message_id": 8,
                "co2_override": {
                    "node_id": "climate_sensor_02",
                    "at": 1.0,
                    "reason": "manual",
                    "token": "STATE_TOKEN",
                },
                "last_actions": {
                    "smart_panel_01:line_off": {
                        "at": 2.0,
                        "success": True,
                        "dome_key": "STATE_DOME_KEY",
                    }
                },
                "failed_until": {"printer_3d_01:start_job": 3.0},
                "notes": {"shift": "night", "password": "STATE_PASSWORD"},
                "handled_message_ids": [1, 2, 3],
                "updated_at": "2026-10-27T12:00:00+03:00",
            }
        )

        response = self.client.get("/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        data = response.json()
        self.assertEqual(
            set(data),
            {
                "task",
                "last_message_id",
                "co2_override",
                "last_actions",
                "failed_until",
                "notes",
                "updated_at",
            },
        )
        self.assertEqual(data["task"]["text"], "Напечатать деталь")
        self.assertEqual(data["task"]["status"], "in_progress")
        self.assertEqual(data["last_message_id"], 8)
        self.assertEqual(data["notes"], {"shift": "night"})
        self.assertEqual(data["updated_at"], "2026-10-27T12:00:00+03:00")
        self.assertNotIn("handled_message_ids", data)
        for value in ("STATE_API_KEY", "STATE_TOKEN", "STATE_DOME_KEY", "STATE_PASSWORD"):
            self.assertNotIn(value, response.text)
        self.assertNotIn("api_key", data["task"])
        self.assertNotIn("token", data["co2_override"])
        self.assertNotIn("dome_key", data["last_actions"]["smart_panel_01:line_off"])
        self.assertNotIn("password", data["notes"])

        self.state_file.unlink()
        missing = self.client.get("/api/state")
        self.assertEqual(missing.status_code, 200)
        self.assertEqual(missing.json()["task"], None)
        self.assertEqual(missing.json()["last_message_id"], 0)
        self.assertEqual(missing.json()["last_actions"], {})
        self.assertEqual(missing.json()["failed_until"], {})
        self.assertEqual(missing.json()["notes"], {})
        self.assertIsNone(missing.json()["updated_at"])

    def test_missing_state_keeps_summary_shape(self) -> None:
        self._write_log(self._default_records())

        data = self.client.get("/api/summary").json()

        self.assertEqual(data["state"]["task"], None)
        self.assertEqual(data["state"]["last_message_id"], 0)
        self.assertEqual(data["state"]["last_actions"], {})
        self.assertEqual(data["state"]["failed_until"], {})
        self.assertEqual(data["state"]["notes"], {})
        self.assertIsNone(data["state"]["updated_at"])


if __name__ == "__main__":
    unittest.main()
