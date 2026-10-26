"""Unit and integration tests for the autonomous Dome agent."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx

from dome_control.client import MISSING
from dome_agent.agent import DomeAgent
from dome_agent.analysis import Snapshot, analyze_co2
from dome_agent.chat import (
    ChatMessage,
    check_cnc_completion,
    check_print_completion,
    new_operator_messages,
    parse_operator_task,
)
from dome_agent.config import AgentConfig
from dome_agent.llm import DeepSeekAdvisor, parse_advice
from dome_agent.logging import AgentLog
from dome_agent.rules import ActionSpec, RuleEngine
from dome_agent.state import StateStore

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests/fixtures/telemetry.jsonl"


def base_telemetry() -> dict:
    with FIXTURE.open(encoding="utf-8") as handle:
        return copy.deepcopy(json.loads(handle.readline()))


def false_co2_telemetry() -> dict:
    data = base_telemetry()
    data["nodes"]["climate_sensor_01"]["co2_ppm"] = 3043
    data["nodes"]["dome_automation_01"]["interlocks"]["IL_CO2"].update(
        {"state": "TRIPPED", "reason": "CO2"}
    )
    data["nodes"]["printer_3d_01"].update(
        {"state": "PAUSED", "paused": True, "hold_level": "WARNING", "hold_reason": "IL_CO2"}
    )
    return data


def real_co2_telemetry() -> dict:
    data = base_telemetry()
    for node_id, ppm in (
        ("climate_sensor_01", 2300),
        ("climate_sensor_02", 2350),
        ("climate_sensor_03", 2280),
    ):
        data["nodes"][node_id]["co2_ppm"] = ppm
    data["nodes"]["dome_automation_01"]["interlocks"]["IL_CO2"]["state"] = "TRIPPED"
    return data


def make_config(tmp: str, **overrides) -> AgentConfig:
    values = {
        "dome_url": "ws://127.0.0.1:8765",
        "dome_key": "dome-secret-key",
        "deepseek_api_key": None,
        "log_dir": Path(tmp) / "logs",
        "state_file": Path(tmp) / "logs" / "agent_state.json",
        "observation_interval_s": 5.0,
        "min_send_gap_s": 0.2,
        "run_seconds": 0.6,
    }
    values.update(overrides)
    return AgentConfig(**values)


def read_logs(log_dir: Path) -> list[dict]:
    paths = sorted(log_dir.glob("agent_*.jsonl"))
    assert len(paths) == 1, paths
    return [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines() if line]


class FakeAPI:
    def __init__(self, telemetry: dict, messages: list[dict] | None = None) -> None:
        self.telemetry = telemetry
        self.messages = list(messages or [])
        self.actions: list[dict] = []
        self.sent_messages: list[dict] = []
        self.closed = False
        self.client = SimpleNamespace(role="team", team=3, url="ws://fake")

    async def connect(self) -> dict:
        return {"type": "auth_response", "success": True, "role": "team", "team": 3}

    async def close(self) -> None:
        self.closed = True

    async def whoami(self) -> dict:
        return {"url": "ws://fake", "role": "team", "team": 3, "name": "test"}

    async def get_telemetry(
        self,
        node_id: str | None = None,
        *,
        max_age: float = 5.0,
        next_frame: bool = False,
    ) -> dict:
        if node_id is not None:
            node = self.telemetry.get("nodes", {}).get(node_id)
            if node is not None:
                return {"telemetry": copy.deepcopy(node)}
        return copy.deepcopy(self.telemetry)

    async def read_messages(self, *, since_id: int | None = None, limit: int = 50) -> list[dict]:
        if since_id is None:
            return copy.deepcopy(self.messages[:limit])
        return copy.deepcopy([item for item in self.messages if int(item.get("id") or 0) > since_id][:limit])

    async def control_device(
        self,
        node_id: str,
        action: str,
        value=MISSING,
        *,
        dry_run: bool = False,
        verify: bool = True,
        max_age: float = 5.0,
    ) -> dict:
        record = {
            "node_id": node_id,
            "action": action,
            "value": None if value is MISSING else value,
            "dry_run": dry_run,
            "verify": verify,
        }
        self.actions.append(record)
        if dry_run:
            return {"dry_run": True, "request": record}
        node = self.telemetry.setdefault("nodes", {}).setdefault(node_id, {})
        if action == "start_job":
            node.update({"state": "PRINTING", "file_name": value, "paused": False, "homed": True})
        elif action == "home":
            node.update({"state": "IDLE", "homed": True})
        elif action in {"line_on", "line_off"}:
            for line in node.get("lines", []):
                if str(line.get("line")) == str(value):
                    line["state"] = "ON" if action == "line_on" else "OFF"
                    break
        elif action == "turn_on":
            node["enabled"] = True
        elif action == "set_mode":
            node["mode"] = value
        elif action == "set_primary_sensor":
            node["primary_co2_sensor"] = value
        elif action == "reset_interlock":
            interlocks = node.get("interlocks", {})
            if isinstance(interlocks, dict) and str(value) in interlocks:
                interlocks[str(value)]["state"] = "CLEAR"
        elif action == "reset_error":
            node.update({"error": False, "error_code": None})
        elif action == "set_fan":
            node["fan_state"] = value
        return {
            "success": True,
            "accepted": True,
            "node_id": node_id,
            "action": action,
            "diff": {"node_id": node_id, "action": action},
            "after": copy.deepcopy(node),
            "verify_error": None,
        }

    async def post_message(self, text: str, *, to: str | None = "operator", name: str | None = None) -> dict:
        self.sent_messages.append({"text": text, "to": to, "name": name})
        return {"success": True}


def operator_message(message_id: int, text: str, when: float | None = None) -> dict:
    return {
        "id": message_id,
        "time": time.time() if when is None else when,
        "sender": {"role": "operator", "team": None, "name": "Оператор"},
        "to": None,
        "text": text,
    }


class ConfigTests(unittest.TestCase):
    def test_from_env_prefers_key_file_and_redacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "team.key"
            key_file.write_text("team: abc123\n", encoding="utf-8")
            env = {
                "DOME_URL": "ws://127.0.0.1:8765/",
                "DOME_KEY": "",
                "DOME_KEY_FILE": str(key_file),
                "DEEPSEEK_API_KEY": "llm-secret",
            }
            with mock.patch.dict(os.environ, env, clear=True):
                config = AgentConfig.from_env(env_file=None)
            self.assertEqual(config.dome_url, "ws://127.0.0.1:8765")
            self.assertEqual(config.dome_key, "abc123")
            safe = config.safe_dict()
            self.assertNotIn("abc123", json.dumps(safe))
            self.assertNotIn("llm-secret", json.dumps(safe))
            self.assertNotIn("dome_key", safe)
            self.assertEqual(config.redact("abc123 and llm-secret"), "[REDACTED] and [REDACTED]")

    def test_invalid_url_and_missing_key(self) -> None:
        with self.assertRaises(ValueError):
            AgentConfig(dome_url="http://bad", dome_key="x")
        with self.assertRaises(ValueError):
            AgentConfig(dome_url="ws://host:1", dome_key="")

    def test_send_gap_enforces_five_per_second(self) -> None:
        with self.assertRaises(ValueError):
            AgentConfig(dome_url="ws://host:1", dome_key="x", min_send_gap_s=0.1)


class ChatTests(unittest.TestCase):
    def test_print_cnc_line_and_missing_info(self) -> None:
        print_task = parse_operator_task(operator_message(1, "Напечатать bracket_v2.gcode"))
        self.assertIsNotNone(print_task)
        assert print_task is not None
        self.assertEqual(print_task.items[0].kind, "print")
        self.assertEqual(print_task.items[0].file, "bracket_v2.gcode")

        cnc_task = parse_operator_task(operator_message(2, "Отфрезеровать pcb_power_board.nc"))
        self.assertIsNotNone(cnc_task)
        assert cnc_task is not None
        self.assertEqual(cnc_task.items[0].kind, "cnc")

        line_task = parse_operator_task(operator_message(3, "Выключи линию 8"))
        self.assertIsNotNone(line_task)
        assert line_task is not None
        self.assertEqual(line_task.items[0].kind, "line_off")
        self.assertEqual(line_task.items[0].line, 8)

        missing = parse_operator_task(operator_message(4, "Напечатай что-нибудь"))
        self.assertIsNotNone(missing)
        assert missing is not None
        self.assertTrue(missing.needs_info)

    def test_non_operator_and_chat_are_ignored(self) -> None:
        raw = operator_message(5, "Напечатать bracket_v2.gcode")
        raw["sender"] = {"role": "team", "team": 2, "name": "other"}
        self.assertIsNone(parse_operator_task(raw))
        chat = parse_operator_task(operator_message(6, "привет"))
        self.assertIsNotNone(chat)
        assert chat is not None
        self.assertEqual(chat.items, ())
        self.assertFalse(chat.actionable)

    def test_operator_message_for_another_team_is_not_routed(self) -> None:
        message = operator_message(7, "Напечатать bracket_v2.gcode")
        message["to"] = "team2"
        self.assertEqual(
            new_operator_messages([message], after_id=0, agent_name="team3-agent", team=3),
            [],
        )
        message["to"] = "team3"
        self.assertEqual(
            len(new_operator_messages([message], after_id=0, agent_name="team3-agent", team=3)),
            1,
        )

    def test_completion_guards_reject_webhook_lie(self) -> None:
        printer = {
            "state": "PRINTING",
            "real_state": "PRINTING",
            "webhook_state": "COMPLETED",
            "progress_pct": 4.0,
            "z_mm": 0.2,
            "paused": False,
        }
        check = check_print_completion(printer)
        self.assertFalse(check.ok)
        self.assertIn("webhook", check.reason)

        printer.update({"state": "COMPLETED", "real_state": "COMPLETED", "progress_pct": 100.0, "z_mm": 42.0})
        self.assertTrue(check_print_completion(printer).ok)

        cnc = {"state": "RUNNING", "progress_pct": 99.0}
        self.assertFalse(check_cnc_completion(cnc).ok)
        cnc.update({"state": "IDLE", "progress_pct": 100.0})
        self.assertTrue(check_cnc_completion(cnc).ok)


class RuleTests(unittest.TestCase):
    def test_baseline_has_no_actions(self) -> None:
        result = RuleEngine().decide(Snapshot(base_telemetry()))
        self.assertEqual(result.actions, [])
        self.assertIn("Нет требующих", result.situation)

    def test_false_co2_switches_sensor_before_reset(self) -> None:
        result = RuleEngine().decide(Snapshot(false_co2_telemetry()))
        actions = [(action.action, action.value) for action in result.actions]
        self.assertIn(("set_primary_sensor", "climate_sensor_02"), actions)
        self.assertIn(("set_data_trust", {"node": "climate_sensor_01", "param": "co2_ppm", "trust": "UNRELIABLE"}), actions)
        self.assertLess(actions.index(("set_primary_sensor", "climate_sensor_02")), actions.index(("reset_interlock", "IL_CO2")))

    def test_real_co2_ventilates_and_does_not_resume(self) -> None:
        result = RuleEngine().decide(Snapshot(real_co2_telemetry()))
        self.assertTrue(any(action.action == "set_mode" and action.value == "FRESH_AIR" for action in result.actions))
        self.assertFalse(any(action.action == "resume" for action in result.actions))

    def test_line_alarm_message_does_not_crash(self) -> None:
        data = base_telemetry()
        line3 = data["nodes"]["smart_panel_01"]["lines"][2]
        line3["alarm"] = True
        line3["alarm_code"] = "E_LINE"
        task = {
            "message_id": 1,
            "items": [
                {"kind": "print", "file": "bracket_v2.gcode", "status": "pending", "text": "print"}
            ],
        }
        result = RuleEngine().decide(Snapshot(data), task)
        self.assertTrue(any(message.dedupe_key == "line3:alarm" for message in result.messages))

    def test_unknown_hold_reason_is_not_resumed(self) -> None:
        data = base_telemetry()
        data["nodes"]["printer_3d_01"].update(
            {"state": "PAUSED", "paused": True, "hold_level": "WARNING", "hold_reason": None}
        )
        result = RuleEngine().decide(Snapshot(data))
        self.assertFalse(any(action.action == "resume" for action in result.actions))

    def test_cnc_is_blocked_when_fume_airflow_is_low(self) -> None:
        data = base_telemetry()
        data["nodes"]["cnc_01"].update({"state": "IDLE", "homed": True, "progress_pct": 0.0})
        data["nodes"]["fume_extraction_01"].update(
            {"enabled": True, "alarm": False, "motor_state": "OK", "airflow_pct": 0.0}
        )
        task = {
            "message_id": 2,
            "items": [
                {"kind": "cnc", "file": "pcb_power_board.nc", "status": "pending", "text": "cnc"}
            ],
        }
        result = RuleEngine().decide(Snapshot(data), task)
        self.assertEqual(result.actions, [])
        self.assertTrue(any("поток" in message.text for message in result.messages))

    def test_two_disagreeing_co2_sensors_choose_high_as_unconfirmed(self) -> None:
        data = base_telemetry()
        data["nodes"]["climate_sensor_01"]["co2_ppm"] = 760
        data["nodes"]["climate_sensor_02"]["co2_ppm"] = 3000
        data["nodes"]["climate_sensor_03"]["co2_ppm"] = None
        assessment = analyze_co2(data["nodes"])
        self.assertEqual(assessment.chosen_ppm, 3000)
        self.assertFalse(assessment.real_event)
        self.assertFalse(assessment.false_alarm)

    def test_missing_file_asks_operator(self) -> None:
        data = base_telemetry()
        task = {
            "message_id": 1,
            "items": [{"kind": "print", "file": "missing.gcode", "status": "pending", "text": "print"}],
        }
        result = RuleEngine().decide(Snapshot(data), task)
        self.assertTrue(any("missing.gcode" in message.text for message in result.messages))
        self.assertIn("needs_info", result.task["items"][0]["status"])


class LoggingStateTests(unittest.TestCase):
    def test_state_round_trip_and_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = StateStore(path)
            state.set_task({"message_id": 7, "items": []})
            state.mark_message(7)
            state.mark_action("x", success=True, now=100.0)
            state.save()
            loaded = StateStore(path)
            self.assertEqual(loaded.task["message_id"], 7)
            self.assertTrue(loaded.seen_message(7))
            self.assertFalse(loaded.action_allowed("x", 10.0, now=105.0))
            self.assertTrue(loaded.action_allowed("x", 10.0, now=111.0))

    def test_log_is_jsonl_and_redacts_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = AgentLog(tmp, secrets=("secret-value",))
            log.write("start", config={"dome_url": "ws://x", "password": "secret-value"})
            log.write("stop", reason="test", text="secret-value")
            log.close()
            records = read_logs(Path(tmp))
            self.assertEqual([record["kind"] for record in records], ["start", "stop"])
            self.assertEqual([record["seq"] for record in records], [1, 2])
            self.assertNotIn("secret-value", json.dumps(records))


class LlmTests(unittest.IsolatedAsyncioTestCase):
    async def test_parse_advice_and_mock_http_call(self) -> None:
        advice = parse_advice(
            '```json\n{"situation":"s","cause":"c","plan":["p"],"operator_message":"m",'
            '"risk":"high","needs_operator":true,"confidence":2}\n```'
        )
        self.assertIsNotNone(advice)
        assert advice is not None
        self.assertEqual(advice.confidence, 1.0)
        self.assertEqual(advice.risk, "high")

        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": json.dumps(advice.as_dict())}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(
                tmp,
                deepseek_api_key="llm-secret",
                deepseek_base_url="https://api.deepseek.test",
                llm_min_interval_s=0.0,
            )
            log = AgentLog(Path(tmp) / "logs", secrets=config.secrets())
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            advisor = DeepSeekAdvisor(config, log, client=client)
            result = await advisor.advise({"x": 1}, purpose="test")
            self.assertIsNotNone(result)
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].headers["authorization"], "Bearer llm-secret")
            log.close()
            records = read_logs(Path(tmp) / "logs")
            self.assertEqual(records[-1]["kind"], "llm_call")
            self.assertTrue(records[-1]["ok"])
            self.assertNotIn("llm-secret", json.dumps(records))
            await advisor.close()
            await client.aclose()

    async def test_llm_failure_returns_none_and_is_logged(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline", request=request)

        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, deepseek_api_key="x")
            log = AgentLog(Path(tmp) / "logs", secrets=config.secrets())
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            advisor = DeepSeekAdvisor(config, log, client=client)
            self.assertIsNone(await advisor.advise({"x": 1}))
            log.close()
            records = read_logs(Path(tmp) / "logs")
            self.assertFalse(records[-1]["ok"])
            await advisor.close()
            await client.aclose()

    async def test_hourly_budget_blocks_second_call(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"situation":"s","cause":"c"}'}}]},
            )

        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, deepseek_api_key="x", llm_min_interval_s=3600.0)
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            advisor = DeepSeekAdvisor(config, None, client=client, clock=lambda: 100.0)
            self.assertIsNotNone(await advisor.advise({"x": 1}))
            self.assertIsNone(await advisor.advise({"x": 2}))
            self.assertEqual(calls, 1)
            await advisor.close()
            await client.aclose()


class DelayedLineAPI(FakeAPI):
    def __init__(self, telemetry: dict) -> None:
        super().__init__(telemetry)
        self.telemetry_reads = 0

    async def get_telemetry(
        self,
        node_id: str | None = None,
        *,
        max_age: float = 5.0,
        next_frame: bool = False,
    ) -> dict:
        self.telemetry_reads += 1
        if node_id is not None:
            node = self.telemetry["nodes"][node_id]
            if self.telemetry_reads >= 2:
                for line in node.get("lines", []):
                    if int(line.get("line", 0)) == 8:
                        line["state"] = "OFF"
            return {"telemetry": copy.deepcopy(node)}
        return copy.deepcopy(self.telemetry)


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_dry_run_false_co2_never_sends_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=True, run_seconds=0.7)
            api = FakeAPI(false_co2_telemetry())
            agent = DomeAgent(config, api=api)
            await agent.run()
            self.assertTrue(api.actions)
            self.assertTrue(all(item["dry_run"] for item in api.actions))
            self.assertEqual(api.sent_messages, [])
            records = read_logs(config.log_dir)
            kinds = [record["kind"] for record in records]
            self.assertEqual(kinds[0], "start")
            self.assertEqual(kinds[-1], "stop")
            self.assertIn("action", kinds)
            self.assertIn("observation", kinds)
            self.assertNotIn("dome-secret-key", json.dumps(records))
            self.assertTrue(api.closed)

    async def test_operator_print_task_starts_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False, run_seconds=0.7)
            api = FakeAPI(
                base_telemetry(),
                messages=[operator_message(10, "Напечатать bracket_v2.gcode")],
            )
            agent = DomeAgent(config, api=api)
            await agent.run()
            self.assertTrue(
                any(item["action"] == "start_job" and item["value"] == "bracket_v2.gcode" for item in api.actions)
            )
            self.assertIsNotNone(agent.state.task)
            records = read_logs(config.log_dir)
            self.assertTrue(any(record["kind"] == "message" and record.get("direction") == "in" for record in records))
            verifications = [record for record in records if record["kind"] == "verification"]
            self.assertTrue(verifications)
            self.assertTrue(all(record["ok"] for record in verifications))

    async def test_line_verification_polls_until_state_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False)
            data = base_telemetry()
            line8 = next(
                line for line in data["nodes"]["smart_panel_01"]["lines"] if int(line["line"]) == 8
            )
            line8["state"] = "ON"
            api = DelayedLineAPI(data)
            agent = DomeAgent(config, api=api)
            self.addAsyncCleanup(agent._shutdown)
            action = ActionSpec(
                key="test-line-off",
                node_id="smart_panel_01",
                action="line_off",
                value=8,
                reason="test delayed telemetry",
            )
            result = {
                "success": True,
                "accepted": True,
                "diff": {},
                "after": copy.deepcopy(data["nodes"]["smart_panel_01"]),
                "verify_error": None,
            }
            ok, observed, error = await agent._verify_action(action, result)
            self.assertTrue(ok)
            self.assertIsNone(error)
            self.assertEqual(observed["state"], "OFF")
            self.assertGreaterEqual(api.telemetry_reads, 2)
            await api.close()

    async def test_run_seconds_stops_without_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=True, run_seconds=0.2)
            api = FakeAPI(base_telemetry())
            agent = DomeAgent(config, api=api)
            started = time.monotonic()
            await agent.run()
            self.assertLess(time.monotonic() - started, 5.0)
            records = read_logs(config.log_dir)
            self.assertEqual(records[-1]["kind"], "stop")
            self.assertEqual(records[-1]["reason"], "run_seconds")


if __name__ == "__main__":
    unittest.main()
