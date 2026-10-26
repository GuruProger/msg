"""Unit and integration tests for the autonomous Dome agent."""

from __future__ import annotations

import asyncio
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
from dome_agent.control import (
    DomeTool,
    ModelCommand,
    ModelDecision,
    is_auto_protective,
    parse_model_decision,
)
from dome_agent.gate import CommandGate, GateError
from dome_agent.llm import DeepSeekAdvisor, parse_advice
from dome_agent.logging import AgentLog
from dome_agent.rules import ActionSpec, RuleEngine
from dome_agent.state import StateStore
from dome_agent.verification import ExpectedEffect, expected_effect, meaningful_diff

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests/fixtures/telemetry.jsonl"


def base_telemetry() -> dict:
    with FIXTURE.open(encoding="utf-8") as handle:
        return copy.deepcopy(json.loads(handle.readline()))


def false_co2_telemetry() -> dict:
    data = base_telemetry()
    data["nodes"]["climate_sensor_01"]["co2_ppm"] = 3043
    # A false-alarm quorum requires current, same-zone peers.  The fixture's
    # reserve channels are delayed, so make this scenario explicitly current.
    for sensor in ("climate_sensor_02", "climate_sensor_03"):
        data["nodes"][sensor]["reading_delay_s"] = 0.0
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
        "control_enabled": True,
        "mode": "auto",
        "policy_file": REPO / "dome_policy.json",
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
        force: bool = False,
        verify: bool = True,
        max_age: float = 5.0,
        allow_without_before: bool = False,
    ) -> dict:
        record = {
            "node_id": node_id,
            "action": action,
            "value": None if value is MISSING else value,
            "dry_run": dry_run,
            "force": force,
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

    def test_delayed_co2_backups_do_not_refute_current_trip(self) -> None:
        data = base_telemetry()
        data["nodes"]["climate_sensor_01"]["co2_ppm"] = 3000
        data["nodes"]["dome_automation_01"]["interlocks"]["IL_CO2"]["state"] = "TRIPPED"
        assessment = analyze_co2(data["nodes"])
        self.assertEqual(assessment.chosen_id, "climate_sensor_01")
        self.assertEqual(assessment.chosen_ppm, 3000)
        self.assertFalse(assessment.false_alarm)
        self.assertIn("climate_sensor_02", assessment.delayed_ids)
        self.assertIn("climate_sensor_03", assessment.delayed_ids)

        result = RuleEngine().decide(Snapshot(data))
        actions = [(action.action, action.value) for action in result.actions]
        self.assertNotIn(("set_primary_sensor", "climate_sensor_02"), actions)
        self.assertFalse(any(action == "set_data_trust" for action, _ in actions))
        self.assertFalse(any(action == "reset_interlock" for action, _ in actions))
        self.assertTrue(
            any(action in {"turn_on", "set_mode"} for action, _ in actions)
        )
        self.assertFalse(any(action == "resume" for action, _ in actions))

    def test_other_zone_safe_sensor_does_not_refute_current_trip(self) -> None:
        data = base_telemetry()
        data["nodes"]["climate_sensor_01"]["co2_ppm"] = 3000
        for sensor in ("climate_sensor_02", "climate_sensor_03"):
            data["nodes"][sensor]["reading_delay_s"] = 0.0
            data["nodes"][sensor]["zone"] = "STORAGE"
        data["nodes"]["dome_automation_01"]["interlocks"]["IL_CO2"]["state"] = "TRIPPED"
        result = RuleEngine().decide(Snapshot(data))
        self.assertFalse(any(action.action == "reset_interlock" for action in result.actions))
        self.assertFalse(any(action.action == "resume" for action in result.actions))

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

    def test_interlock_list_is_normalized_and_blocks_resume(self) -> None:
        data = base_telemetry()
        automation = data["nodes"]["dome_automation_01"]
        automation["interlocks"] = [
            {"id": name, **entry}
            for name, entry in automation["interlocks"].items()
        ]
        for entry in automation["interlocks"]:
            if entry["id"] == "IL_CO2":
                entry["state"] = "TRIPPED"
        data["nodes"]["printer_3d_01"].update(
            {"state": "PAUSED", "paused": True, "hold_level": "WARNING", "hold_reason": "IL_CO2"}
        )
        result = RuleEngine().decide(Snapshot(data))
        self.assertFalse(any(action.action == "resume" for action in result.actions))

    def test_malformed_interlocks_fail_closed(self) -> None:
        data = base_telemetry()
        data["nodes"]["dome_automation_01"]["interlocks"] = "broken"
        data["nodes"]["printer_3d_01"].update(
            {"state": "PAUSED", "paused": True, "hold_level": "WARNING", "hold_reason": "IL_CO2"}
        )
        result = RuleEngine().decide(Snapshot(data))
        self.assertFalse(any(action.action == "resume" for action in result.actions))
        self.assertFalse(any(action.action == "reset_interlock" for action in result.actions))

    def test_missing_co2_does_not_reset_interlock(self) -> None:
        data = base_telemetry()
        for sensor in ("climate_sensor_01", "climate_sensor_02", "climate_sensor_03"):
            data["nodes"][sensor]["co2_ppm"] = None
        data["nodes"]["dome_automation_01"]["interlocks"]["IL_CO2"]["state"] = "TRIPPED"
        result = RuleEngine().decide(Snapshot(data))
        self.assertFalse(any(action.action == "reset_interlock" for action in result.actions))

    def test_missing_voltage_does_not_resume(self) -> None:
        data = base_telemetry()
        data["nodes"]["smart_panel_01"]["bus_voltage_v"] = None
        data["nodes"]["printer_3d_01"].update(
            {"state": "PAUSED", "paused": True, "hold_level": "WARNING", "hold_reason": "UNDERVOLTAGE"}
        )
        result = RuleEngine().decide(Snapshot(data))
        self.assertFalse(any(action.action == "resume" for action in result.actions))

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
        data["nodes"]["climate_sensor_02"]["reading_delay_s"] = 0.0
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

    def test_failed_action_cooldown_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = StateStore(path)
            state.mark_action(
                "line8-off",
                success=False,
                now=100.0,
                failure_cooldown_s=30.0,
            )
            self.assertTrue(state.action_failed_recently("line8-off", now=101.0))
            self.assertEqual(state.failure_remaining("line8-off", now=101.0), 29.0)
            self.assertFalse(state.action_failed_recently("line8-off", now=131.0))
            loaded = StateStore(path)
            self.assertTrue(loaded.action_failed_recently("line8-off", now=101.0))
            loaded.mark_action("line8-off", success=True, now=102.0)
            self.assertFalse(loaded.action_failed_recently("line8-off", now=103.0))

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

    async def test_decide_bad_json_returns_none_and_is_logged(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "not-json"}}]},
            )

        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, deepseek_api_key="x", llm_min_interval_s=0.0)
            log = AgentLog(Path(tmp) / "logs", secrets=config.secrets())
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            advisor = DeepSeekAdvisor(config, log, client=client)
            self.assertIsNone(await advisor.decide({"x": 1}, mode="normal"))
            log.close()
            records = read_logs(Path(tmp) / "logs")
            self.assertEqual(records[-1]["kind"], "llm_call")
            self.assertFalse(records[-1]["ok"])
            self.assertIn("valid control JSON", records[-1]["error"])
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


class FakeAdvisor:
    """Small deterministic stand-in for DeepSeekAdvisor in mode tests."""

    def __init__(self, decisions: list[ModelDecision | None]) -> None:
        self.decisions = list(decisions)
        self.calls: list[dict] = []
        self.available = True

    async def decide(self, context, *, mode="normal", force=False, purpose="control"):
        self.calls.append(
            {"mode": mode, "force": force, "purpose": purpose, "context": context}
        )
        return self.decisions.pop(0) if self.decisions else None

    async def close(self) -> None:
        return None


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


class GatedVerificationAPI(FakeAPI):
    """Block the next-frame read so tests can inspect the fast loop."""

    def __init__(self, telemetry: dict) -> None:
        super().__init__(telemetry)
        self.verify_started = asyncio.Event()
        self.verify_release = asyncio.Event()

    async def get_telemetry(
        self,
        node_id: str | None = None,
        *,
        max_age: float = 5.0,
        next_frame: bool = False,
    ) -> dict:
        if node_id is not None and next_frame:
            self.verify_started.set()
            await self.verify_release.wait()
        return await super().get_telemetry(
            node_id,
            max_age=max_age,
            next_frame=next_frame,
        )


class BeforeAwareAPI(FakeAPI):
    """Refuse non-emergency controls when before-telemetry is unavailable."""

    async def control_device(
        self,
        node_id: str,
        action: str,
        value=MISSING,
        *,
        dry_run: bool = False,
        force: bool = False,
        verify: bool = True,
        max_age: float = 5.0,
        allow_without_before: bool = False,
    ) -> dict:
        if not allow_without_before:
            return {
                "success": False,
                "accepted": False,
                "blocked": True,
                "unknown": False,
                "sent": False,
                "error": "fresh before-telemetry unavailable: test",
                "before_error": "test",
                "after": None,
                "diff": None,
                "verify_error": None,
            }
        return await super().control_device(
            node_id,
            action,
            value,
            dry_run=dry_run,
            force=force,
            verify=verify,
            max_age=max_age,
            allow_without_before=allow_without_before,
        )


class NoEffectAPI(FakeAPI):
    """Accept a command but leave telemetry unchanged."""

    async def control_device(
        self,
        node_id: str,
        action: str,
        value=MISSING,
        *,
        dry_run: bool = False,
        force: bool = False,
        verify: bool = True,
        max_age: float = 5.0,
        allow_without_before: bool = False,
    ) -> dict:
        record = {
            "node_id": node_id,
            "action": action,
            "value": None if value is MISSING else value,
            "dry_run": dry_run,
            "force": force,
            "verify": verify,
        }
        self.actions.append(record)
        if dry_run:
            return {"dry_run": True, "request": record}
        node = self.telemetry.setdefault("nodes", {}).setdefault(node_id, {})
        return {
            "success": True,
            "accepted": True,
            "node_id": node_id,
            "action": action,
            "diff": {},
            "after": copy.deepcopy(node),
            "verify_error": None,
        }


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

    async def test_slow_verification_does_not_block_safety_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False, mode="auto")
            data = false_co2_telemetry()
            line8 = next(
                line
                for line in data["nodes"]["smart_panel_01"]["lines"]
                if int(line["line"]) == 8
            )
            line8["state"] = "ON"
            line8["protection_state"] = "TRIPPED"
            api = GatedVerificationAPI(data)
            agent = DomeAgent(config, api=api)
            self.addAsyncCleanup(agent._shutdown)

            slow = ActionSpec(
                key="slow-verification",
                node_id="smart_panel_01",
                action="reset_protection",
                value=8,
                reason="hold the verifier",
                expect=expected_effect(
                    "smart_panel_01",
                    "reset_protection",
                    8,
                    timeout_s=1.5,
                ),
            )
            agent._enqueue_action(slow, None)
            await asyncio.wait_for(api.verify_started.wait(), timeout=1.0)

            started = time.monotonic()
            await asyncio.wait_for(agent._cycle(), timeout=1.0)
            self.assertLess(time.monotonic() - started, 0.8)

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if any(
                    item["action"] == "set_primary_sensor"
                    and item["value"] == "climate_sensor_02"
                    for item in api.actions
                ) and any(
                    item["action"] == "reset_interlock"
                    and item["value"] == "IL_CO2"
                    for item in api.actions
                ):
                    break
                await asyncio.sleep(0.01)
            self.assertFalse(api.verify_release.is_set())
            self.assertTrue(
                any(
                    item["action"] == "set_primary_sensor"
                    and item["value"] == "climate_sensor_02"
                    for item in api.actions
                )
            )
            self.assertTrue(
                any(
                    item["action"] == "reset_interlock"
                    and item["value"] == "IL_CO2"
                    for item in api.actions
                )
            )

            api.verify_release.set()
            await asyncio.wait_for(
                asyncio.gather(
                    *(queue.join() for queue in agent._action_queues.values())
                ),
                timeout=3.0,
            )

    async def test_before_telemetry_policy_blocks_normal_but_not_emergency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False)
            data = base_telemetry()
            line8 = next(
                line
                for line in data["nodes"]["smart_panel_01"]["lines"]
                if int(line["line"]) == 8
            )
            line8["state"] = "ON"
            api = BeforeAwareAPI(data)
            agent = DomeAgent(config, api=api)
            self.addAsyncCleanup(agent._shutdown)

            normal = ActionSpec(
                key="normal-before-required",
                node_id="dome_automation_01",
                action="set_primary_sensor",
                value="climate_sensor_02",
                reason="requires before evidence",
            )
            await agent._execute_action(normal, None)
            self.assertEqual(api.actions, [])
            records = read_logs(config.log_dir)
            blocked = [
                record for record in records if record["kind"] == "action_blocked"
            ]
            self.assertTrue(blocked)
            self.assertEqual(blocked[-1]["source"], "before_telemetry")

            emergency = ActionSpec(
                key="emergency-before-allowed",
                node_id="smart_panel_01",
                action="line_off",
                value=8,
                reason="protective action",
            )
            await agent._execute_action(emergency, None)
            self.assertTrue(
                any(
                    item["action"] == "line_off" and item["value"] == 8
                    for item in api.actions
                )
            )

    async def test_no_effect_blocks_retry_until_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False)
            data = base_telemetry()
            line8 = next(
                line for line in data["nodes"]["smart_panel_01"]["lines"] if int(line["line"]) == 8
            )
            line8["state"] = "ON"
            api = NoEffectAPI(data)
            agent = DomeAgent(config, api=api)
            self.addAsyncCleanup(agent._shutdown)
            action = ActionSpec(
                key="test-line-off-no-effect",
                node_id="smart_panel_01",
                action="line_off",
                value=8,
                reason="test no effect",
                expect=expected_effect(
                    "smart_panel_01",
                    "line_off",
                    8,
                    timeout_s=0.05,
                ),
            )
            await agent._execute_action(action, None)
            self.assertTrue(agent.state.action_failed_recently(action.key))
            sent = len(api.actions)
            await agent._execute_action(action, None)
            self.assertEqual(len(api.actions), sent)
            records = read_logs(config.log_dir)
            self.assertTrue(
                any(
                    record["kind"] == "verification"
                    and record["ok"] is False
                    and record["no_effect"] is True
                    for record in records
                )
            )
            self.assertTrue(
                any(
                    record["kind"] == "action_blocked"
                    and "retry cooldown" in record["reason"]
                    for record in records
                )
            )

    async def test_duplicate_pending_action_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False)
            api = FakeAPI(base_telemetry())
            agent = DomeAgent(config, api=api)
            self.addAsyncCleanup(agent._shutdown)
            action = ActionSpec(
                key="test-line-off-pending",
                node_id="smart_panel_01",
                action="line_off",
                value=8,
                reason="test duplicate",
            )
            agent._pending_action_keys.add(agent._action_lock_key(action))
            await agent._execute_action(action, None)
            self.assertEqual(api.actions, [])
            records = read_logs(config.log_dir)
            self.assertTrue(
                any(
                    record["kind"] == "action_blocked"
                    and "pending verification" in record["reason"]
                    for record in records
                )
            )

    async def test_unknown_action_uses_diff_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False)
            api = FakeAPI(base_telemetry())
            agent = DomeAgent(config, api=api)
            self.addAsyncCleanup(agent._shutdown)
            action = ActionSpec(
                key="test-unknown-action",
                node_id="smart_panel_01",
                action="mystery_action",
                reason="test unknown",
            )
            result = {
                "success": True,
                "diff": {"changed": True},
                "after": copy.deepcopy(base_telemetry()["nodes"]["smart_panel_01"]),
                "verify_error": None,
            }
            ok, observed, error = await agent._verify_action(action, result)
            self.assertTrue(ok)
            self.assertEqual(observed, {"changed": True})
            self.assertIsNone(error)

    async def test_unknown_action_metadata_only_diff_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False)
            api = FakeAPI(base_telemetry())
            agent = DomeAgent(config, api=api)
            self.addAsyncCleanup(agent._shutdown)
            action = ActionSpec(
                key="test-unknown-metadata",
                node_id="smart_panel_01",
                action="mystery_action",
                reason="test unknown metadata",
            )
            diff = {"node_id": "smart_panel_01", "action": "mystery_action"}
            result = {
                "success": True,
                "diff": diff,
                "after": copy.deepcopy(base_telemetry()["nodes"]["smart_panel_01"]),
                "verify_error": None,
            }
            ok, observed, error = await agent._verify_action(action, result)
            self.assertFalse(ok)
            self.assertEqual(observed, diff)
            self.assertIsNone(error)

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


class VerificationTests(unittest.TestCase):
    def test_fan_effect_requires_physical_state(self) -> None:
        effect = expected_effect("solar_inverter_01", "set_fan", "ON")
        self.assertFalse(
            effect.check({"fan_mode": "ON", "fan_state": "OFF", "fan_current_a": 0.0})
        )
        self.assertFalse(
            effect.check({"fan_state": "ON", "fan_current_a": 0.0})
        )
        self.assertTrue(
            effect.check({"fan_state": "ON", "fan_current_a": 0.25})
        )
        off = expected_effect("solar_inverter_01", "set_fan", "OFF")
        self.assertTrue(
            off.check({"fan_state": "OFF", "fan_current_a": 0.0})
        )
        self.assertFalse(
            off.check({"fan_state": "OFF"})
        )

    def test_reset_interlock_effect_accepts_list_shape(self) -> None:
        effect = expected_effect(
            "dome_automation_01", "reset_interlock", "IL_CO2"
        )
        self.assertTrue(
            effect.check(
                {
                    "interlocks": [
                        {"id": "IL_CO2", "state": "ARMED"},
                        {"id": "IL_SMOKE", "state": "TRIPPED"},
                    ]
                }
            )
        )
        self.assertFalse(
            effect.check({"interlocks": [{"id": "IL_CO2", "state": "TRIPPED"}]})
        )

    def test_metadata_only_diff_is_not_a_physical_effect(self) -> None:
        self.assertFalse(meaningful_diff({"node_id": "smart_panel_01", "action": "line_off"}))
        self.assertTrue(meaningful_diff({"line": 8, "state": "OFF"}))

    def test_expected_effect_line_and_unknown_action(self) -> None:
        effect = expected_effect("smart_panel_01", "line_off", 8)
        self.assertTrue(effect.check({"lines": [{"line": 8, "state": "OFF"}]}))
        self.assertFalse(effect.check({"lines": [{"line": 8, "state": "ON"}]}))
        self.assertEqual(
            effect.observe({"lines": [{"line": 8, "state": "OFF"}]})["state"],
            "OFF",
        )
        unknown = expected_effect("smart_panel_01", "mystery", None)
        self.assertIsNone(unknown.check({"state": "IDLE"}))


class GateTests(unittest.TestCase):
    def test_read_only_default_denies(self) -> None:
        gate = CommandGate(REPO / "dome_policy.json", control_enabled=False)
        action = ActionSpec(
            key="line8-on",
            node_id="smart_panel_01",
            action="line_on",
            value=8,
            reason="test",
        )
        decision = gate.check(action)
        self.assertFalse(decision.allowed)
        self.assertIn("control disabled", decision.reason)

    def test_policy_allows_known_action_and_rejects_bad_value(self) -> None:
        gate = CommandGate(REPO / "dome_policy.json", control_enabled=True)
        allowed = ActionSpec(
            key="line8-on",
            node_id="smart_panel_01",
            action="line_on",
            value=8,
            reason="test",
        )
        bad_value = ActionSpec(
            key="line9-on",
            node_id="smart_panel_01",
            action="line_on",
            value=9,
            reason="test",
        )
        self.assertTrue(gate.check(allowed).allowed)
        self.assertFalse(gate.check(bad_value).allowed)

    def test_policy_rejects_dangerous_action_even_if_node_is_listed(self) -> None:
        gate = CommandGate(REPO / "dome_policy.json", control_enabled=True)
        action = ActionSpec(
            key="unlock",
            node_id="access_control_01",
            action="unlock_door",
            reason="test",
        )
        decision = gate.check(action)
        self.assertFalse(decision.allowed)
        self.assertIn("denied", decision.reason)

    def test_policy_rejects_unknown_node(self) -> None:
        gate = CommandGate(REPO / "dome_policy.json", control_enabled=True)
        action = ActionSpec(
            key="unknown",
            node_id="unknown_node",
            action="turn_on",
            reason="test",
        )
        self.assertFalse(gate.check(action).allowed)

    def test_missing_policy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GateError):
                CommandGate(Path(tmp) / "missing.json", control_enabled=True)

    def test_yolo_bypasses_policy_and_marks_unsafe(self) -> None:
        gate = CommandGate(
            Path("/nonexistent/policy.json"),
            control_enabled=False,
            yolo=True,
        )
        action = ActionSpec(
            key="unlock",
            node_id="access_control_01",
            action="unlock_door",
            reason="test",
        )
        decision = gate.check(action)
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.unsafe)
        self.assertIn("YOLO", decision.reason)

    def test_yolo_config_implies_control_enabled(self) -> None:
        config = AgentConfig(
            dome_url="wss://example.test",
            dome_key="x",
            mode="yolo",
            policy_file=REPO / "dome_policy.json",
        )
        self.assertTrue(config.control_enabled)
        self.assertTrue(config.yolo)

    def test_env_flags(self) -> None:
        env = {
            "DOME_URL": "wss://example.test",
            "DOME_KEY": "team-key",
            "AGENT_ALLOW_CONTROL": "1",
            "AGENT_YOLO": "0",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = AgentConfig.from_env(env_file=None)
        self.assertTrue(config.control_enabled)
        self.assertFalse(config.yolo)

    def test_yolo_bypasses_rule_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(Path(tmp) / "state.json")
            state.mark_action("key", success=False, now=100.0)
            normal = RuleEngine(state, clock=lambda: 101.0)
            unsafe = RuleEngine(state, clock=lambda: 101.0, yolo=True)
            self.assertFalse(normal._allowed("key", 9999))
            self.assertTrue(unsafe._allowed("key", 9999))


class ControlTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_model_decision_drops_hard_denied(self) -> None:
        decision = parse_model_decision(
            json.dumps(
                {
                    "analysis": "test",
                    "operator_message": "",
                    "commands": [
                        {
                            "tool": "control",
                            "node_id": "access_control_01",
                            "action": "unlock_door",
                        },
                        {
                            "tool": "control",
                            "node_id": "smart_panel_01",
                            "action": "line_on",
                            "value": 8,
                        },
                    ],
                }
            )
        )
        assert decision is not None
        self.assertEqual(len(decision.commands), 1)
        self.assertEqual(decision.commands[0].action, "line_on")
        self.assertTrue(any("hard-denied" in warning for warning in decision.warnings))

    def test_auto_protective_rules(self) -> None:
        self.assertTrue(
            is_auto_protective(
                ModelCommand(tool="control", node_id="anything", action="turn_off")
            )
        )
        self.assertTrue(
            is_auto_protective(
                ModelCommand(
                    tool="control",
                    node_id="solar_inverter_01",
                    action="set_fan",
                    value="ON",
                )
            )
        )
        self.assertFalse(
            is_auto_protective(
                ModelCommand(
                    tool="control",
                    node_id="solar_inverter_01",
                    action="set_fan",
                    value="OFF",
                )
            )
        )
        self.assertFalse(
            is_auto_protective(
                ModelCommand(
                    tool="control",
                    node_id="smart_panel_01",
                    action="line_on",
                    value=8,
                )
            )
        )

    async def test_dome_tool_hard_deny_never_calls_api(self) -> None:
        api = FakeAPI(base_telemetry())
        tool = DomeTool(api)
        result = await tool.execute(
            ModelCommand(
                tool="control",
                node_id="access_control_01",
                action="unlock_door",
            ),
            force=True,
        )
        self.assertFalse(result["ok"])
        self.assertTrue(result["blocked"])
        self.assertFalse(result["forceable"])
        self.assertEqual(api.actions, [])


class GateAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_read_only_default_blocks_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False, control_enabled=False)
            api = FakeAPI(base_telemetry())
            agent = DomeAgent(config, api=api)
            self.addAsyncCleanup(agent._shutdown)
            action = ActionSpec(
                key="line8-on",
                node_id="smart_panel_01",
                action="line_on",
                value=8,
                reason="test gate",
            )
            await agent._execute_action(action, None)
            await agent._execute_action(action, None)
            self.assertEqual(api.actions, [])
            records = read_logs(config.log_dir)
            blocked = [record for record in records if record["kind"] == "action_blocked"]
            self.assertEqual(len(blocked), 1)
            self.assertIn("control disabled", blocked[0]["reason"])

    async def test_yolo_executes_llm_command_without_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(
                tmp,
                dry_run=False,
                mode="yolo",
                control_enabled=False,
            )
            api = FakeAPI(base_telemetry())
            advisor = FakeAdvisor(
                [
                    ModelDecision(
                        analysis="Оператор просит начать печать",
                        operator_message="Запускаю печать.",
                        commands=(
                            ModelCommand(
                                tool="control",
                                node_id="printer_3d_01",
                                action="start_job",
                                value="bracket_v2.gcode",
                                reason="operator request",
                            ),
                        ),
                        risk="medium",
                    )
                ]
            )
            agent = DomeAgent(config, api=api, advisor=advisor)
            self.addAsyncCleanup(agent._shutdown)
            await agent._cycle()
            assert agent._control_task is not None
            await agent._control_task
            self.assertTrue(
                any(
                    item["action"] == "start_job"
                    and item["value"] == "bracket_v2.gcode"
                    and item["verify"] is False
                    for item in api.actions
                )
            )
            records = read_logs(config.log_dir)
            actions = [record for record in records if record["kind"] == "action"]
            self.assertTrue(actions)
            self.assertTrue(all(record.get("unsafe") is True for record in actions))
            self.assertTrue(all(record.get("source") == "yolo" for record in actions))
            self.assertEqual(
                [record for record in records if record["kind"] == "verification"], []
            )

    async def test_yolo_hard_deny_is_not_sent_even_with_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False, mode="yolo")
            api = FakeAPI(base_telemetry())
            advisor = FakeAdvisor(
                [
                    ModelDecision(
                        analysis="trying a forbidden action",
                        operator_message="",
                        commands=(
                            ModelCommand(
                                tool="control",
                                node_id="access_control_01",
                                action="unlock_door",
                                reason="test",
                            ),
                        ),
                    )
                ]
            )
            agent = DomeAgent(config, api=api, advisor=advisor)
            self.addAsyncCleanup(agent._shutdown)
            await agent._cycle()
            assert agent._control_task is not None
            await agent._control_task
            self.assertEqual(api.actions, [])
            records = read_logs(config.log_dir)
            blocked = [record for record in records if record["kind"] == "action_blocked"]
            self.assertTrue(blocked)
            self.assertIn("hard-denied", blocked[0]["reason"])

    async def test_model_command_uses_explicit_effect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False, mode="normal")
            api = FakeAPI(base_telemetry())
            agent = DomeAgent(config, api=api)
            self.addAsyncCleanup(agent._shutdown)
            effect = ExpectedEffect(
                description="custom effect",
                check=lambda node: True,
                observe=lambda node: {"custom": True},
            )
            command = ModelCommand(
                tool="control",
                node_id="smart_panel_01",
                action="line_off",
                value=8,
                reason="test",
                expect=effect,
            )
            result = await agent._execute_model_command(command, force=False, source="test")
            self.assertTrue(result["ok"])
            self.assertTrue(result["result"]["verified"])
            self.assertEqual(result["result"]["observed"], {"custom": True})
            verifications = [
                record
                for record in read_logs(config.log_dir)
                if record["kind"] == "verification"
            ]
            self.assertEqual(verifications[-1]["expected"], "custom effect")
            self.assertTrue(verifications[-1]["ok"])

    async def test_model_action_failure_cooldown_blocks_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False, mode="normal")
            data = base_telemetry()
            line8 = next(
                line
                for line in data["nodes"]["smart_panel_01"]["lines"]
                if int(line["line"]) == 8
            )
            line8["state"] = "ON"
            api = NoEffectAPI(data)
            agent = DomeAgent(config, api=api)
            self.addAsyncCleanup(agent._shutdown)
            command = ModelCommand(
                tool="control",
                node_id="smart_panel_01",
                action="line_off",
                value=8,
                reason="test no effect",
            )
            first = await agent._execute_model_command(command, force=False, source="test")
            self.assertFalse(first["ok"])
            self.assertIn("did not confirm", first["error"])
            self.assertEqual(len(api.actions), 1)

            second = await agent._execute_model_command(command, force=False, source="test")
            self.assertTrue(second["blocked"])
            self.assertIn("retry cooldown", second["error"])
            self.assertEqual(len(api.actions), 1)
            blocked = [
                record
                for record in read_logs(config.log_dir)
                if record["kind"] == "action_blocked"
            ]
            self.assertTrue(any("retry cooldown" in record["reason"] for record in blocked))

    async def test_rule_and_model_paths_share_pending_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False, mode="normal")
            api = FakeAPI(base_telemetry())
            agent = DomeAgent(config, api=api)
            self.addAsyncCleanup(agent._shutdown)
            action = ActionSpec(
                key="rule-line-off",
                node_id="smart_panel_01",
                action="line_off",
                value=8,
                reason="test",
            )
            command = ModelCommand(
                tool="control",
                node_id="smart_panel_01",
                action="line_off",
                value=8,
                reason="test",
            )
            agent._pending_action_keys.add(agent._action_lock_key(action))
            result = await agent._execute_model_command(command, force=False, source="test")
            self.assertTrue(result["blocked"])
            self.assertIn("pending verification", result["error"])
            self.assertEqual(api.actions, [])

    async def test_normal_proposes_then_confirms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False, mode="normal")
            api = FakeAPI(base_telemetry())
            advisor = FakeAdvisor(
                [
                    ModelDecision(
                        analysis="line 8 can be enabled",
                        operator_message="Предлагаю включить линию 8.",
                        commands=(
                            ModelCommand(
                                tool="control",
                                node_id="smart_panel_01",
                                action="line_on",
                                value=8,
                                reason="operator request",
                            ),
                        ),
                    )
                ]
            )
            agent = DomeAgent(config, api=api, advisor=advisor)
            self.addAsyncCleanup(agent._shutdown)
            await agent._cycle()
            assert agent._control_task is not None
            await agent._control_task
            self.assertEqual(api.actions, [])
            self.assertIsNotNone(agent._pending)
            self.assertTrue(any("Подтвердить" in item["text"] for item in api.sent_messages))

            agent._incoming.append(ChatMessage.from_raw(operator_message(20, "да")))
            await agent._handle_pending()
            self.assertIsNone(agent._pending)
            self.assertTrue(
                any(
                    item["action"] == "line_on" and item["value"] == 8
                    for item in api.actions
                )
            )
            records = read_logs(config.log_dir)
            confirmed = [
                record
                for record in records
                if record["kind"] == "action" and record.get("source") == "confirmed"
            ]
            self.assertTrue(confirmed)

    async def test_normal_does_not_replace_pending_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False, mode="normal")
            api = FakeAPI(base_telemetry())
            advisor = FakeAdvisor(
                [
                    ModelDecision(
                        analysis="first",
                        operator_message="first proposal",
                        commands=(
                            ModelCommand(
                                tool="control",
                                node_id="smart_panel_01",
                                action="line_on",
                                value=8,
                            ),
                        ),
                    ),
                    ModelDecision(
                        analysis="second",
                        operator_message="second proposal",
                        commands=(
                            ModelCommand(
                                tool="control",
                                node_id="smart_panel_01",
                                action="line_off",
                                value=7,
                            ),
                        ),
                    ),
                ]
            )
            agent = DomeAgent(config, api=api, advisor=advisor)
            self.addAsyncCleanup(agent._shutdown)
            await agent._cycle()
            assert agent._control_task is not None
            await agent._control_task
            assert agent._pending is not None
            original = agent._pending.signature

            agent._last_control_at = 0.0
            await agent._cycle()
            assert agent._control_task is not None
            await agent._control_task
            assert agent._pending is not None
            self.assertEqual(agent._pending.signature, original)
            self.assertTrue(any("ожидающее" in item["text"] for item in api.sent_messages))

    async def test_normal_blocked_proposal_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp, dry_run=False, mode="normal")
            api = FakeAPI(base_telemetry())
            advisor = FakeAdvisor(
                [
                    ModelDecision(
                        analysis="line 9 is outside policy",
                        operator_message="Предлагаю включить линию 9.",
                        commands=(
                            ModelCommand(
                                tool="control",
                                node_id="smart_panel_01",
                                action="line_on",
                                value=9,
                                reason="test force",
                            ),
                        ),
                    )
                ]
            )
            agent = DomeAgent(config, api=api, advisor=advisor)
            self.addAsyncCleanup(agent._shutdown)
            await agent._cycle()
            assert agent._control_task is not None
            await agent._control_task
            agent._incoming.append(ChatMessage.from_raw(operator_message(21, "да")))
            await agent._handle_pending()
            self.assertEqual(api.actions, [])
            self.assertIsNotNone(agent._pending)
            assert agent._pending is not None
            self.assertTrue(agent._pending.awaiting_force)
            self.assertTrue(any("принудительно" in item["text"] for item in api.sent_messages))

            agent._incoming.append(
                ChatMessage.from_raw(operator_message(22, "да, принудительно"))
            )
            await agent._handle_pending()
            self.assertIsNone(agent._pending)
            self.assertTrue(
                any(
                    item["action"] == "line_on" and item["value"] == 9
                    for item in api.actions
                )
            )
            records = read_logs(config.log_dir)
            forced = [
                record
                for record in records
                if record["kind"] == "action"
                and record.get("source") == "confirmed_force"
            ]
            self.assertTrue(forced)
            self.assertTrue(forced[0].get("force") is True)


if __name__ == "__main__":
    unittest.main()
