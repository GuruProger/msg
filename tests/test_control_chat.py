"""Deterministic panel-chat controls: parsing, confirmation and execution."""
from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dome_web import control_chat
from dome_web.conversation import Conversation
from dome_web.service import DashboardService
from dome_web.control_chat import LineCommand


def panel_lines():
    return [
        {"line": 1, "name": "Производственный отдел (рабочее место, вытяжка FabLab)", "state": "ON", "protection_state": "ARMED"},
        {"line": 2, "name": "Теплица ФабЛаб", "state": "ON", "protection_state": "ARMED"},
        {"line": 3, "name": "3D-принтер", "state": "ON", "protection_state": "ARMED"},
        {"line": 4, "name": "ЧПУ-фрезер", "state": "ON", "protection_state": "ARMED"},
        {"line": 5, "name": "Теплица Химлаб", "state": "ON", "protection_state": "ARMED"},
        {"line": 6, "name": "Обычное освещение", "state": "ON", "protection_state": "ARMED"},
        {"line": 7, "name": "Аварийное освещение", "state": "OFF", "protection_state": "ARMED", "alarm": False, "alarm_code": None},
        {"line": 8, "name": "Резервная линия / фантомное потребление", "state": "ON", "protection_state": "ARMED"},
    ]


class ParserTests(unittest.TestCase):
    def test_explicit_command_is_recognised(self):
        self.assertTrue(control_chat.looks_like_line_command("включи аварийное освещение"))
        self.assertTrue(control_chat.looks_like_line_command("выключи линию 3"))
        self.assertTrue(control_chat.is_confirmation("да"))
        self.assertTrue(control_chat.is_confirmation("Да."))
        self.assertTrue(control_chat.is_cancellation("нет"))

    def test_questions_are_not_commands(self):
        for text in (
            "как включить аварийное освещение?",
            "почему не горит свет?",
            "можно включить линию 7?",
            "что такое аварийное освещение",
        ):
            self.assertFalse(control_chat.looks_like_line_command(text), text)

    def test_name_resolves_to_live_line(self):
        command = control_chat.resolve_line_command("включи аварийное освещение", panel_lines())
        self.assertIsNotNone(command)
        self.assertEqual(command.line, 7)
        self.assertEqual(command.action, "line_on")

    def test_informal_name_resolves(self):
        command = control_chat.resolve_line_command("включи аварийку", panel_lines())
        self.assertEqual(command.line, 7)

    def test_explicit_number_resolves(self):
        command = control_chat.resolve_line_command("выключи линию 3", panel_lines())
        self.assertEqual(command.line, 3)
        self.assertEqual(command.action, "line_off")

    def test_missing_line_is_reported_not_guessed(self):
        command = control_chat.resolve_line_command("включи линию 9", panel_lines())
        self.assertIsNone(command.line)
        self.assertIn("Линии 9 нет", command.reason)

    def test_ambiguous_name_is_never_guessed(self):
        command = control_chat.resolve_line_command("включи освещение", panel_lines())
        self.assertIsNone(command.line)
        self.assertIn("уточните", command.reason.lower())


class FakeService:
    def __init__(self, lines=None):
        self.lines = copy.deepcopy(panel_lines() if lines is None else lines)
        self.executed = []
        self.events = []
        self.logged = []
        self.telemetry = {"nodes": {"smart_panel_01": {"lines": self.lines}}}
        self.settings = {
            "api_key": "",
            "api_key_saved": False,
            "llm_enabled": False,
            "control_enabled": True,
            "mode": "normal",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "url": "demo://test",
            "environment": "custom",
        }
        self.demo = True
        self.catalog = []
        self.connection = "demo"
        self.api = None
        self.agent = None
        self.agent_task = None
        self._demo_agent = False
        self.panel_log = self
        self.knowledge = None

    def redact(self, value):
        return value

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def write(self, kind, **fields):
        self.logged.append((kind, fields))

    async def panel_lines(self):
        return copy.deepcopy(self.lines)

    async def execute_line_command(self, command):
        self.executed.append(command)
        row = next(x for x in self.lines if x["line"] == command.line)
        row["state"] = "ON" if command.action == "line_on" else "OFF"
        return {
            "ok": True,
            "verified": True,
            "noop": False,
            "line": command.line,
            "label": command.label,
            "action": command.action,
            "state": row["state"],
        }

    def search(self, query):
        return {"results": [], "answer": None}

    def _alerts(self):
        return []


class ConversationControlTests(unittest.IsolatedAsyncioTestCase):
    def make(self, lines=None):
        service = FakeService(lines)
        return service, Conversation(service)

    async def ask(self, conversation, text):
        return (await conversation.ask(text))["reply"]

    async def test_command_is_proposed_not_executed(self):
        service, conversation = self.make()
        reply = await self.ask(conversation, "включи аварийное освещение")
        self.assertEqual(reply["verdict"], "CONTROL")
        self.assertIn("линия 7", reply["text"].lower())
        self.assertIn("да", reply["text"].lower())
        self.assertEqual(service.executed, [])

    async def test_confirmation_executes_once(self):
        service, conversation = self.make()
        await self.ask(conversation, "включи аварийное освещение")
        reply = await self.ask(conversation, "да")
        self.assertEqual(len(service.executed), 1)
        self.assertEqual(service.executed[0].action, "line_on")
        self.assertEqual(service.executed[0].line, 7)
        self.assertIn("Готово", reply["text"])
        self.assertIsNone(conversation._pending)

    async def test_second_confirmation_does_not_repeat(self):
        service, conversation = self.make()
        await self.ask(conversation, "включи аварийное освещение")
        await self.ask(conversation, "да")
        reply = await self.ask(conversation, "да")
        self.assertEqual(len(service.executed), 1)
        self.assertIn("Нет команды", reply["text"])

    async def test_cancellation_executes_nothing(self):
        service, conversation = self.make()
        await self.ask(conversation, "включи аварийное освещение")
        reply = await self.ask(conversation, "нет")
        self.assertEqual(service.executed, [])
        self.assertIn("Отменено", reply["text"])

    async def test_expired_proposal_is_not_executed(self):
        service, conversation = self.make()
        await self.ask(conversation, "включи аварийное освещение")
        conversation._pending["created_at"] -= 121
        reply = await self.ask(conversation, "да")
        self.assertEqual(service.executed, [])
        self.assertIn("истекло", reply["text"])

    async def test_unknown_line_is_rejected(self):
        service, conversation = self.make()
        reply = await self.ask(conversation, "включи линию 9")
        self.assertEqual(service.executed, [])
        self.assertIn("Линии 9 нет", reply["text"])

    async def test_already_on_line_is_not_sent(self):
        lines = panel_lines()
        lines[6]["state"] = "ON"
        service, conversation = self.make(lines)
        reply = await self.ask(conversation, "включи аварийное освещение")
        self.assertEqual(service.executed, [])
        self.assertIn("уже включена", reply["text"])

    async def test_tripped_protection_blocks_line_on(self):
        lines = panel_lines()
        lines[6]["protection_state"] = "TRIPPED"
        service, conversation = self.make(lines)
        reply = await self.ask(conversation, "включи аварийное освещение")
        self.assertEqual(service.executed, [])
        self.assertIn("защита", reply["text"].lower())

    async def test_control_disabled_does_not_propose(self):
        service, conversation = self.make()
        service.settings["control_enabled"] = False
        reply = await self.ask(conversation, "включи аварийное освещение")
        self.assertEqual(service.executed, [])
        self.assertIn("Управление выключено", reply["text"])

    async def test_no_live_lines_does_not_propose(self):
        service, conversation = self.make(lines=[])
        service.telemetry = {"nodes": {}}
        reply = await self.ask(conversation, "включи аварийное освещение")
        self.assertEqual(service.executed, [])
        self.assertIn("Нет live-данных", reply["text"])

    async def test_model_is_never_asked_for_a_control_turn(self):
        service, conversation = self.make()
        service.settings.update(api_key="local-test-only", llm_enabled=True)
        with patch("dome_agent.llm.DeepSeekAdvisor.advise", side_effect=AssertionError("must not call model")):
            await self.ask(conversation, "включи аварийное освещение")
            await self.ask(conversation, "да")
        self.assertEqual(len(service.executed), 1)

    async def test_new_command_replaces_pending(self):
        service, conversation = self.make()
        await self.ask(conversation, "включи аварийное освещение")
        await self.ask(conversation, "выключи линию 3")
        await self.ask(conversation, "да")
        self.assertEqual(len(service.executed), 1)
        self.assertEqual(service.executed[0].line, 3)
        self.assertEqual(service.executed[0].action, "line_off")


class DemoServiceControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "dome_policy.json").write_text(json.dumps({
            "default": "deny",
            "allow": {"smart_panel_01": {"actions": ["line_on", "line_off"], "values": {"line_on": list(range(1, 9)), "line_off": list(range(1, 9))}}},
        }), encoding="utf-8")
        self.service = DashboardService(self.root, demo=True, env_file="absent.env")
        self.service.settings.update(control_enabled=True, mode="normal", dry_run=False)
        self.service._mutation_lock = asyncio.Lock()
        self.conversation = Conversation(self.service)

    async def asyncTearDown(self):
        self.service.close()
        self.tmp.cleanup()

    async def test_confirmed_line_turns_on_in_demo_telemetry(self):
        first = (await self.conversation.ask("включи аварийное освещение"))["reply"]
        self.assertEqual(first["verdict"], "CONTROL")
        second = (await self.conversation.ask("да"))["reply"]
        self.assertIn("Готово", second["text"])
        line = next(x for x in self.service.telemetry["nodes"]["smart_panel_01"]["lines"] if x["line"] == 7)
        self.assertEqual(line["state"], "ON")

    async def test_chat_control_still_uses_policy_gate(self):
        self.service.settings["control_enabled"] = False
        reply = (await self.conversation.ask("включи аварийное освещение"))["reply"]
        self.assertIn("Управление выключено", reply["text"])
        line = next(x for x in self.service.telemetry["nodes"]["smart_panel_01"]["lines"] if x["line"] == 7)
        self.assertEqual(line["state"], "OFF")


if __name__ == "__main__":
    unittest.main()
