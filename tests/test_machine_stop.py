"""Stop/pause requests are confirmed, never silently turned into a print."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from dome_agent.agent import DomeAgent
from dome_agent.analysis import Snapshot
from dome_agent.chat import ChatMessage, parse_operator_task
from dome_agent.gate import CommandGate
from dome_agent.logging import AgentLog
from dome_agent.rules import ActionSpec
from tests.test_dome_agent import FakeAPI, base_telemetry, make_config

REPO = Path(__file__).resolve().parents[1]


def operator_message(message_id: int, text: str) -> ChatMessage:
    return ChatMessage.from_raw(
        {
            "id": message_id,
            "time": 1.0,
            "text": text,
            "sender": {"role": "operator", "name": "Оператор"},
        }
    )


class MachineIntentTests(unittest.TestCase):
    def kinds(self, text: str) -> list[tuple[str, str | None, int | None]]:
        task = parse_operator_task(operator_message(1, text))
        if task is None:
            return []
        return [(item.kind, item.file, item.line) for item in task.items]

    def test_stop_never_becomes_a_print(self) -> None:
        for text in (
            "останови печать part.gcode",
            "выключи 3д печать",
            "отключи принтер",
            "stop print",
        ):
            with self.subTest(text=text):
                kinds = self.kinds(text)
                self.assertEqual(kinds[0][0], "print_stop")
                self.assertNotIn("print", [kind for kind, _, _ in kinds])

    def test_pause_is_recognised(self) -> None:
        for text in ("пауза принтера", "поставь печать на паузу", "приостанови печать"):
            with self.subTest(text=text):
                self.assertEqual(self.kinds(text)[0][0], "print_pause")

    def test_cnc_stop_is_recognised(self) -> None:
        self.assertEqual(self.kinds("останови фрезер")[0][0], "cnc_stop")

    def test_negated_stop_is_not_a_command(self) -> None:
        for text in ("не выключай принтер", "не надо останавливать печать"):
            with self.subTest(text=text):
                self.assertFalse(
                    any(
                        kind.endswith(("_stop", "_pause"))
                        for kind, _, _ in self.kinds(text)
                    )
                )

    def test_print_and_power_commands_stay_intact(self) -> None:
        self.assertEqual(self.kinds("напечатай part.gcode")[0][0], "print")
        self.assertEqual(self.kinds("выключи линию 3")[0], ("line_off", None, 3))

    def test_competition_policy_allows_confirmed_line_off(self) -> None:
        gate = CommandGate(
            REPO / "dome_policy.competition.json",
            control_enabled=True,
            competition=True,
        )
        line_off = ActionSpec(
            key="stop-print",
            node_id="smart_panel_01",
            action="line_off",
            value=3,
            reason="operator stop",
        )
        stop_job = ActionSpec(
            key="stop-job",
            node_id="printer_3d_01",
            action="stop_job",
            reason="operator stop",
        )
        self.assertTrue(gate.check(line_off).allowed)
        self.assertFalse(gate.check(stop_job).allowed)


class MachineStopAgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.api = FakeAPI(base_telemetry())
        self.api.telemetry["nodes"]["printer_3d_01"].update(
            {
                "state": "PRINTING",
                "file_name": "part.gcode",
                "paused": False,
                "bed_clear": True,
            }
        )
        self.config = make_config(self.tmp.name, mode="auto", dry_run=False)
        self.log = AgentLog(self.config.log_dir)
        self.addCleanup(self.log.close)
        self.agent = DomeAgent(self.config, api=self.api, log=self.log)

    async def request(self, message_id: int, text: str) -> None:
        await self.agent._on_operator_message(
            operator_message(message_id, text),
            Snapshot(self.api.telemetry),
        )

    async def confirm(self, message_id: int, text: str = "да") -> None:
        self.agent._incoming.append(operator_message(message_id, text))
        await self.agent._handle_pending()

    def sent(self) -> list[str]:
        return [item["text"] for item in self.api.sent_messages]

    async def test_stop_requires_confirmation_and_switches_line_off(self) -> None:
        await self.request(10, "останови печать part.gcode")

        pending = self.agent._pending
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.source, "operator_stop")
        self.assertEqual(pending.commands[0].action, "line_off")
        self.assertEqual(pending.commands[0].value, 3)
        self.assertEqual(self.api.actions, [])
        self.assertTrue(any("Подтвердить остановку" in text for text in self.sent()))
        self.assertIsNone(self.agent.state.task)

        await self.confirm(11)

        self.assertEqual(
            [(item["node_id"], item["action"], item["value"]) for item in self.api.actions],
            [("smart_panel_01", "line_off", 3)],
        )
        self.assertIsNone(self.agent._pending)

    async def test_cancel_sends_nothing(self) -> None:
        await self.request(10, "останови печать part.gcode")
        await self.confirm(11, "отмена")
        self.assertEqual(self.api.actions, [])
        self.assertIsNone(self.agent._pending)

    async def test_old_yes_cannot_confirm_a_later_request(self) -> None:
        await self.request(10, "пауза принтера")
        self.agent._incoming.append(operator_message(5, "да"))
        await self.agent._handle_pending()
        self.assertEqual(self.api.actions, [])
        self.assertIsNotNone(self.agent._pending)

        await self.confirm(11)
        self.assertEqual(len(self.api.actions), 1)

    async def test_pause_explains_the_api_and_offers_a_stop(self) -> None:
        await self.request(10, "поставь печать на паузу")
        pending = self.agent._pending
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertTrue(any("не принимает pause" in text for text in self.sent()))
        self.assertEqual(pending.commands[0].action, "line_off")
        await self.confirm(11)
        self.assertEqual(len(self.api.actions), 1)

    async def test_idle_machine_is_not_stopped(self) -> None:
        self.api.telemetry["nodes"]["printer_3d_01"].update(
            {"state": "IDLE", "paused": False}
        )
        await self.request(10, "останови печать")
        self.assertIsNone(self.agent._pending)
        self.assertTrue(any("не выполняет задание" in text for text in self.sent()))

    async def test_disabled_control_asks_the_operator_to_stop_at_the_panel(self) -> None:
        config = make_config(self.tmp.name, mode="auto", control_enabled=False)
        log = AgentLog(config.log_dir)
        self.addCleanup(log.close)
        agent = DomeAgent(config, api=self.api, log=log)
        await agent._on_operator_message(
            operator_message(10, "останови печать"),
            Snapshot(self.api.telemetry),
        )
        self.assertIsNone(agent._pending)
        self.assertTrue(
            any("AGENT_ALLOW_CONTROL=0" in text for text in self.sent())
        )

    async def test_cycle_confirms_in_auto_mode(self) -> None:
        now = time.time()
        self.api.messages = [
            {
                "id": 21,
                "time": now,
                "text": "останови печать part.gcode",
                "sender": {"role": "operator", "name": "Оператор"},
                "to": None,
            }
        ]
        await self.agent._cycle()
        self.assertIsNotNone(self.agent._pending)
        self.assertEqual(self.api.actions, [])

        self.api.messages.append(
            {
                "id": 22,
                "time": now + 1,
                "text": "да",
                "sender": {"role": "operator", "name": "Оператор"},
                "to": None,
            }
        )
        self.agent._last_chat_poll = 0.0
        await self.agent._cycle()
        self.assertEqual(
            [(item["node_id"], item["action"], item["value"]) for item in self.api.actions],
            [("smart_panel_01", "line_off", 3)],
        )
        self.assertIsNone(self.agent._pending)

    async def test_competition_profile_allows_a_confirmed_stop(self) -> None:
        config = make_config(
            self.tmp.name,
            mode="auto",
            dry_run=False,
            competition=True,
            policy_file=REPO / "dome_policy.competition.json",
            llm_enabled=False,
        )
        log = AgentLog(config.log_dir)
        self.addCleanup(log.close)
        agent = DomeAgent(config, api=self.api, log=log)
        await agent._on_operator_message(
            operator_message(10, "останови печать part.gcode"),
            Snapshot(self.api.telemetry),
        )
        agent._incoming.append(operator_message(11, "да"))
        await agent._handle_pending()
        self.assertEqual(
            [(item["node_id"], item["action"], item["value"]) for item in self.api.actions],
            [("smart_panel_01", "line_off", 3)],
        )
        self.assertFalse(any(item["action"] == "stop_job" for item in self.api.actions))

    async def test_stop_is_logged_without_internal_details(self) -> None:
        await self.request(10, "останови печать part.gcode")
        records = [
            json.loads(line)
            for path in self.config.log_dir.glob("agent_*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        requests = [
            row
            for row in records
            if row["kind"] == "decision"
            and row.get("situation") == "Запрошена остановка станка"
        ]
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["action"], "stop")
        self.assertEqual(requests[0]["pending_after_id"], 10)
        self.assertNotIn("DOME_KEY", json.dumps(requests[0]))


if __name__ == "__main__":
    unittest.main()
