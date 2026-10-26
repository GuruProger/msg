from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from dome_control.agent_api import DomeAgentAPI


ROOT = Path(__file__).resolve().parents[1]
MOCK = Path(__file__).with_name("mock_dome_api.py")


class AgentAPIIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.server = subprocess.Popen(
            [sys.executable, str(MOCK)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert self.server.stdout is not None
        line = self.server.stdout.readline().strip()
        if not line.startswith("READY "):
            stderr = self.server.stderr.read() if self.server.stderr else ""
            self.server.kill()
            raise RuntimeError(f"mock failed to start: {line!r} {stderr}")
        self.url = f"ws://127.0.0.1:{line.split()[1]}"

    async def asyncTearDown(self) -> None:
        self.server.terminate()
        try:
            await self._wait_process()
        except subprocess.TimeoutExpired:
            self.server.kill()
            self.server.wait(timeout=5)
        for stream in (self.server.stdout, self.server.stderr):
            if stream is not None:
                stream.close()

    async def _wait_process(self) -> None:
        # Avoid blocking the event loop while the mock closes.
        import asyncio

        await asyncio.to_thread(self.server.wait, timeout=5)

    async def test_agent_tools_against_mock(self) -> None:
        async with DomeAgentAPI(self.url, "test-key", name="agent-test", timeout=5.0) as api:
            whoami = await api.whoami()
            self.assertEqual(whoami["role"], "team")
            self.assertEqual(whoami["team"], 3)

            devices = await api.list_devices(search="smart")
            self.assertEqual([node["node_id"] for node in devices], ["smart_panel_01"])

            telemetry = await api.get_telemetry("smart_panel_01")
            self.assertTrue(telemetry["telemetry"]["online"])
            self.assertGreaterEqual(len(telemetry["telemetry"].get("lines") or []), 8)

            dry = await api.control_device("smart_panel_01", "line_enable", 3, dry_run=True)
            self.assertEqual(dry["request"]["action"], "line_on")

            result = await api.control_device("smart_panel_01", "line_on", 3)
            self.assertTrue(result["success"])
            self.assertIn("lines", result["diff"] or {})

            query = await api.query_device("cnc_01", "list_files")
            self.assertTrue(query["success"])
            self.assertGreaterEqual(len(query["result"]), 1)

    async def test_operator_action_is_rejected_against_mock(self) -> None:
        from dome_control.domain import DomeValidationError

        async with DomeAgentAPI(self.url, "test-key", name="agent-test", timeout=5.0) as api:
            with self.assertRaises(DomeValidationError):
                await api.control_device("printer_3d_01", "remove_part")


if __name__ == "__main__":
    unittest.main()
