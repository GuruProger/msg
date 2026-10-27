"""Real bridge/client against the project's local mock, never external devices."""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from dome_web.service import DashboardService, WebError


class LocalMockBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "dome_policy.json").write_text(json.dumps({"default": "deny", "allow": {"smart_panel_01": {"actions": ["line_off"], "values": {"line_off": [8]}}}}), encoding="utf-8")
        script = Path(__file__).with_name("mock_dome_api.py")
        self.proc = subprocess.Popen([sys.executable, str(script), "--port", "0", "--key", "web-test-key"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ready = await asyncio.to_thread(self.proc.stdout.readline)
        if not ready.startswith("READY "):
            raise RuntimeError("Local mock failed to start: " + self.proc.stderr.read())
        self.service = DashboardService(self.root, demo=False, env_file="none.env")
        self.service.settings.update(url="ws://127.0.0.1:" + ready.split()[1], key="web-test-key", api_key="", llm_enabled=False, control_enabled=False, mode="auto", dry_run=False)
        self.service._mutation_lock = asyncio.Lock()
        await self.service._open()
        self.service.telemetry = await self.service.api.client.wait_for_telemetry(timeout=5)

    async def asyncTearDown(self):
        await self.service._close()
        self.service.loop.close()
        self.proc.terminate()
        await asyncio.to_thread(self.proc.wait, 5)
        self.proc.stdout.close()
        self.proc.stderr.close()
        self.tmp.cleanup()

    async def test_readonly_snapshot_uses_real_client(self):
        s = await self.service.snapshot()
        self.assertEqual(s["connection"]["role"], "team")
        self.assertFalse(s["settings"]["control_enabled"])
        self.assertNotIn("web-test-key", json.dumps(s))

    async def test_control_verified_against_mock_telemetry(self):
        self.service.settings["control_enabled"] = True
        r = await self.service.dispatch("control", {"node_id": "smart_panel_01", "action": "line_off", "value": 8})
        self.assertTrue(r["accepted"])
        self.assertTrue(r["verified"])

    async def test_missing_permissions_block_live_bridge(self):
        with self.assertRaises(WebError):
            await self.service.dispatch("control", {"node_id": "smart_panel_01", "action": "line_off", "value": 8})

    async def test_agent_uses_shared_connection_and_stops(self):
        old = self.service.api
        await self.service.dispatch("agent/start", {})
        await asyncio.sleep(.5)
        self.assertIs(self.service.agent.api, old)
        await self.service.dispatch("agent/stop", {})
        self.assertTrue(self.service.agent_task.done())

    async def test_agent_chat_uses_existing_model_client_without_controls(self):
        self.service.settings.update(api_key="fake-model-key", llm_enabled=True)
        captured = []
        def respond(request):
            captured.append(json.loads(request.content))
            self.assertNotIn("web-test-key", request.content.decode())
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"situation": "Нужна проверка", "operator_message": "Заряд проверен по переданным показаниям.", "plan": ["Сверить время измерения"], "risk": "low", "commands": [{"action": "line_off", "value": 8}]})}}]})
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        async def get_client(advisor):
            return client
        before = json.dumps(self.service.api.client.telemetry, sort_keys=True)
        try:
            with patch("dome_agent.llm.DeepSeekAdvisor._get_client", get_client):
                r = await self.service.dispatch("agent/chat", {"text": "Какой заряд батареи?"})
            self.assertEqual(r["reply"]["engine"], self.service.settings["model"])
            self.assertIn("Заряд проверен", r["reply"]["text"])
            self.assertIn("Какой заряд батареи?", captured[0]["messages"][1]["content"])
            self.assertEqual(json.loads(before)["nodes"]["smart_panel_01"]["lines"], self.service.api.client.telemetry["nodes"]["smart_panel_01"]["lines"])
        finally:
            await client.aclose()


if __name__ == "__main__":
    unittest.main()
