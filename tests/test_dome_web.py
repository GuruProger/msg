"""Local UI bridge tests. No external Dome or LLM requests."""
from __future__ import annotations

import asyncio
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from dome_web.server import DashboardHTTPServer
from dome_web.service import DashboardService, WebError

POLICY = {"default": "deny", "allow": {"smart_panel_01": {"actions": ["line_on", "line_off"], "values": {"line_on": list(range(1, 9)), "line_off": list(range(1, 9))}}, "printer_3d_01": ["start_job", "home", "resume"], "access_control_01": ["unlock_door"]}}


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "dome_policy.json").write_text(json.dumps(POLICY), encoding="utf-8")
        self.service = DashboardService(self.root, demo=True, env_file="absent.env")
        self.service.settings.update(control_enabled=False, mode="normal", dry_run=False)
        self.service._mutation_lock = asyncio.Lock()

    async def asyncTearDown(self):
        self.service.close()
        self.tmp.cleanup()

    async def test_snapshot_contains_telemetry_catalog_and_csrf(self):
        s = await self.service.snapshot()
        self.assertGreaterEqual(len(s["catalog"]), 36)
        self.assertIn("solar_inverter_01", s["telemetry"]["nodes"])
        self.assertTrue(s["csrf"])

    async def test_secret_values_never_returned(self):
        await self.service.dispatch("settings", {"key": "PRIVATE-DOME-KEY", "api_key": "PRIVATE-LLM-KEY"})
        payload = json.dumps(await self.service.snapshot()) + self.service.log_download()
        self.assertNotIn("PRIVATE-DOME-KEY", payload)
        self.assertNotIn("PRIVATE-LLM-KEY", payload)

    async def test_default_readonly_blocks_control(self):
        with self.assertRaises(WebError):
            await self.service.dispatch("control", {"node_id": "smart_panel_01", "action": "line_off", "value": 8})

    async def test_enabled_command_changes_demo_state(self):
        self.service.settings["control_enabled"] = True
        result = await self.service.dispatch("control", {"node_id": "smart_panel_01", "action": "line_off", "value": 8})
        self.assertTrue(result["verified"])
        line = next(l for l in self.service.telemetry["nodes"]["smart_panel_01"]["lines"] if l["line"] == 8)
        self.assertEqual(line["state"], "OFF")

    async def test_disallowed_parameter_blocked(self):
        self.service.settings["control_enabled"] = True
        with self.assertRaises(WebError):
            await self.service.dispatch("control", {"node_id": "smart_panel_01", "action": "line_off", "value": 99})

    async def test_hard_deny_survives_yolo(self):
        self.service.settings.update(mode="yolo", control_enabled=True)
        with self.assertRaises(WebError):
            await self.service.dispatch("control", {"node_id": "access_control_01", "action": "unlock_door", "value": "STORAGE"})

    async def test_yolo_requires_explicit_ack(self):
        with self.assertRaises(WebError):
            await self.service.dispatch("settings", {"mode": "yolo"})

    async def test_agent_and_manual_control_do_not_compete(self):
        self.service.settings["control_enabled"] = True
        await self.service.dispatch("agent/start", {})
        with self.assertRaises(WebError):
            await self.service.dispatch("control", {"node_id": "smart_panel_01", "action": "line_off", "value": 8})
        await self.service.dispatch("agent/stop", {})
        self.assertFalse((await self.service.snapshot())["agent"]["running"])

    async def test_settings_require_stopped_agent(self):
        await self.service.dispatch("agent/start", {})
        with self.assertRaises(WebError):
            await self.service.dispatch("settings", {"mode": "auto"})

    async def test_chat_and_dry_run(self):
        await self.service.dispatch("chat", {"text": "Проверка", "to": "operator"})
        self.assertEqual(self.service.messages[-1]["text"], "Проверка")
        self.service.settings["dry_run"] = True
        with self.assertRaises(WebError):
            await self.service.dispatch("chat", {"text": "Не отправлять"})

    async def test_query_only_from_catalog(self):
        r = await self.service.dispatch("query", {"node_id": "printer_3d_01", "query": "list_files"})
        self.assertTrue(r["result"])
        with self.assertRaises(WebError):
            await self.service.dispatch("query", {"node_id": "printer_3d_01", "query": "unknown"})

    async def test_start_requires_known_file(self):
        self.service.settings["control_enabled"] = True
        with self.assertRaises(WebError):
            await self.service.dispatch("control", {"node_id": "printer_3d_01", "action": "start_job", "value": "unknown.gcode"})

    async def test_unknown_operation_rejected(self):
        with self.assertRaises(WebError):
            await self.service.dispatch("shell", {"cmd": "ignored"})

    async def test_agent_chat_fallback_does_not_send_meeting_or_controls(self):
        self.service.settings.update(api_key="", llm_enabled=False)
        before = json.dumps(self.service.telemetry, sort_keys=True)
        count = len(self.service.messages)
        r = await self.service.dispatch("agent/chat", {"text": "Какой заряд АКБ?"})
        self.assertEqual(r["reply"]["engine"], "Справка без модели")
        self.assertIn("battery_01", r["reply"]["text"])
        self.assertEqual(len(self.service.messages), count)
        self.assertEqual(json.dumps(self.service.telemetry, sort_keys=True), before)
        self.assertEqual(len((await self.service.snapshot())["agent_chat"]), 2)

    async def test_change_complex_clears_old_key_and_telemetry(self):
        self.service.settings.update(url="wss://test.example", key="OLD-KEY", key_file="old.txt", control_enabled=True)
        await self.service.dispatch("settings", {"url": "wss://real.example", "environment": "real"})
        self.assertFalse(self.service.settings["key"])
        self.assertFalse(self.service.settings["key_file"])
        self.assertFalse(self.service.settings["control_enabled"])
        self.assertFalse(self.service.telemetry["nodes"])

    async def test_new_complex_uses_new_key_and_configurable_endpoint(self):
        self.service.settings.update(url="wss://test.example", key="OLD-KEY")
        await self.service.dispatch("settings", {"url": "wss://real.example:443/path", "key": "NEW-KEY", "environment": "real"})
        self.assertEqual(self.service.settings["key"], "NEW-KEY")
        self.assertEqual(self.service.settings["url"], "wss://real.example:443/path")

    async def test_agent_chat_does_not_block_equipment_lock(self):
        self.service.settings.update(api_key="", llm_enabled=False)
        async with self.service._mutation_lock:
            r = await asyncio.wait_for(self.service.dispatch("agent/chat", {"text": "Статус Купола"}), 1)
        self.assertTrue(r["ok"])


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.service = DashboardService(Path(cls.tmp.name), demo=True, env_file="absent.env")
        cls.service.start()
        cls.server = DashboardHTTPServer(("127.0.0.1", 0), cls.service)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=2)
        cls.service.close()
        cls.tmp.cleanup()

    def request(self, method, url, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request(method, url, body, headers or {})
        r = c.getresponse()
        result = r.status, r.read(), dict(r.getheaders())
        c.close()
        return result

    def test_index_and_assets_served(self):
        for url in ("/", "/app.js", "/styles.css", "/favicon.svg"):
            status, body, headers = self.request("GET", url)
            self.assertEqual(status, 200)
            self.assertTrue(body)
            self.assertIn("Content-Security-Policy", headers)

    def test_no_arbitrary_files(self):
        self.assertEqual(self.request("GET", "/../dome_policy.json")[0], 404)
        self.assertEqual(self.request("GET", "/.env")[0], 404)

    def test_no_csrf_no_mutation(self):
        s, _, _ = self.request("POST", "/api/settings", "{}", {"Content-Type": "application/json"})
        self.assertEqual(s, 403)

    def test_cross_origin_mutation_rejected(self):
        s, _, _ = self.request("POST", "/api/ping", "{}", {"Content-Type": "application/json", "X-Dome-CSRF": self.service.csrf, "Origin": "https://foreign.invalid"})
        self.assertEqual(s, 403)

    def test_foreign_host_rejected(self):
        self.assertEqual(self.request("GET", "/api/state", headers={"Host": "foreign.invalid"})[0], 403)

    def test_valid_post_and_export(self):
        s, body, _ = self.request("POST", "/api/ping", "{}", {"Content-Type": "application/json", "X-Dome-CSRF": self.service.csrf})
        self.assertEqual(s, 200)
        self.assertTrue(json.loads(body)["ok"])
        s, body, headers = self.request("GET", "/api/logs/download")
        self.assertEqual(s, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        for line in body.decode().splitlines():
            json.loads(line)


if __name__ == "__main__":
    unittest.main()
