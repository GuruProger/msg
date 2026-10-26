from __future__ import annotations

import unittest
from typing import Any

from dome_control.agent_api import DomeAgentAPI, tool_specs
from dome_control.client import MISSING, DomeTimeout
from dome_control.domain import DomeValidationError


CATALOG = [
    {
        "node_id": "smart_panel_01",
        "code_name": "smart_panel",
        "title": "Smart panel",
        "system": "Power",
        "category": "power",
        "controls": ["line_on", "line_off"],
        "control_aliases": {"line_on": ["enable_line"]},
        "operator_controls": ["inspect_line"],
        "queries": [],
        "service_actions": ["reconnect"],
    },
    {
        "node_id": "meeting_01",
        "code_name": "meeting",
        "title": "Sandbox",
        "system": "Sandbox",
        "category": "sandbox",
        "controls": ["post_message"],
        "control_aliases": {"post_message": ["say"]},
        "operator_controls": [],
        "queries": ["read_messages"],
        "service_actions": [],
    },
]


def frame(value: int = 1) -> dict[str, Any]:
    return {
        "nodes": {
            "smart_panel_01": {
                "online": True,
                "value": value,
                "lines": [{"line": 3, "state": "OFF" if value == 1 else "ON"}],
            },
            "meeting_01": {"message_count": 1},
        },
        "game_time": {"day": 0, "hour": 12.0},
    }


class FakeClient:
    def __init__(self) -> None:
        self.url = "ws://fake"
        self.role = "team"
        self.team = 3
        self.name = "test-agent"
        self.telemetry_timestamp = 123.0
        self.telemetry_age = 0.25
        self.catalog = CATALOG
        self.control_calls: list[tuple[str, str, Any]] = []
        self.query_calls: list[tuple[str, str, dict[str, Any]]] = []
        self._frames = [frame(1), frame(2)]
        self._fallback = frame(2)

    async def describe_nodes(self) -> list[dict[str, Any]]:
        return self.catalog

    async def wait_for_telemetry(
        self,
        timeout: float = 10.0,
        *,
        next_frame: bool = False,
        max_age: float | None = None,
    ) -> dict[str, Any]:
        if self._frames:
            return self._frames.pop(0)
        return self._fallback

    async def control(self, node_id: str, action: str, value: Any = MISSING) -> dict[str, Any]:
        self.control_calls.append((node_id, action, value))
        return {"success": True, "request_id": "1", "mode": "virtual"}

    async def query(
        self,
        node_id: str,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.query_calls.append((node_id, query, params or {}))
        return {"success": True, "result": [{"name": "part.gcode"}]}

    async def read_messages(self, *, since_id: int | None = None, limit: int = 50) -> list[dict]:
        return [{"id": 1, "text": "hello"}]

    async def post_message(
        self,
        text: str,
        *,
        to: str | None = "operator",
        name: str | None = None,
    ) -> dict[str, Any]:
        return {"success": True, "message": {"text": text, "to": to, "name": name}}

    async def ping(self) -> dict[str, Any]:
        return {"success": True, "type": "ping_response"}

    async def close(self) -> None:
        return None


class BeforeUnavailableClient(FakeClient):
    """Fresh before-telemetry is unavailable; after frames still work."""

    async def wait_for_telemetry(
        self,
        timeout: float = 10.0,
        *,
        next_frame: bool = False,
        max_age: float | None = None,
    ) -> dict[str, Any]:
        if not next_frame:
            raise DomeTimeout("no fresh before frame")
        return await super().wait_for_telemetry(
            timeout,
            next_frame=next_frame,
            max_age=max_age,
        )


class AgentAPITests(unittest.IsolatedAsyncioTestCase):
    def make_api(self) -> tuple[DomeAgentAPI, FakeClient]:
        client = FakeClient()
        return DomeAgentAPI(client=client), client

    async def test_get_telemetry_for_node(self) -> None:
        api, _ = self.make_api()
        result = await api.get_telemetry("smart_panel_01")
        self.assertEqual(result["node_id"], "smart_panel_01")
        self.assertTrue(result["telemetry"]["online"])
        self.assertEqual(result["server_timestamp"], 123.0)

    async def test_get_status_includes_all_nodes(self) -> None:
        api, _ = self.make_api()
        result = await api.get_status()
        self.assertIn("smart_panel_01", result["nodes"])
        self.assertIn("game_time", result)

    async def test_list_devices_filters(self) -> None:
        api, _ = self.make_api()
        power = await api.list_devices(system="power")
        self.assertEqual([node["node_id"] for node in power], ["smart_panel_01"])
        found = await api.list_devices(search="sandbox")
        self.assertEqual([node["node_id"] for node in found], ["meeting_01"])

    async def test_get_device_combines_catalog_and_telemetry(self) -> None:
        api, _ = self.make_api()
        result = await api.get_device("smart_panel")
        self.assertEqual(result["node_id"], "smart_panel_01")
        self.assertEqual(result["catalog"]["node_id"], "smart_panel_01")
        self.assertTrue(result["telemetry"]["online"])

    async def test_control_dry_run_does_not_send(self) -> None:
        api, client = self.make_api()
        result = await api.control_device("smart_panel_01", "enable_line", 3, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["request"]["action"], "line_on")
        self.assertEqual(result["request"]["value"], 3)
        self.assertEqual(client.control_calls, [])

    async def test_control_validates_and_verifies(self) -> None:
        api, client = self.make_api()
        result = await api.control_device("smart_panel_01", "line_on", 3)
        self.assertTrue(result["success"])
        self.assertEqual(client.control_calls, [("smart_panel_01", "line_on", 3)])
        self.assertEqual(result["before"]["lines"][0]["state"], "OFF")
        self.assertEqual(result["after"]["lines"][0]["state"], "ON")
        self.assertIn("lines", result["diff"])
        self.assertFalse(result["unknown"])

    async def test_missing_before_blocks_non_emergency_control(self) -> None:
        client = BeforeUnavailableClient()
        api = DomeAgentAPI(client=client)
        result = await api.control_device("smart_panel_01", "line_on", 3)
        self.assertFalse(result["success"])
        self.assertTrue(result["blocked"])
        self.assertFalse(result["sent"])
        self.assertIn("before-telemetry", result["error"])
        self.assertEqual(client.control_calls, [])

    async def test_missing_before_allows_emergency_as_unknown(self) -> None:
        client = BeforeUnavailableClient()
        api = DomeAgentAPI(client=client)
        result = await api.control_device(
            "smart_panel_01",
            "line_off",
            3,
            allow_without_before=True,
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["sent"])
        self.assertTrue(result["unknown"])
        self.assertIsNone(result["diff"])
        self.assertEqual(client.control_calls, [("smart_panel_01", "line_off", 3)])

    async def test_operator_action_is_rejected(self) -> None:
        api, client = self.make_api()
        with self.assertRaises(DomeValidationError):
            await api.control_device("smart_panel_01", "inspect_line")
        self.assertEqual(client.control_calls, [])

    async def test_query_device(self) -> None:
        api, client = self.make_api()
        result = await api.query_device("meeting_01", "read_messages")
        self.assertTrue(result["success"])
        self.assertEqual(client.query_calls, [("meeting_01", "read_messages", {})])

    async def test_call_tool_dispatch(self) -> None:
        api, _ = self.make_api()
        result = await api.call_tool("dome_get_telemetry", {"node_id": "smart_panel_01"})
        self.assertEqual(result["node_id"], "smart_panel_01")
        result = await api.call_tool("dome_control_device", {
            "node_id": "smart_panel_01",
            "action": "line_on",
            "value": 3,
            "dry_run": True,
        })
        self.assertTrue(result["dry_run"])

    async def test_whoami_and_ping(self) -> None:
        api, _ = self.make_api()
        self.assertEqual((await api.whoami())["team"], 3)
        self.assertTrue((await api.ping())["success"])

    def test_tool_specs_are_json_serializable(self) -> None:
        specs = tool_specs()
        self.assertEqual({item["name"] for item in specs}, {
            "dome_get_telemetry",
            "dome_get_status",
            "dome_list_devices",
            "dome_get_device",
            "dome_control_device",
            "dome_query_device",
            "dome_read_messages",
            "dome_post_message",
            "dome_whoami",
            "dome_ping",
        })


if __name__ == "__main__":
    unittest.main()
