from __future__ import annotations

import asyncio
import time
import unittest

from dome_control.client import DomeClient, DomeTimeout, retry_policy


class TelemetryFreshnessTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self) -> DomeClient:
        return DomeClient("ws://example.test", "key", min_send_gap=0.0)

    async def test_fresh_cached_frame_is_returned(self) -> None:
        client = self.make_client()
        client.telemetry = {"nodes": {"a": {"online": True}}}
        client._last_telemetry_monotonic = time.monotonic()
        result = await client.wait_for_telemetry(max_age=5.0, timeout=0.05)
        self.assertIn("a", result["nodes"])
        self.assertLess(client.telemetry_age, 1.0)

    async def test_stale_cached_frame_times_out(self) -> None:
        client = self.make_client()
        client.telemetry = {"nodes": {"a": {}}}
        client._last_telemetry_monotonic = time.monotonic() - 10.0
        with self.assertRaises(DomeTimeout):
            await client.wait_for_telemetry(max_age=5.0, timeout=0.02)

    async def test_stale_cached_frame_can_be_replaced(self) -> None:
        client = self.make_client()
        client.telemetry = {"nodes": {"a": {"value": 1}}}
        client._last_telemetry_monotonic = time.monotonic() - 10.0

        async def update() -> None:
            await asyncio.sleep(0.01)
            client.telemetry = {"nodes": {"a": {"value": 2}}}
            client._last_telemetry_monotonic = time.monotonic()
            client._telemetry_event.set()

        task = asyncio.create_task(update())
        result = await client.wait_for_telemetry(max_age=5.0, timeout=1.0)
        await task
        self.assertEqual(result["nodes"]["a"]["value"], 2)

    async def test_next_frame_ignores_cached_frame(self) -> None:
        client = self.make_client()
        client.telemetry = {"nodes": {"a": {}}}
        client._last_telemetry_monotonic = time.monotonic()
        with self.assertRaises(DomeTimeout):
            await client.wait_for_telemetry(timeout=0.02, next_frame=True)


class RetryPolicyTests(unittest.TestCase):
    def test_short_timeout_fails_fast(self) -> None:
        self.assertEqual(retry_policy(1.0), (1, 0.5))

    def test_medium_timeout(self) -> None:
        self.assertEqual(retry_policy(5.0), (2, 1.0))

    def test_normal_timeout(self) -> None:
        self.assertEqual(retry_policy(10.0), (4, 2.0))


if __name__ == "__main__":
    unittest.main()
