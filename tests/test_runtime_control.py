"""Runtime overrides for mode and the Dome address. No network, no live dome."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dome_agent.agent import DomeAgent
from dome_agent.config import AgentConfig
from dome_agent.logging import AgentLog
from dome_agent.runtime import (
    RuntimeConfigError,
    normalize_dome_url,
    normalize_mode,
    read_runtime,
    read_status,
    status_path,
    write_json_atomic,
    write_runtime,
)
from dome_web.service import DashboardService, WebError

REPO = Path(__file__).resolve().parents[1]


class RuntimeFileTests(unittest.TestCase):
    def test_mode_and_url_validation(self) -> None:
        self.assertEqual(normalize_mode(" AUTO "), "auto")
        self.assertEqual(
            normalize_dome_url("wss://dome.example:443/ws"),
            "wss://dome.example:443/ws",
        )
        for bad in (
            "http://dome.example/ws",
            "wss:///ws",
            "wss://user:pass@dome.example/ws",
            "wss://dome.example:99999/ws",
            "wss://dome.example/ws?token=1",
            "wss://dome.example/ws#fragment",
            "",
        ):
            with self.assertRaises(RuntimeConfigError, msg=bad):
                normalize_dome_url(bad)
        with self.assertRaises(RuntimeConfigError):
            normalize_mode("turbo")

    def test_atomic_round_trip_merges_and_is_private(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent_runtime.json"
            write_runtime(path, mode="auto", dome_url="ws://host:8765/ws")
            state = read_runtime(path)
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual((state.mode, state.dome_url), ("auto", "ws://host:8765/ws"))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

            # A mode-only write must keep the address (and vice versa).
            write_runtime(path, mode="normal")
            state = read_runtime(path)
            assert state is not None
            self.assertEqual((state.mode, state.dome_url), ("normal", "ws://host:8765/ws"))
            write_runtime(path, dome_url="ws://other:9000/ws")
            state = read_runtime(path)
            assert state is not None
            self.assertEqual((state.mode, state.dome_url), ("normal", "ws://other:9000/ws"))

    def test_status_is_optional_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / "agent_runtime.json"
            self.assertIsNone(read_status(status_path(runtime)))
            write_json_atomic(status_path(runtime), {"running": True, "mode": "auto"})
            self.assertEqual(read_status(status_path(runtime))["mode"], "auto")


class RuntimeConfigTests(unittest.TestCase):
    def test_startup_applies_runtime_unless_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            runtime = log_dir / "agent_runtime.json"
            write_runtime(runtime, mode="auto", dome_url="wss://runtime.example:8443/ws")
            env = {
                "DOME_URL": "ws://env.example:1111/ws",
                "DOME_KEY": "secret",
                "AGENT_MODE": "normal",
                "AGENT_LLM": "0",
            }
            overrides = {"log_dir": log_dir, "state_file": log_dir / "state.json"}
            with mock.patch.dict(os.environ, env, clear=True):
                config = AgentConfig.from_env(env_file=None, overrides=overrides)
            self.assertEqual(config.mode, "auto")
            self.assertEqual(config.dome_url, "wss://runtime.example:8443/ws")
            self.assertEqual(config.runtime_file, runtime)

            with mock.patch.dict(os.environ, env, clear=True):
                explicit = AgentConfig.from_env(
                    env_file=None,
                    overrides={
                        **overrides,
                        "mode": "normal",
                        "dome_url": "ws://cli.example:2222/ws",
                    },
                )
            self.assertEqual(explicit.mode, "normal")
            self.assertEqual(explicit.dome_url, "ws://cli.example:2222/ws")

    def test_competition_rejects_runtime_yolo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            write_runtime(log_dir / "agent_runtime.json", mode="yolo")
            env = {
                "DOME_URL": "ws://mock:8765",
                "DOME_KEY": "secret",
                "AGENT_COMPETITION": "1",
                "AGENT_LLM": "0",
                "AGENT_POLICY_FILE": str(REPO / "dome_policy.competition.json"),
            }
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError) as ctx:
                    AgentConfig.from_env(
                        env_file=None,
                        overrides={
                            "log_dir": log_dir,
                            "state_file": log_dir / "state.json",
                        },
                    )
            self.assertIn("yolo", str(ctx.exception).casefold())


class StubAPI:
    def __init__(self) -> None:
        self.closed = False
        self.client = SimpleNamespace(role="team", team=3, url="ws://stub")

    async def close(self) -> None:
        self.closed = True


class FakeDomeAgentAPI:
    instances: list["FakeDomeAgentAPI"] = []

    def __init__(self, url: str, key: str, **kwargs) -> None:
        self.url = url
        self.key = key
        self.closed = False
        self.client = SimpleNamespace(role=None, team=None, url=url)
        FakeDomeAgentAPI.instances.append(self)

    async def close(self) -> None:
        self.closed = True


class AgentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_mode_switch_updates_gate_and_clears_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            config = AgentConfig(
                dome_url="ws://127.0.0.1:9999",
                dome_key="secret",
                log_dir=log_dir,
                state_file=log_dir / "state.json",
                runtime_file=log_dir / "agent_runtime.json",
                mode="normal",
                control_enabled=False,
                llm_enabled=False,
                kb_enabled=False,
                policy_file=REPO / "dome_policy.json",
            )
            log = AgentLog(log_dir)
            api = StubAPI()
            agent = DomeAgent(config, api=api, log=log)
            agent._pending = object()  # type: ignore[assignment]
            write_runtime(config.runtime_file, mode="yolo")
            await agent._poll_runtime_overrides()
            self.assertEqual(agent.config.mode, "yolo")
            self.assertTrue(agent.gate.yolo)
            self.assertTrue(agent.config.control_enabled)
            self.assertIsNone(agent._pending)

            write_runtime(config.runtime_file, mode="auto")
            await agent._poll_runtime_overrides()
            self.assertEqual(agent.config.mode, "auto")
            self.assertFalse(agent.gate.yolo)
            self.assertFalse(agent.config.control_enabled)
            log.close()

            records = [
                json.loads(line)
                for path in log_dir.glob("agent_*.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            changes = [
                row
                for row in records
                if row["kind"] == "decision" and row.get("source") == "runtime"
            ]
            self.assertEqual([row["mode"] for row in changes], ["yolo", "auto"])
            self.assertTrue(changes[0]["pending_cleared"])

    async def test_url_change_rebuilds_owned_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            config = AgentConfig(
                dome_url="ws://127.0.0.1:9999",
                dome_key="secret",
                log_dir=log_dir,
                state_file=log_dir / "state.json",
                runtime_file=log_dir / "agent_runtime.json",
                mode="auto",
                llm_enabled=False,
                kb_enabled=False,
                policy_file=REPO / "dome_policy.json",
            )
            log = AgentLog(log_dir)
            agent = DomeAgent(config, api=StubAPI(), log=log)
            old_api = agent.api
            agent._owns_api = True
            write_runtime(config.runtime_file, dome_url="wss://new.example:8443/ws")
            with mock.patch("dome_agent.agent.DomeAgentAPI", FakeDomeAgentAPI):
                await agent._poll_runtime_overrides()
            self.assertTrue(old_api.closed)
            self.assertEqual(agent.config.dome_url, "wss://new.example:8443/ws")
            self.assertIsInstance(agent.api, FakeDomeAgentAPI)
            self.assertIs(agent.tool.api, agent.api)
            self.assertEqual(agent.api.url, "wss://new.example:8443/ws")
            log.close()

    async def test_shared_api_refuses_url_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            config = AgentConfig(
                dome_url="ws://127.0.0.1:9999",
                dome_key="secret",
                log_dir=log_dir,
                state_file=log_dir / "state.json",
                runtime_file=log_dir / "agent_runtime.json",
                mode="auto",
                llm_enabled=False,
                kb_enabled=False,
                policy_file=REPO / "dome_policy.json",
            )
            log = AgentLog(log_dir)
            agent = DomeAgent(config, api=StubAPI(), log=log)
            write_runtime(config.runtime_file, dome_url="ws://127.0.0.1:8888")
            await agent._poll_runtime_overrides()
            self.assertEqual(agent.config.dome_url, "ws://127.0.0.1:9999")
            log.close()
            records = [
                json.loads(line)
                for path in log_dir.glob("agent_*.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertTrue(
                any(
                    row["kind"] == "error"
                    and "shared with the panel" in row.get("error", "")
                    for row in records
                )
            )


class PanelRuntimeTests(unittest.TestCase):
    def service(self, tmp: str, *, competition: bool = False) -> DashboardService:
        env = {
            "AGENT_LOG_DIR": str(Path(tmp) / "logs"),
            "AGENT_RUNTIME_FILE": str(Path(tmp) / "logs" / "agent_runtime.json"),
            "AGENT_KB": "0",
            "AGENT_COMPETITION": "1" if competition else "0",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            service = DashboardService(Path(tmp), demo=False, env_file="absent.env")
        self.addCleanup(service.close)
        return service

    def test_mode_and_url_are_written_for_the_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self.service(tmp)
            result = service.apply_runtime({"mode": "auto"})
            self.assertTrue(result["ok"])
            self.assertEqual(service.runtime_snapshot()["requested"]["mode"], "auto")

            service.apply_runtime(
                {"dome_url": "wss://dome.example:443/ws", "ack_url": True}
            )
            state = read_runtime(service.runtime_file)
            assert state is not None
            self.assertEqual(state.mode, "auto")
            self.assertEqual(state.dome_url, "wss://dome.example:443/ws")

            with self.assertRaises(WebError) as ctx:
                service.apply_runtime({"dome_url": "ws://other.example:9000/ws"})
            self.assertEqual(ctx.exception.status, 409)
            self.assertIn("подтверждения", str(ctx.exception))

    def test_live_status_and_competition_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self.service(tmp, competition=True)
            write_json_atomic(
                service.runtime_status_file,
                {
                    "running": True,
                    "updated_at": "2099-01-01T00:00:00+00:00",
                    "mode": "normal",
                    "dome_url": "wss://dome.example/ws",
                    "pid": 1,
                    "run_id": "test",
                    "competition": True,
                },
            )
            snapshot = service.runtime_snapshot()
            self.assertTrue(snapshot["live"])
            self.assertTrue(snapshot["competition"])
            with self.assertRaises(WebError) as ctx:
                service.apply_runtime({"mode": "yolo", "unsafe_ack": True})
            self.assertEqual(ctx.exception.status, 403)
            service.apply_runtime({"mode": "auto"})
            self.assertEqual(read_runtime(service.runtime_file).mode, "auto")


if __name__ == "__main__":
    unittest.main()
