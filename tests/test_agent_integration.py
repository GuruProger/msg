"""End-to-end agent test against the local mock Dome WebSocket server."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MOCK = REPO / "tests" / "mock_dome_api.py"


class AgentIntegrationTests(unittest.TestCase):
    def test_agent_reads_operator_task_and_starts_print(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = root / "empty.env"
            env_file.write_text("", encoding="utf-8")
            log_dir = root / "logs"
            state_file = root / "state.json"

            mock_process = subprocess.Popen(
                [
                    sys.executable,
                    str(MOCK),
                    "--port",
                    "0",
                    "--key",
                    "test-key",
                    "--operator-task",
                    "Напечатать bracket_v2.gcode",
                ],
                cwd=str(REPO),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                assert mock_process.stdout is not None
                ready = mock_process.stdout.readline().strip()
                self.assertTrue(ready.startswith("READY "), ready)
                url = f"ws://127.0.0.1:{ready.split()[1]}"
                env = os.environ.copy()
                env.update(
                    {
                        "DOME_URL": url,
                        "DOME_KEY": "test-key",
                        "DOME_KEY_FILE": "",
                        "AGENT_NAME": "integration-agent",
                        "AGENT_LLM": "0",
                        "AGENT_DRY_RUN": "0",
                        "AGENT_ALLOW_CONTROL": "1",
                        "AGENT_MODE": "auto",
                        "AGENT_MIN_SEND_GAP": "0.2",
                    }
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        "main.py",
                        "--env-file",
                        str(env_file),
                        "--run-seconds",
                        "4",
                        "--log-dir",
                        str(log_dir),
                        "--state-file",
                        str(state_file),
                    ],
                    cwd=str(REPO),
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
                stderr = (completed.stderr or "").strip()
                self.assertEqual(completed.returncode, 0, stderr)
                records = [
                    json.loads(line)
                    for line in sorted(log_dir.glob("agent_*.jsonl"))[0]
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ]
                kinds = [record["kind"] for record in records]
                self.assertEqual(kinds[0], "start")
                self.assertEqual(kinds[-1], "stop")
                self.assertEqual(records[-1]["reason"], "run_seconds")
                self.assertTrue(any(record["kind"] == "message" and record.get("direction") == "in" for record in records))
                actions = [record for record in records if record["kind"] == "action"]
                self.assertTrue(
                    any(
                        record["request"].get("node_id") == "printer_3d_01"
                        and record["request"].get("action") == "start_job"
                        and record["request"].get("value") == "bracket_v2.gcode"
                        for record in actions
                    ),
                    actions,
                )
                verifications = [record for record in records if record["kind"] == "verification"]
                self.assertTrue(any(record.get("ok") is True for record in verifications), verifications)
                self.assertNotIn("test-key", json.dumps(records))
            finally:
                mock_process.terminate()
                try:
                    mock_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    mock_process.kill()
                    mock_process.wait(timeout=5)
                for stream in (mock_process.stdout, mock_process.stderr):
                    if stream is not None:
                        stream.close()


if __name__ == "__main__":
    unittest.main()
