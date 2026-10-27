"""Offline failure scenarios against the local mock Dome.

These scenarios complement :mod:`dome_agent.replay`: replay covers historical
telemetry, while this module exercises the transport and budget failure modes
that only appear while the agent process is running.

Every scenario starts ``tests/mock_dome_api.py`` on an ephemeral port, runs the
real ``main.py`` entry point for a few seconds, and summarizes the JSONL log.
No external network, no live dome and no real LLM calls are used.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .config import project_root

REPO = project_root()
MOCK = REPO / "tests" / "mock_dome_api.py"
POLICY = REPO / "dome_policy.competition.json"

BASE_ENV = {
    "AGENT_LLM": "0",
    "AGENT_COMPETITION": "1",
    "AGENT_MODE": "auto",
    "AGENT_ALLOW_CONTROL": "1",
    "AGENT_DRY_RUN": "0",
    "AGENT_MIN_SEND_GAP": "0.2",
    "AGENT_RECONNECT_DELAY": "1",
    "AGENT_OBSERVATION_INTERVAL": "5",
    "AGENT_POLICY_FILE": str(POLICY),
    "DOME_KEY": "test-key",
    "DOME_KEY_FILE": "",
    "AGENT_NAME": "scenario-agent",
}

SCENARIOS: dict[str, dict[str, Any]] = {
    "baseline": {
        "mock": ["--operator-task", "Напечатать bracket_v2.gcode"],
        "env": {},
        "seconds": 5,
    },
    "no_llm": {
        "mock": ["--operator-task", "Напечатать bracket_v2.gcode"],
        "env": {"AGENT_LLM": "0"},
        "seconds": 4,
    },
    "no_rag": {
        "mock": ["--operator-task", "Напечатать bracket_v2.gcode"],
        "env": {"AGENT_KB_DIR": "/nonexistent/dome-rag", "AGENT_LLM": "0"},
        "seconds": 4,
    },
    "session_revoked": {
        "mock": ["--revoke-after", "2"],
        "env": {"AGENT_LLM": "0"},
        "seconds": 7,
    },
    "corrupt_json": {
        "mock": ["--corrupt-after", "2"],
        "env": {"AGENT_LLM": "0"},
        "seconds": 4,
    },
}


def _start_mock(extra: list[str]) -> tuple[subprocess.Popen[str], str]:
    proc = subprocess.Popen(
        [sys.executable, str(MOCK), "--port", "0", *extra],
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    if not line.startswith("READY "):
        stderr = proc.stderr.read() if proc.stderr else ""
        proc.kill()
        raise RuntimeError(f"mock did not start: {line!r} {stderr}")
    return proc, f"ws://127.0.0.1:{line.split()[1]}"


def _read_log(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def run_scenario(name: str) -> dict[str, Any]:
    spec = SCENARIOS[name]
    with tempfile.TemporaryDirectory(prefix=f"dome-scenario-{name}-") as tmp:
        root = Path(tmp)
        log_dir = root / "logs"
        state_file = root / "state.json"
        env_file = root / "empty.env"
        env_file.write_text("", encoding="utf-8")
        mock, url = _start_mock(["--key", "test-key", *spec["mock"]])
        try:
            env = os.environ.copy()
            env.update(BASE_ENV)
            env.update(spec.get("env") or {})
            env["DOME_URL"] = url
            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--env-file",
                    str(env_file),
                    "--run-seconds",
                    str(spec["seconds"]),
                    "--log-dir",
                    str(log_dir),
                    "--state-file",
                    str(state_file),
                ],
                cwd=str(REPO),
                env=env,
                text=True,
                capture_output=True,
                timeout=float(spec["seconds"]) + 25,
            )
        finally:
            mock.terminate()
            try:
                mock.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mock.kill()
                mock.wait(timeout=5)
            for stream in (mock.stdout, mock.stderr):
                if stream is not None:
                    stream.close()

        logs = sorted(log_dir.glob("agent_*.jsonl"))
        records = _read_log(logs[0]) if logs else []
        kinds = Counter(str(record.get("kind")) for record in records)
        connections = [
            (record.get("event"), (record.get("details") or {}).get("error"))
            for record in records
            if record.get("kind") == "connection"
        ]
        errors = [
            record
            for record in records
            if record.get("kind") == "error"
        ]
        verification = [
            record.get("ok")
            for record in records
            if record.get("kind") == "verification"
        ]
        start = next(
            (record for record in records if record.get("kind") == "start"),
            {},
        )
        competition = start.get("competition") or {}
        knowledge = start.get("knowledge") or {}
        stats = next(
            (record.get("stats") for record in reversed(records) if record.get("kind") == "stop"),
            {},
        ) or {}

        checks: dict[str, bool] = {}
        if name == "baseline":
            checks = {
                "started": bool(start),
                "telemetry_seen": int(stats.get("telemetry_frames") or 0) > 0,
                "action_sent": int(stats.get("actions") or 0) > 0,
                "verification_ok": any(value is True for value in verification),
                "competition_profile": bool(competition.get("enabled")),
                "no_errors": not errors,
            }
        elif name in {"no_llm", "no_rag"}:
            checks = {
                "started": bool(start),
                "telemetry_seen": int(stats.get("telemetry_frames") or 0) > 0,
                "llm_disabled": not bool(competition.get("llm_enabled")),
                "no_errors": not errors,
            }
            if name == "no_rag":
                checks["knowledge_missing"] = not bool(knowledge.get("available"))
            else:
                checks["knowledge_still_available"] = bool(knowledge.get("available"))
        elif name == "session_revoked":
            checks = {
                "started": bool(start),
                "auth_count": sum(1 for event, _ in connections if event == "auth") >= 2,
                "revoked_seen": any(event == "revoked" for event, _ in connections),
            }
        elif name == "corrupt_json":
            checks = {
                "started": bool(start),
                "telemetry_seen": int(stats.get("telemetry_frames") or 0) > 0,
                "no_errors": not errors,
            }

        return {
            "scenario": name,
            "ok": bool(checks) and all(checks.values()),
            "checks": checks,
            "returncode": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-500:],
            "kinds": dict(sorted(kinds.items())),
            "connections": connections,
            "errors": [
                {
                    "where": record.get("where"),
                    "error": str(record.get("error"))[:300],
                }
                for record in errors[:8]
            ],
            "verification": verification,
            "stats": stats,
            "competition": {
                "enabled": competition.get("enabled"),
                "policy_sha256": competition.get("policy_sha256"),
                "model": competition.get("deepseek_model"),
                "kb_available": competition.get("kb_available"),
            },
            "knowledge": knowledge,
        }


def run_all(names: list[str] | None = None) -> dict[str, Any]:
    selected = names or list(SCENARIOS)
    unknown = set(selected) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"unknown scenario(s): {', '.join(sorted(unknown))}")
    reports = [run_scenario(name) for name in selected]
    return {
        "ok": all(report["ok"] for report in reports),
        "scenarios": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dome-scenarios",
        description="Run the real agent against the local mock Dome failure scenarios.",
    )
    parser.add_argument("--scenario", action="append", default=None, help="scenario name (repeatable)")
    parser.add_argument("--list", action="store_true", help="list scenario names")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for name in SCENARIOS:
            print(name)
        return 0
    try:
        report = run_all(args.scenario)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"scenario error: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
