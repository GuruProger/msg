"""Acceptance tests for the production Dome API CLI against a local mock."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
MOCK = Path(__file__).with_name("mock_dome_api.py")
DEMO = REPO / "tests/fixtures/telemetry.jsonl"
LOG = Path(tempfile.gettempdir()) / f"dome_cli_audit_{os.getpid()}.jsonl"

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def start_server(extra: list[str] | None = None) -> tuple[subprocess.Popen, str]:
    args = [str(PYTHON), str(MOCK)] + (extra or [])
    proc = subprocess.Popen(
        args,
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    if not line.startswith("READY "):
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"mock failed to start: {line!r} {stderr}")
    port = line.split()[1]
    return proc, f"ws://127.0.0.1:{port}"


def run(
    args: list[str],
    *,
    url: str,
    key: str = "test-key",
    stdin: str | None = None,
    timeout: float = 25.0,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DOME_URL"] = url
    env["DOME_KEY"] = key
    env["DOME_KEY_FILE"] = ""
    env["AGENT_NAME"] = "test-agent"
    completed = subprocess.run(
        [str(PYTHON), "-m", "dome_control", *args],
        cwd=str(REPO),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        **({"input": stdin} if stdin is not None else {"stdin": subprocess.DEVNULL}),
    )
    preview = (completed.stdout or completed.stderr).strip().replace("\n", " | ")[:320]
    print(f"  $ {' '.join(args)} -> rc={completed.returncode} {preview}")
    return completed


def parse_json(completed: subprocess.CompletedProcess[str]) -> dict | list | None:
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        print(f"  ! stdout is not JSON: {exc}")
        return None


def expect_json(name: str, completed: subprocess.CompletedProcess[str]):
    data = parse_json(completed)
    check(name, completed.returncode == 0 and data is not None, f"rc={completed.returncode}")
    return data


def main() -> int:
    server, url = start_server()
    try:
        print(f"mock: {url}")

        data = expect_json("whoami --json", run(["whoami", "--json"], url=url))
        if isinstance(data, dict):
            check("whoami role/team", data.get("role") == "team" and int(data.get("team")) == 3, str(data))

        data = expect_json("ping --json", run(["ping", "--json"], url=url))
        if isinstance(data, dict):
            check("ping success", data.get("success") is True, str(data))

        data = expect_json("catalog --json", run(["catalog", "--json"], url=url))
        if isinstance(data, (dict, list)):
            text = json.dumps(data, ensure_ascii=False)
            check("catalog contains smart_panel", "smart_panel_01" in text)

        data = expect_json("nodes --json", run(["nodes", "--json"], url=url))
        if isinstance(data, dict):
            nodes = data.get("nodes") or {}
            check("nodes count", len(nodes) >= 30, f"count={len(nodes)}")

        data = expect_json("status --json", run(["status", "--json"], url=url))
        if isinstance(data, dict):
            check("status has nodes/game_time", "nodes" in data or "game_time" in data, str(list(data)))

        data = expect_json(
            "telemetry --json --node",
            run(["telemetry", "--json", "--node", "smart_panel_01"], url=url),
        )
        if isinstance(data, dict):
            check(
                "telemetry node",
                "lines" in data or "online" in data,
                str(list(data))[:200],
            )

        data = expect_json(
            "control --dry-run --json",
            run(
                ["control", "smart_panel_01", "line_off", "8", "--dry-run", "--json"],
                url=url,
            ),
        )
        if isinstance(data, dict):
            check("dry-run flagged", data.get("dry_run") is True or "dry" in json.dumps(data).lower(), str(data))

        data = expect_json(
            "control --yes --json",
            run(
                ["control", "smart_panel_01", "line_off", "8", "--yes", "--json"],
                url=url,
            ),
        )
        if isinstance(data, dict):
            text = json.dumps(data, ensure_ascii=False)
            check("control accepted", "success" in text or "accepted" in text.lower(), text[:250])

        completed = run(
            ["control", "smart_panel_01", "line_off", "8", "--json"], url=url
        )
        text = (completed.stdout + completed.stderr).lower()
        check(
            "control requires --yes",
            completed.returncode != 0 and "--yes" in text,
            f"rc={completed.returncode}",
        )

        completed = run(
            [
                "send",
                '{"type":"control","node_id":"smart_panel_01","action":"line_off","value":8}',
            ],
            url=url,
        )
        text = (completed.stdout + completed.stderr).lower()
        check(
            "raw control requires --yes",
            completed.returncode != 0 and "--yes" in text,
            f"rc={completed.returncode}",
        )

        data = expect_json(
            "query --json",
            run(["query", "printer_3d_01", "list_files", "--json"], url=url),
        )
        if isinstance(data, dict):
            check("query result", "gear_m1_z24.gcode" in json.dumps(data, ensure_ascii=False), str(data)[:250])

        data = expect_json("chat --json", run(["chat", "--json"], url=url))
        if isinstance(data, (dict, list)):
            check("chat has operator message", "Оператор" in json.dumps(data, ensure_ascii=False))

        data = expect_json(
            "say --json",
            run(["say", "hello from acceptance", "--json"], url=url),
        )
        if isinstance(data, dict):
            check("say success", "hello from acceptance" in json.dumps(data, ensure_ascii=False), str(data)[:250])

        data = expect_json(
            "send --json",
            run(["send", '{"type":"ping"}', "--json"], url=url),
        )
        if isinstance(data, dict):
            check("send ping success", data.get("success") is True, str(data))

        completed = run(["doctor", "--json"], url=url)
        data = parse_json(completed)
        check("doctor --json", completed.returncode == 0 and data is not None, f"rc={completed.returncode}")

        completed = run(
            ["control", "printer_3d_01", "remove_part", "--yes"], url=url
        )
        text = (completed.stdout + completed.stderr).lower()
        check(
            "operator-only rejected",
            completed.returncode != 0 and ("оператор" in text or "operator" in text),
            f"rc={completed.returncode}",
        )

        completed = run(
            ["control", "smart_panel_01", "bogus_action", "--yes"], url=url
        )
        text = (completed.stdout + completed.stderr).lower()
        check(
            "unknown action rejected",
            completed.returncode != 0 and ("разрешено" in text or "allowed" in text or "недоступ" in text),
            f"rc={completed.returncode}",
        )

        completed = run(["whoami"], url=url, key="bad-key")
        text = (completed.stdout + completed.stderr).lower()
        check(
            "bad key rejected",
            completed.returncode != 0 and ("auth" in text or "ключ" in text or "неверн" in text),
            f"rc={completed.returncode}",
        )

        completed = run(["whoami"], url="http://127.0.0.1:1")
        check("bad url rejected", completed.returncode != 0, f"rc={completed.returncode}")

        completed = run(
            ["whoami", "--timeout", "1"], url="ws://127.0.0.1:1", timeout=12
        )
        check("unreachable rejected", completed.returncode != 0, f"rc={completed.returncode}")

        completed = run(
            ["repl"],
            url=url,
            stdin="whoami\nstatus\nquit\n",
            timeout=30,
        )
        text = completed.stdout + completed.stderr
        check(
            "repl",
            completed.returncode == 0 and ("team" in text.lower() or "КУПОЛ" in text),
            f"rc={completed.returncode}",
        )

        data = expect_json(
            "demo status --json",
            run(["status", "--demo", str(DEMO), "--json"], url=url),
        )
        if isinstance(data, dict):
            check("demo status payload", "nodes" in data or "game_time" in data, str(list(data)))

        LOG.unlink(missing_ok=True)
        completed = run(
            ["status", "--json", "--log-file", str(LOG)], url=url
        )
        check("log run rc", completed.returncode == 0, f"rc={completed.returncode}")
        if LOG.exists():
            lines = [line for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
            ok = bool(lines)
            try:
                for line in lines:
                    json.loads(line)
            except json.JSONDecodeError:
                ok = False
            check("audit log is JSONL", ok, f"lines={len(lines)}")
            check("audit log has no key", "test-key" not in LOG.read_text(encoding="utf-8"))
        else:
            check("audit log created", False, "file missing")

        guest_server, guest_url = start_server(["--key", "guest-key"])
        try:
            data = expect_json("guest whoami --json", run(["whoami", "--json"], url=guest_url, key="guest-key"))
            if isinstance(data, dict):
                check("guest role", data.get("role") == "guest", str(data))
        finally:
            guest_server.terminate()
            guest_server.wait(timeout=5)

        revoke_server, revoke_url = start_server(["--revoke-after", "1"])
        try:
            completed = run(["ping", "--timeout", "3"], url=revoke_url, timeout=15)
            text = (completed.stdout + completed.stderr).lower()
            check(
                "session revoked handled",
                completed.returncode != 0 and ("revok" in text or "сесси" in text),
                f"rc={completed.returncode}",
            )
        finally:
            revoke_server.terminate()
            revoke_server.wait(timeout=5)

        print()
        if failures:
            print(f"FAILED: {len(failures)} checks: {failures}")
            return 1
        print("ALL ACCEPTANCE CHECKS PASSED")
        return 0
    finally:
        server.terminate()
        server.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
