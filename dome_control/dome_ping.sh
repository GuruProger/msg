#!/usr/bin/env bash
# Team 3: auth + one application-level ping. No equipment or chat commands.
# Bash equivalent of dome_ping.ps1. Requires python3 + websockets.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

URL="${DOME_URL:-}"
KEY_FILE=""
TIMEOUT_SECONDS=10
DUMP=0

usage() {
    cat <<'EOF'
Usage: dome_ping.sh --url ws://HOST:PORT [--key-file PATH] [--timeout 1..60] [--dump]

  --url, -u       WebSocket URL (or DOME_URL)
  --key-file, -k  file with 'team: <key>' or 'DOME_KEY=<key>'
  --timeout, -t   total timeout per phase, seconds (default: 10)
  --dump          print every received JSON message, including telemetry
  --help, -h      show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url|-u)
            [[ $# -ge 2 ]] || { echo "DOME_PING: --url requires a value" >&2; exit 2; }
            URL="$2"
            shift 2
            ;;
        --key-file|-k)
            [[ $# -ge 2 ]] || { echo "DOME_PING: --key-file requires a value" >&2; exit 2; }
            KEY_FILE="$2"
            shift 2
            ;;
        --timeout|-t)
            [[ $# -ge 2 ]] || { echo "DOME_PING: --timeout requires a value" >&2; exit 2; }
            TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --dump)
            DUMP=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "DOME_PING: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$URL" ]]; then
    echo "DOME_PING: specify --url ws://HOST:PORT or set DOME_URL." >&2
    exit 2
fi
if [[ ! "$URL" =~ ^wss?://[^[:space:]]+$ ]]; then
    echo "DOME_PING: invalid WebSocket URL; use ws:// or wss://." >&2
    exit 2
fi
if [[ ! "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || (( TIMEOUT_SECONDS < 1 || TIMEOUT_SECONDS > 60 )); then
    echo "DOME_PING: --timeout must be an integer from 1 to 60." >&2
    exit 2
fi

read_key_file() {
    local file="$1"
    if [[ ! -r "$file" ]]; then
        echo "DOME_PING: cannot read key file: $file" >&2
        return 1
    fi
    awk '
        /^[[:space:]]*(team|DOME_KEY)[[:space:]]*[:=]/ {
            sub(/^[^:=]*[:=][[:space:]]*/, "")
            sub(/[[:space:]]+$/, "")
            print
            found = 1
            exit
        }
        NF {
            count++
            last = $0
        }
        END {
            if (!found && count == 1 && last !~ /^[A-Za-z_]+[[:space:]]*[:=]/) {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", last)
                print last
            }
        }
    ' "$file"
}

KEY="${DOME_KEY:-}"
if [[ -n "$KEY_FILE" ]]; then
    KEY="$(read_key_file "$KEY_FILE")" || exit 2
elif [[ -z "$KEY" ]]; then
    if [[ -n "${DOME_KEY_FILE:-}" ]]; then
        KEY_FILE="$DOME_KEY_FILE"
    else
        for candidate in "$REPO_DIR/Keys/team_3.txt" "$REPO_DIR/Dome_RAG-main/Keys/team_3.txt"; do
            if [[ -r "$candidate" ]]; then
                KEY_FILE="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$KEY_FILE" ]]; then
        echo "DOME_PING: team key not found; pass --key-file, set DOME_KEY_FILE, or place it at $REPO_DIR/Keys/team_3.txt." >&2
        exit 2
    fi
    KEY="$(read_key_file "$KEY_FILE")" || exit 2
fi

KEY="${KEY#"${KEY%%[![:space:]]*}"}"
KEY="${KEY%"${KEY##*[![:space:]]}"}"
if [[ -z "$KEY" ]]; then
    echo "DOME_PING: team key is empty or not found." >&2
    exit 2
fi

if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    PYTHON="$REPO_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "DOME_PING: python3 not found." >&2
    exit 2
fi
if ! "$PYTHON" -c 'import websockets' >/dev/null 2>&1; then
    echo "DOME_PING: python package 'websockets' is required." >&2
    exit 2
fi

export DOME_PING_URL="$URL"
export DOME_PING_KEY="$KEY"
export DOME_PING_TIMEOUT="$TIMEOUT_SECONDS"
export DOME_PING_DUMP="$DUMP"

exec "$PYTHON" - <<'PY'
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid

import websockets

URL = os.environ["DOME_PING_URL"]
KEY = os.environ["DOME_PING_KEY"]
TIMEOUT = float(os.environ["DOME_PING_TIMEOUT"])
DUMP = os.environ.get("DOME_PING_DUMP") == "1"


def fail(message: str) -> None:
    raise RuntimeError(message)


async def wait_for(ws, expected_type: str, *, request_id: str | None = None, deadline: float):
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            fail(f"timed out waiting for {expected_type}")
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if DUMP:
            print(f"<<< {raw}", flush=True)
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            fail("server returned invalid JSON")
        if not isinstance(message, dict):
            fail("expected a JSON object")
        if message.get("type") == "session_revoked":
            fail(f"session revoked: {message.get('reason') or ''}".rstrip(": "))
        if message.get("type") == expected_type and (
            request_id is None or str(message.get("request_id")) == request_id
        ):
            return message


async def main() -> int:
    print(f"DOME_PING: connecting to {URL}", flush=True)
    try:
        async with websockets.connect(
            URL,
            open_timeout=TIMEOUT,
            close_timeout=3,
            max_size=32 * 1024 * 1024,
        ) as ws:
            print("WebSocket connected.", flush=True)

            await ws.send(json.dumps({"type": "auth", "key": KEY}))
            auth = await wait_for(
                ws, "auth_response", deadline=time.monotonic() + TIMEOUT
            )
            if auth.get("success") is not True:
                fail(f"authentication rejected: {auth.get('error') or ''}".rstrip(": "))
            if str(auth.get("team")) != "3":
                fail(f"server did not confirm team 3 (team={auth.get('team')!r})")
            role = auth.get("role")
            if role not in ("team", "guest"):
                fail(f"unknown role: {role!r}")
            print(f"Authenticated: team 3; role: {role}.", flush=True)
            if role == "guest":
                print(
                    "Guest is valid: your team is not active; equipment control is unavailable.",
                    flush=True,
                )

            request_id = "team3-ping-" + uuid.uuid4().hex
            started = time.monotonic()
            await ws.send(json.dumps({"type": "ping", "request_id": request_id}))
            reply = await wait_for(
                ws,
                "ping_response",
                request_id=request_id,
                deadline=time.monotonic() + TIMEOUT,
            )
            elapsed = (time.monotonic() - started) * 1000
            if reply.get("success") is not True:
                fail(f"application-level ping rejected: {reply.get('error') or ''}".rstrip(": "))
            print(f"PING OK: {elapsed:.1f} ms; request_id matched.", flush=True)
            return 0
    except Exception as exc:
        print(f"DOME_PING: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


raise SystemExit(asyncio.run(main()))
PY
