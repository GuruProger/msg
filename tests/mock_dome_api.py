"""Mock Dome WebSocket API for acceptance tests of the production CLI."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import time
from pathlib import Path

import websockets

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dome_control.panel import load_demo_telemetry  # noqa: E402

TELEMETRY_PATH = REPO / "tests/fixtures/telemetry.jsonl"
BASE = load_demo_telemetry(str(TELEMETRY_PATH))

CATALOG = [
    {
        "node_id": "smart_panel_01",
        "code_name": "smart_panel",
        "title": "Умный щиток",
        "system": "Электроснабжение",
        "category": "power",
        "controls": ["line_on", "line_off", "reset_protection", "all_off"],
        "control_aliases": {
            "line_on": ["line_enable"],
            "line_off": ["line_disable"],
            "reset_protection": ["reset_rcd"],
        },
        "operator_controls": [],
        "queries": [],
        "service_actions": ["reconnect"],
    },
    {
        "node_id": "printer_3d_01",
        "code_name": "printer_3d",
        "title": "3D-принтер",
        "system": "Производство",
        "category": "production",
        "controls": [
            "delete_file",
            "home",
            "move",
            "resume",
            "set_speed",
            "start_job",
            "upload_file",
        ],
        "control_aliases": {"start_job": ["start_print", "print_file"]},
        "operator_controls": ["remove_part"],
        "queries": ["list_files"],
        "service_actions": ["reconnect"],
    },
    {
        "node_id": "cnc_01",
        "code_name": "cnc",
        "title": "ЧПУ-фрезер",
        "system": "Производство",
        "category": "production",
        "controls": [
            "home",
            "move",
            "probe_z",
            "resume",
            "set_zero",
            "start_job",
        ],
        "control_aliases": {"set_zero": ["zero_xy", "zero_z"]},
        "operator_controls": ["connect_probe", "remove_part"],
        "queries": ["list_files"],
        "service_actions": ["reconnect"],
    },
    {
        "node_id": "dome_automation_01",
        "code_name": "dome_automation",
        "title": "Автоматика купола",
        "system": "Автоматика",
        "category": "automation",
        "controls": ["reset_interlock", "set_data_trust", "set_primary_sensor"],
        "control_aliases": {"reset_interlock": ["clear_interlock"]},
        "operator_controls": [],
        "queries": [],
        "service_actions": ["reconnect"],
    },
    {
        "node_id": "meeting_01",
        "code_name": "meeting",
        "title": "Песочница",
        "system": "Песочница",
        "category": "chat",
        "controls": ["post_message"],
        "control_aliases": {"post_message": ["say"]},
        "operator_controls": [],
        "queries": ["read_messages"],
        "service_actions": [],
    },
]

MESSAGES = [
    {
        "id": 1,
        "time": time.time(),
        "sender": {"role": "operator", "team": None, "name": "Оператор"},
        "to": None,
        "text": "Задание: проверить щиток.",
    }
]


def catalog_for(node_id: str) -> dict | None:
    return next((item for item in CATALOG if item["node_id"] == node_id), None)


def allowed_actions(item: dict) -> set[str]:
    names = set(item.get("controls") or [])
    for key, aliases in (item.get("control_aliases") or {}).items():
        names.add(key)
        names.update(aliases)
    names.update(item.get("service_actions") or [])
    return {name.lower() for name in names}


def make_telemetry() -> dict:
    return copy.deepcopy(BASE)


def apply_control(telemetry: dict, node_id: str, action: str, value) -> None:
    nodes = telemetry.setdefault("nodes", {})
    node = nodes.setdefault(node_id, {})
    node["updated_at"] = time.time()

    if node_id == "smart_panel_01":
        lines = node.get("lines") or []
        if action in ("line_on", "line_off") and isinstance(value, int):
            for line in lines:
                if line.get("line") == value:
                    line["state"] = "ON" if action == "line_on" else "OFF"
                    line["power_w"] = 25.0 if action == "line_on" else 0.0
                    break
        if action == "all_off":
            for line in lines:
                line["state"] = "OFF"
                line["power_w"] = 0.0
        if action == "reset_protection":
            for line in lines:
                if line.get("line") == value:
                    line["protection_state"] = "ARMED"
                    line["alarm"] = False
                    line["alarm_code"] = None
        node["lines_on_count"] = sum(1 for line in lines if line.get("state") == "ON")
        node["total_power_w"] = sum(float(line.get("power_w") or 0) for line in lines)

    elif node_id == "printer_3d_01":
        if action in ("start_job", "resume"):
            node["state"] = "PRINTING"
            node["progress_pct"] = 0.0
            if action == "start_job":
                node["job_file"] = value
        elif action == "home":
            node["state"] = "IDLE"
        elif action == "set_speed":
            node["speed_pct"] = value

    elif node_id == "cnc_01":
        if action == "home":
            node["state"] = "IDLE"
        elif action == "move" and isinstance(value, dict):
            node["x_mm"] = node.get("x_mm", 0) + value.get("x", 0)
            node["y_mm"] = node.get("y_mm", 0) + value.get("y", 0)
            node["z_mm"] = node.get("z_mm", 0) + value.get("z", 0)
        elif action == "set_zero":
            if value in (None, "xyz"):
                node["x_mm"] = node["y_mm"] = node["z_mm"] = 0
            elif value == "xy":
                node["x_mm"] = node["y_mm"] = 0
            elif value == "z":
                node["z_mm"] = 0
        elif action == "probe_z":
            node["z_mm"] = 0
            node["last_probe"] = {"x_mm": 0, "y_mm": 0, "z_mm": 0, "success": True}

    node["last_action"] = action


def append_operator_task(telemetry: dict, text: str) -> dict:
    """Inject an operator message for an end-to-end agent test."""
    message = {
        "id": len(MESSAGES) + 1,
        "time": time.time(),
        "sender": {"role": "operator", "team": None, "name": "Оператор"},
        "to": None,
        "text": text,
    }
    MESSAGES.append(message)
    meeting = telemetry.setdefault("nodes", {}).setdefault("meeting_01", {})
    meeting["message_count"] = len(MESSAGES)
    meeting["last_message_id"] = message["id"]
    meeting["recent_messages"] = MESSAGES[-10:]
    return message


def apply_message(telemetry: dict, text: str, to: str | None, name: str | None) -> dict:
    message = {
        "id": len(MESSAGES) + 1,
        "time": time.time(),
        "sender": {"role": "team", "team": 3, "name": name or "team3-agent"},
        "to": to,
        "text": text,
    }
    MESSAGES.append(message)
    meeting = telemetry.setdefault("nodes", {}).setdefault("meeting_01", {})
    meeting["message_count"] = len(MESSAGES)
    meeting["last_message_id"] = message["id"]
    meeting["recent_messages"] = MESSAGES[-10:]
    return message


def query_result(node_id: str, query: str, params: dict) -> tuple[bool, object, str | None]:
    if node_id == "printer_3d_01" and query == "list_files":
        return (
            True,
            [
                {"name": "gear_m1_z24.gcode", "size_bytes": 132000, "estimated_time_s": 2400},
                {"name": "housing.gcode", "size_bytes": 240000, "estimated_time_s": 4200},
            ],
            None,
        )
    if node_id == "cnc_01" and query == "list_files":
        return (
            True,
            [
                {"name": "front_panel_engrave.nc", "size_bytes": 88000, "valid": True},
                {"name": "pcb_power_board.nc", "size_bytes": 92000, "valid": True},
            ],
            None,
        )
    if node_id == "meeting_01" and query == "read_messages":
        since = int(params.get("since_id") or 0)
        limit = max(1, min(200, int(params.get("limit") or 50)))
        result = [message for message in MESSAGES if message["id"] > since][:limit]
        return True, result, None
    return False, None, f"Запрос '{query}' для {node_id} недоступен."


async def handler(
    ws,
    *,
    key: str,
    revoke_after: int,
    operator_task: str | None = None,
) -> None:
    raw = await ws.recv()
    auth = json.loads(raw)
    if auth.get("type") != "auth" or auth.get("key") != key:
        await ws.send(
            json.dumps(
                {"type": "auth_response", "success": False, "error": "Неверный ключ доступа"}
            )
        )
        await ws.close(code=4401, reason="bad key")
        return

    role = "guest" if auth.get("key") == "guest-key" else "team"
    await ws.send(
        json.dumps(
            {"type": "auth_response", "success": True, "role": role, "team": 3}
        )
    )

    telemetry = make_telemetry()
    if role == "guest":
        telemetry["nodes"] = {"meeting_01": telemetry["nodes"]["meeting_01"]}
    elif operator_task:
        append_operator_task(telemetry, operator_task)

    async def send_telemetry() -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "telemetry",
                    "timestamp": time.time(),
                    "role": role,
                    "data": telemetry,
                }
            )
        )

    await send_telemetry()

    async def ticker() -> None:
        while True:
            await asyncio.sleep(0.4)
            try:
                await send_telemetry()
            except websockets.exceptions.ConnectionClosed:
                return

    tick = asyncio.create_task(ticker())
    seen = 0
    try:
        async for raw in ws:
            message = json.loads(raw)
            seen += 1
            if revoke_after and seen >= revoke_after:
                await ws.send(
                    json.dumps(
                        {
                            "type": "session_revoked",
                            "reason": "Тестовая смена активной команды",
                        }
                    )
                )
                await ws.close(code=4403, reason="revoked")
                return

            kind = message.get("type")
            request_id = message.get("request_id")
            if kind == "ping":
                await ws.send(
                    json.dumps(
                        {"type": "ping_response", "request_id": request_id, "success": True}
                    )
                )
                continue

            if kind == "describe_nodes":
                await ws.send(
                    json.dumps(
                        {
                            "type": "describe_nodes_response",
                            "request_id": request_id,
                            "success": True,
                            "nodes": CATALOG,
                        }
                    )
                )
                continue

            if kind == "control":
                node_id = str(message.get("node_id") or "")
                action = str(message.get("action") or "").lower()
                value = message.get("value")
                item = catalog_for(node_id)
                if item is None:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "control_response",
                                "request_id": request_id,
                                "success": False,
                                "node_id": node_id,
                                "action": action,
                                "error": f"Неизвестный узел: {node_id}",
                            }
                        )
                    )
                    continue
                if action in {str(name).lower() for name in item.get("operator_controls") or []}:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "control_response",
                                "request_id": request_id,
                                "success": False,
                                "node_id": node_id,
                                "action": action,
                                "error": "Действие доступно только Оператору.",
                            }
                        )
                    )
                    continue
                if action not in allowed_actions(item):
                    await ws.send(
                        json.dumps(
                            {
                                "type": "control_response",
                                "request_id": request_id,
                                "success": False,
                                "node_id": node_id,
                                "action": action,
                                "error": (
                                    f"Действие '{action}' для {node_id} недоступно. "
                                    f"Разрешено: {sorted(allowed_actions(item))}"
                                ),
                            }
                        )
                    )
                    continue

                if node_id == "meeting_01" and action in ("post_message", "say"):
                    payload = value if isinstance(value, dict) else {"text": str(value)}
                    created = apply_message(
                        telemetry,
                        str(payload.get("text") or ""),
                        payload.get("to"),
                        payload.get("name"),
                    )
                    await ws.send(
                        json.dumps(
                            {
                                "type": "control_response",
                                "request_id": request_id,
                                "success": True,
                                "node_id": node_id,
                                "action": action,
                                "mode": "virtual",
                                "message": created,
                            }
                        )
                    )
                    await ws.send(
                        json.dumps({"type": "meeting_message", "message": created})
                    )
                else:
                    apply_control(telemetry, node_id, action, value)
                    await ws.send(
                        json.dumps(
                            {
                                "type": "control_response",
                                "request_id": request_id,
                                "success": True,
                                "node_id": node_id,
                                "action": action,
                                "mode": "virtual",
                            }
                        )
                    )
                await send_telemetry()
                continue

            if kind == "query":
                success, result, error = query_result(
                    str(message.get("node_id") or ""),
                    str(message.get("query") or ""),
                    message.get("params") or {},
                )
                payload = {
                    "type": "query_response",
                    "request_id": request_id,
                    "success": success,
                    "node_id": message.get("node_id"),
                    "query": message.get("query"),
                    "mode": "virtual",
                }
                if success:
                    payload["result"] = result
                else:
                    payload["error"] = error
                await ws.send(json.dumps(payload))
                continue

            await ws.send(
                json.dumps(
                    {
                        "type": "response",
                        "request_id": request_id,
                        "success": False,
                        "error": f"Неизвестный тип запроса: {kind}",
                    }
                )
            )
    finally:
        tick.cancel()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--key", default="test-key")
    parser.add_argument("--revoke-after", type=int, default=0)
    parser.add_argument("--operator-task", default=None)
    args = parser.parse_args()

    async def handler_for(ws):
        await handler(
            ws,
            key=args.key,
            revoke_after=args.revoke_after,
            operator_task=args.operator_task,
        )

    server = await websockets.serve(handler_for, "127.0.0.1", args.port)
    port = server.sockets[0].getsockname()[1]
    print(f"READY {port}", flush=True)
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
