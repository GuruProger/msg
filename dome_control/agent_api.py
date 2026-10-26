"""Agent-facing async facade and JSON tool schemas for the Dome API.

The CLI remains the human/administrative surface.  This module is the small,
explicit contract an autonomous agent can call:

    async with DomeAgentAPI.from_env() as api:
        devices = await api.list_devices()
        telemetry = await api.get_telemetry("smart_panel_01")
        result = await api.control_device("smart_panel_01", "line_on", 3)

Every method returns JSON-serializable data.  ``control_device`` validates the
action against the live ``describe_nodes`` catalog, refuses operator-only
actions, and verifies the result against the next telemetry frame by default.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any

from .client import (
    MISSING,
    DomeClient,
    DomeError,
    retry_policy,
)
from .domain import (
    DomeValidationError,
    find_node,
    filter_catalog,
    resolve_control_action,
    resolve_query_name,
    telemetry_diff,
    telemetry_node,
)
from .panel import load_dotenv, read_team_key


TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "name": "dome_get_telemetry",
        "description": "Return fresh dome telemetry, optionally for one node.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": ["string", "null"], "description": "Node id or code_name."},
                "max_age": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Maximum telemetry age in seconds before waiting for a new frame.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "dome_get_status",
        "description": "Return the full dome status: nodes and game_time.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_age": {"type": "number", "minimum": 0},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "dome_list_devices",
        "description": "List devices from the live describe_nodes catalog.",
        "parameters": {
            "type": "object",
            "properties": {
                "system": {"type": ["string", "null"]},
                "search": {"type": ["string", "null"]},
                "refresh": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "dome_get_device",
        "description": "Return one device's catalog entry and latest telemetry.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "max_age": {"type": "number", "minimum": 0},
            },
            "required": ["node_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "dome_control_device",
        "description": (
            "Validate and send a control action. Operator-only actions are rejected. "
            "Use dry_run=true to inspect the request without sending it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "action": {"type": "string"},
                "value": {"description": "Optional JSON value for the action."},
                "dry_run": {"type": "boolean"},
                "force": {"type": "boolean"},
                "verify": {"type": "boolean"},
            },
            "required": ["node_id", "action"],
            "additionalProperties": False,
        },
    },
    {
        "name": "dome_query_device",
        "description": "Run a validated query against a device.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "query": {"type": "string"},
                "params": {"type": ["object", "null"]},
                "force": {"type": "boolean"},
            },
            "required": ["node_id", "query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "dome_read_messages",
        "description": "Read meeting_01 messages.",
        "parameters": {
            "type": "object",
            "properties": {
                "since_id": {"type": ["integer", "null"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "dome_post_message",
        "description": "Post a message to meeting_01.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "to": {"type": ["string", "null"]},
                "name": {"type": ["string", "null"]},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "dome_whoami",
        "description": "Return the authenticated role, team and endpoint.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "dome_ping",
        "description": "Send an application-level ping and return the response.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
)


def tool_specs() -> list[dict[str, Any]]:
    """Return a fresh, JSON-serializable copy of the tool schemas."""
    return copy.deepcopy(list(TOOL_SCHEMAS))


class DomeAgentAPI:
    """High-level async API intended for an autonomous agent.

    ``client`` can be injected for tests or for sharing an already connected
    ``DomeClient``.  When it is omitted, the facade owns and closes it.
    """

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        *,
        client: DomeClient | None = None,
        name: str = "team3-agent",
        timeout: float = 10.0,
        min_send_gap: float = 0.35,
        catalog_ttl: float = 60.0,
    ) -> None:
        if client is None:
            if not url:
                raise ValueError("url is required when client is not provided")
            if not key:
                raise ValueError("key is required when client is not provided")
            client = DomeClient(
                url,
                key,
                name=name,
                request_timeout=timeout,
                open_timeout=min(timeout, 10.0),
                auth_timeout=min(timeout, 10.0),
                min_send_gap=min_send_gap,
            )
            self._owns_client = True
            self._connected = False
        else:
            self._owns_client = False
            self._connected = True
        self.client = client
        self.timeout = timeout
        self.catalog_ttl = max(0.0, float(catalog_ttl))
        self._catalog: list[dict[str, Any]] | None = None
        self._catalog_at = 0.0

    @classmethod
    def from_env(
        cls,
        *,
        env_file: str | None = ".env",
        url: str | None = None,
        key: str | None = None,
        key_file: str | None = None,
        name: str | None = None,
        timeout: float = 10.0,
        min_send_gap: float = 0.35,
        catalog_ttl: float = 60.0,
    ) -> "DomeAgentAPI":
        load_dotenv(env_file)
        url = (url or os.getenv("DOME_URL") or "").strip()
        key = (key or os.getenv("DOME_KEY") or "").strip()
        key_file = (key_file or os.getenv("DOME_KEY_FILE") or "").strip()
        if not key and key_file:
            key = read_team_key(key_file) or ""
        if not url:
            raise ValueError("Не задан DOME_URL (аргумент или переменная окружения).")
        if not key:
            raise ValueError("Не задан DOME_KEY (аргумент, DOME_KEY или key_file).")
        resolved_name = name or os.getenv("AGENT_NAME") or "team3-agent"
        return cls(
            url,
            key,
            name=resolved_name,
            timeout=timeout,
            min_send_gap=min_send_gap,
            catalog_ttl=catalog_ttl,
        )

    async def __aenter__(self) -> "DomeAgentAPI":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def connect(self) -> dict[str, Any]:
        if self._connected:
            return {"success": True, "already_connected": True}
        if not isinstance(self.client, DomeClient):
            self._connected = True
            return {"success": True, "already_connected": True}
        auth = await self.client.connect_with_retry(*retry_policy(self.timeout))
        self._connected = True
        self._invalidate_catalog()
        return auth

    async def close(self) -> None:
        if self._owns_client:
            await self.client.close()
        self._connected = False
        self._invalidate_catalog()

    async def reconnect(self) -> dict[str, Any]:
        if isinstance(self.client, DomeClient):
            auth = await self.client.reconnect()
        else:
            auth = {"success": True}
        self._connected = True
        self._invalidate_catalog()
        return auth

    def _invalidate_catalog(self) -> None:
        self._catalog = None
        self._catalog_at = 0.0

    async def _ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    async def catalog(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        await self._ensure_connected()
        fresh = self._catalog is not None and (time.monotonic() - self._catalog_at) <= self.catalog_ttl
        if refresh or not fresh:
            self._catalog = await self.client.describe_nodes()
            self._catalog_at = time.monotonic()
        return list(self._catalog or [])

    async def list_devices(
        self,
        *,
        system: str | None = None,
        search: str | None = None,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        catalog = await self.catalog(refresh=refresh)
        return filter_catalog(catalog, system=system, search=search)

    async def get_telemetry(
        self,
        node_id: str | None = None,
        *,
        max_age: float = 5.0,
        next_frame: bool = False,
    ) -> dict[str, Any]:
        await self._ensure_connected()
        data = await self.client.wait_for_telemetry(
            timeout=self.timeout,
            max_age=max_age,
            next_frame=next_frame,
        )
        age = self.client.telemetry_age
        meta: dict[str, Any] = {
            "age_s": round(age, 3) if age is not None else None,
            "server_timestamp": self.client.telemetry_timestamp,
            "game_time": data.get("game_time"),
        }
        if node_id is None:
            return {**meta, "nodes": data.get("nodes") or {}}

        found = telemetry_node(data, node_id)
        if found is None:
            raise DomeValidationError(f"Узел {node_id!r} отсутствует в телеметрии")
        resolved_id, node = found
        return {**meta, "node_id": resolved_id, "telemetry": node}

    async def get_status(self, *, max_age: float = 5.0) -> dict[str, Any]:
        return await self.get_telemetry(max_age=max_age)

    async def get_device(self, node_id: str, *, max_age: float = 5.0) -> dict[str, Any]:
        catalog = await self.catalog()
        node = find_node(catalog, node_id)
        if node is None:
            raise DomeValidationError(f"Узел {node_id!r} не найден в каталоге")
        resolved_id = str(node.get("node_id"))
        telemetry = await self.get_telemetry(resolved_id, max_age=max_age)
        return {
            "catalog": node,
            "node_id": resolved_id,
            "telemetry": telemetry.get("telemetry"),
            "age_s": telemetry.get("age_s"),
            "server_timestamp": telemetry.get("server_timestamp"),
        }

    async def control_device(
        self,
        node_id: str,
        action: str,
        value: Any = MISSING,
        *,
        dry_run: bool = False,
        force: bool = False,
        verify: bool = True,
        max_age: float = 5.0,
        allow_without_before: bool = False,
    ) -> dict[str, Any]:
        catalog = await self.catalog()
        node = find_node(catalog, node_id)
        if node is None:
            raise DomeValidationError(f"Узел {node_id!r} не найден в каталоге")
        resolved_id = str(node.get("node_id"))
        resolved_action = resolve_control_action(node, action, force=force)

        payload: dict[str, Any] = {
            "type": "control",
            "node_id": resolved_id,
            "action": resolved_action,
        }
        if value is not MISSING:
            payload["value"] = value

        if dry_run:
            return {"dry_run": True, "request": payload}

        before: dict[str, Any] | None = None
        before_error: str | None = None
        try:
            before_data = await self.client.wait_for_telemetry(
                timeout=min(self.timeout, 10.0),
                max_age=max_age,
            )
        except DomeError as exc:
            before_error = str(exc)
        else:
            found = telemetry_node(before_data, resolved_id)
            before = found[1] if found else None
            if before is None:
                before_error = f"node {resolved_id!r} absent from fresh telemetry"

        if before is None and not allow_without_before:
            return {
                "success": False,
                "accepted": False,
                "blocked": True,
                "unknown": False,
                "sent": False,
                "error": f"fresh before-telemetry unavailable: {before_error}",
                "node_id": resolved_id,
                "action": resolved_action,
                "request": payload,
                "response": None,
                "before": None,
                "after": None,
                "diff": None,
                "before_error": before_error,
                "verify_error": None,
            }

        response = await self.client.control(resolved_id, resolved_action, value)
        success = bool(response.get("success"))

        after: dict[str, Any] | None = None
        diff: dict[str, Any] | None = None
        verify_error: str | None = None
        if success and verify:
            try:
                after_data = await self.client.wait_for_telemetry(
                    timeout=min(self.timeout, 10.0),
                    next_frame=True,
                )
            except DomeError as exc:
                verify_error = str(exc)
            else:
                found = telemetry_node(after_data, resolved_id)
                after = found[1] if found else None
                diff = telemetry_diff(before, after)

        return {
            "success": success,
            "accepted": success,
            "blocked": False,
            "unknown": before is None,
            "sent": True,
            "node_id": resolved_id,
            "action": resolved_action,
            "request": payload,
            "response": response,
            "before": before,
            "after": after,
            "diff": diff,
            "before_error": before_error,
            "verify_error": verify_error,
        }

    async def query_device(
        self,
        node_id: str,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        catalog = await self.catalog()
        node = find_node(catalog, node_id)
        if node is None:
            raise DomeValidationError(f"Узел {node_id!r} не найден в каталоге")
        resolved_id = str(node.get("node_id"))
        resolved_query = resolve_query_name(node, query, force=force)
        return await self.client.query(resolved_id, resolved_query, params or {})

    async def read_messages(
        self,
        *,
        since_id: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        await self._ensure_connected()
        return await self.client.read_messages(since_id=since_id, limit=limit)

    async def post_message(
        self,
        text: str,
        *,
        to: str | None = "operator",
        name: str | None = None,
    ) -> dict[str, Any]:
        await self._ensure_connected()
        return await self.client.post_message(text, to=to, name=name)

    async def whoami(self) -> dict[str, Any]:
        await self._ensure_connected()
        return {
            "url": self.client.url,
            "role": self.client.role,
            "team": self.client.team,
            "name": self.client.name,
        }

    async def ping(self) -> dict[str, Any]:
        await self._ensure_connected()
        return await self.client.ping()

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Dispatch a tool by name; useful for generic function-calling loops."""
        args = arguments or {}
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be an object")
        if name == "dome_get_telemetry":
            return await self.get_telemetry(
                args.get("node_id"),
                max_age=float(args.get("max_age", 5.0)),
            )
        if name == "dome_get_status":
            return await self.get_status(max_age=float(args.get("max_age", 5.0)))
        if name == "dome_list_devices":
            return await self.list_devices(
                system=args.get("system"),
                search=args.get("search"),
                refresh=bool(args.get("refresh", False)),
            )
        if name == "dome_get_device":
            return await self.get_device(
                str(args["node_id"]),
                max_age=float(args.get("max_age", 5.0)),
            )
        if name == "dome_control_device":
            value = args.get("value", MISSING)
            return await self.control_device(
                str(args["node_id"]),
                str(args["action"]),
                value,
                dry_run=bool(args.get("dry_run", False)),
                force=bool(args.get("force", False)),
                verify=bool(args.get("verify", True)),
            )
        if name == "dome_query_device":
            return await self.query_device(
                str(args["node_id"]),
                str(args["query"]),
                args.get("params"),
                force=bool(args.get("force", False)),
            )
        if name == "dome_read_messages":
            return await self.read_messages(
                since_id=args.get("since_id"),
                limit=int(args.get("limit", 50)),
            )
        if name == "dome_post_message":
            return await self.post_message(
                str(args["text"]),
                to=args.get("to", "operator"),
                name=args.get("name"),
            )
        if name == "dome_whoami":
            return await self.whoami()
        if name == "dome_ping":
            return await self.ping()
        raise ValueError(f"unknown tool: {name}")
