"""Async client for the Dome WebSocket API.

Protocol reference: Dome API specification (``01_API.md``).
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json
import logging
import time
from contextlib import suppress
from typing import Any, Callable
from urllib.parse import urlsplit

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)

MISSING = object()
MessageCallback = Callable[[dict[str, Any]], Any]
DEFAULT_MIN_SEND_GAP = 0.35


def retry_policy(timeout: float) -> tuple[int, float]:
    """Return (attempts, base_delay) appropriate for the caller's timeout."""
    if timeout <= 2.0:
        return 1, 0.5
    if timeout <= 5.0:
        return 2, 1.0
    return 4, 2.0


class DomeError(RuntimeError):
    """Base class for dome client errors."""


class DomeConfigError(DomeError):
    """Raised for an invalid client configuration that must not be retried."""


class DomeAuthError(DomeError):
    """Raised when the server rejects the team key."""


class DomeTimeout(DomeError):
    """Raised when the server does not answer in time."""


class DomeConnectionError(DomeError):
    """Raised when the transport is closed or a send fails."""


class DomeSessionRevoked(DomeError):
    """Raised when the server revokes this session (role change)."""


class DomeClient:
    """Small request/response client on top of the dome WebSocket protocol.

    The server pushes telemetry every simulator tick; the client stores the
    latest frame in :attr:`telemetry` and resolves correlated request futures
    from the same reader task.
    """

    def __init__(
        self,
        url: str,
        key: str,
        *,
        name: str = "team3-panel",
        request_timeout: float = 30.0,
        open_timeout: float = 10.0,
        auth_timeout: float = 10.0,
        max_size: int = 50_000_000,
        on_message: MessageCallback | None = None,
        min_send_gap: float = DEFAULT_MIN_SEND_GAP,
    ) -> None:
        self.url = url
        self.key = key
        self.name = name
        self.request_timeout = request_timeout
        self.open_timeout = open_timeout
        self.auth_timeout = auth_timeout
        self.max_size = max_size
        self.on_message = on_message
        self.min_send_gap = max(0.0, float(min_send_gap))

        self.role: str | None = None
        self.team: int | None = None
        self.telemetry: dict[str, Any] | None = None
        self.telemetry_timestamp: float | None = None
        self.revoked_reason: str | None = None
        self.last_telemetry_at: float | None = None
        self._last_telemetry_monotonic: float | None = None
        self._last_send_monotonic: float | None = None

        self._ws: Any = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._ids = itertools.count(1)
        self._telemetry_event = asyncio.Event()
        self._meeting_messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._closed = False
        self._callback_tasks: set[asyncio.Task[Any]] = set()

    # --- connection -------------------------------------------------

    async def connect(self) -> dict[str, Any]:
        if self._ws is not None:
            raise DomeError("already connected")

        parts = urlsplit(self.url)
        if parts.scheme.lower() not in {"ws", "wss"} or not parts.netloc:
            raise DomeConfigError(
                f"invalid WebSocket URL {self.url!r}: expected ws://host:port or wss://host"
            )

        self._closed = False
        self.revoked_reason = None
        try:
            self._ws = await websockets.connect(
                self.url,
                max_size=self.max_size,
                open_timeout=self.open_timeout,
                ping_interval=20,
                ping_timeout=20,
            )
        except (OSError, asyncio.TimeoutError, WebSocketException) as exc:
            self._ws = None
            raise DomeConnectionError(f"connect failed: {exc}") from exc

        try:
            await self._ws.send(json.dumps({"type": "auth", "key": self.key}))
            raw = await asyncio.wait_for(self._ws.recv(), timeout=self.auth_timeout)
        except asyncio.TimeoutError as exc:
            await self._discard_socket()
            raise DomeTimeout("authentication timed out") from exc
        except (ConnectionClosed, OSError, WebSocketException) as exc:
            await self._discard_socket()
            raise DomeConnectionError(f"authentication failed: {exc}") from exc

        try:
            message = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            await self._discard_socket()
            raise DomeConnectionError("authentication response is not valid JSON") from exc

        if message.get("type") != "auth_response" or not message.get("success"):
            await self._discard_socket()
            raise DomeAuthError(str(message.get("error") or "authentication failed"))

        self.role = message.get("role")
        self.team = message.get("team")
        self._closed = False
        self._reader_task = asyncio.create_task(self._reader(), name="dome-reader")
        return message

    async def connect_with_retry(
        self,
        attempts: int = 4,
        delay: float = 2.0,
        max_delay: float = 8.0,
    ) -> dict[str, Any]:
        """Connect, retrying only transient transport failures.

        The public dome endpoint occasionally drops the opening handshake when
        several short-lived CLI processes connect back-to-back. Authentication
        failures are never retried.
        """
        last: Exception | None = None
        for attempt in range(1, max(1, attempts) + 1):
            try:
                return await self.connect()
            except DomeAuthError:
                raise
            except (DomeConnectionError, DomeTimeout, OSError, asyncio.TimeoutError) as exc:
                last = exc
                if attempt >= attempts:
                    break
                wait = min(delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    "connect attempt %s/%s failed: %s; retrying in %.1fs",
                    attempt,
                    attempts,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
        assert last is not None
        raise last

    async def reconnect(self) -> dict[str, Any]:
        """Drop the current socket, clear session state and connect again."""
        await self.close()
        self._pending.clear()
        self.revoked_reason = None
        self.role = None
        self.team = None
        self.telemetry = None
        self.telemetry_timestamp = None
        self.last_telemetry_at = None
        self._last_telemetry_monotonic = None
        self._last_send_monotonic = None
        self._closed = False
        self._telemetry_event.clear()
        while not self._meeting_messages.empty():
            with suppress(asyncio.QueueEmpty):
                self._meeting_messages.get_nowait()
        return await self.connect()

    async def close(self) -> None:
        """Close the socket. Safe to call more than once and never raises."""
        self._closed = True
        reader = self._reader_task
        self._reader_task = None
        if reader is not None:
            reader.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await reader

        for task in list(self._callback_tasks):
            task.cancel()
        if self._callback_tasks:
            with suppress(asyncio.CancelledError, Exception):
                await asyncio.gather(*self._callback_tasks, return_exceptions=True)
            self._callback_tasks.clear()

        self._fail_pending(DomeConnectionError("connection closed"))
        ws = self._ws
        self._ws = None
        if ws is not None:
            with suppress(Exception):
                await ws.close()

    async def __aenter__(self) -> "DomeClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def _discard_socket(self) -> None:
        ws = self._ws
        self._ws = None
        self._closed = True
        if ws is not None:
            with suppress(Exception):
                await ws.close()

    # --- reader -----------------------------------------------------

    async def _reader(self) -> None:
        assert self._ws is not None
        ws = self._ws
        try:
            async for raw in ws:
                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("ignoring malformed message: %s", exc)
                    continue

                self._notify(message)
                kind = message.get("type")
                if kind == "telemetry":
                    data = message.get("data") or {}
                    self.telemetry = data
                    timestamp = message.get("timestamp")
                    self.telemetry_timestamp = (
                        float(timestamp) if isinstance(timestamp, (int, float)) else None
                    )
                    self.last_telemetry_at = time.time()
                    self._last_telemetry_monotonic = time.monotonic()
                    self._telemetry_event.set()
                elif kind == "meeting_message":
                    self._meeting_messages.put_nowait(message.get("message") or {})
                elif kind == "session_revoked":
                    self.revoked_reason = str(message.get("reason") or "session revoked")
                    self._closed = True
                    self._telemetry_event.set()
                    self._fail_pending(DomeSessionRevoked(self.revoked_reason))
                    break
                else:
                    request_id = message.get("request_id")
                    if request_id is not None:
                        future = self._pending.pop(str(request_id), None)
                        if future is not None and not future.done():
                            future.set_result(message)
        except asyncio.CancelledError:
            raise
        except ConnectionClosed as exc:
            self._closed = True
            self._fail_pending(DomeConnectionError(f"connection closed: {exc}"))
        except Exception as exc:
            self._closed = True
            self._fail_pending(DomeError(f"reader failed: {exc}"))
        finally:
            self._closed = True
            self._telemetry_event.set()

    def _notify(self, message: dict[str, Any]) -> None:
        callback = self.on_message
        if callback is None:
            return
        try:
            result = callback(message)
        except Exception:
            logger.exception("message callback failed")
            return
        if inspect.isawaitable(result):
            task = asyncio.create_task(self._await_callback(result))
            self._callback_tasks.add(task)
            task.add_done_callback(self._callback_tasks.discard)

    async def _await_callback(self, awaitable: Any) -> None:
        try:
            await awaitable
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("async message callback failed")

    def _fail_pending(self, exc: Exception) -> None:
        pending = list(self._pending.values())
        self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(exc)

    # --- requests ---------------------------------------------------

    async def request(self, **payload: Any) -> dict[str, Any]:
        message = {**payload, "request_id": str(next(self._ids))}
        return await self._send_request(message)

    async def request_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send *payload* as-is; only add ``request_id`` when it is absent."""
        if not isinstance(payload, dict):
            raise DomeError("request payload must be a JSON object")
        message = dict(payload)
        if "request_id" not in message:
            message["request_id"] = str(next(self._ids))
        return await self._send_request(message)

    async def _send_request(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.revoked_reason:
            raise DomeSessionRevoked(self.revoked_reason)
        if self._ws is None:
            raise DomeConnectionError("not connected")
        if self._closed:
            raise DomeConnectionError("connection closed")

        request_id = message.get("request_id")
        key = str(request_id)
        if key in self._pending:
            raise DomeError(f"request_id {key!r} is already pending")

        try:
            encoded = json.dumps(message, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise DomeError(f"invalid request payload: {exc}") from exc

        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[key] = future
        try:
            if self.min_send_gap > 0:
                now = asyncio.get_running_loop().time()
                if self._last_send_monotonic is not None:
                    wait = self.min_send_gap - (now - self._last_send_monotonic)
                    if wait > 0:
                        await asyncio.sleep(wait)
            try:
                await self._ws.send(encoded)
                self._last_send_monotonic = asyncio.get_running_loop().time()
            except (ConnectionClosed, OSError, WebSocketException) as exc:
                self._closed = True
                error = DomeConnectionError(f"send failed: {exc}")
                self._fail_pending(error)
                raise error from exc
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError as exc:
            raise DomeTimeout(f"{message.get('type', 'request')} timed out") from exc
        finally:
            self._pending.pop(key, None)

    async def control(self, node_id: str, action: str, value: Any = MISSING) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "control", "node_id": node_id, "action": action}
        if value is not MISSING:
            payload["value"] = value
        return await self.request(**payload)

    async def query(
        self,
        node_id: str,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.request(type="query", node_id=node_id, query=query, params=params or {})

    async def describe_nodes(self) -> list[dict[str, Any]]:
        response = await self.request(type="describe_nodes")
        nodes = response.get("nodes")
        if not isinstance(nodes, list):
            raise DomeError(f"describe_nodes failed: {response.get('error') or response}")
        return nodes

    async def ping(self) -> dict[str, Any]:
        return await self.request(type="ping")

    # --- chat -------------------------------------------------------

    async def post_message(
        self,
        text: str,
        *,
        to: str | None = "operator",
        name: str | None = None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {"text": text}
        if to:
            value["to"] = to
        value["name"] = name or self.name
        return await self.control("meeting_01", "post_message", value)

    async def read_messages(
        self,
        *,
        since_id: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if since_id is not None:
            params["since_id"] = since_id
        response = await self.query("meeting_01", "read_messages", params)
        result = response.get("result")
        if not isinstance(result, list):
            raise DomeError(f"read_messages failed: {response.get('error') or response}")
        return result

    # --- telemetry --------------------------------------------------

    @property
    def telemetry_age(self) -> float | None:
        """Seconds since the last telemetry frame was received."""
        if self._last_telemetry_monotonic is None:
            return None
        return max(0.0, time.monotonic() - self._last_telemetry_monotonic)

    async def wait_for_telemetry(
        self,
        timeout: float = 10.0,
        *,
        next_frame: bool = False,
        max_age: float | None = None,
    ) -> dict[str, Any]:
        """Return telemetry, optionally requiring it to be no older than *max_age*.

        With ``next_frame=True`` the current cached frame is ignored. With
        ``max_age`` the call waits for a newer frame when the cached one is
        stale, instead of silently returning stale data.
        """
        if self.revoked_reason:
            raise DomeSessionRevoked(self.revoked_reason)

        if not next_frame and self.telemetry is not None:
            age = self.telemetry_age
            if max_age is None or (age is not None and age <= max_age):
                return self.telemetry
        if self._closed and self.telemetry is not None:
            raise DomeConnectionError("connection closed before fresh telemetry")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise DomeTimeout(
                    "telemetry is stale" if self.telemetry is not None else "telemetry did not arrive"
                )

            self._telemetry_event.clear()
            try:
                await asyncio.wait_for(self._telemetry_event.wait(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise DomeTimeout(
                    "telemetry is stale" if self.telemetry is not None else "telemetry did not arrive"
                ) from exc

            if self.revoked_reason:
                raise DomeSessionRevoked(self.revoked_reason)
            if self.telemetry is None:
                if self._closed:
                    raise DomeConnectionError("connection closed before telemetry")
                continue

            age = self.telemetry_age
            if max_age is None or (age is not None and age <= max_age):
                return self.telemetry
            if self._closed:
                raise DomeConnectionError("connection closed before fresh telemetry")

    async def next_meeting_message(self, timeout: float = 30.0) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(self._meeting_messages.get(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise DomeTimeout("meeting message did not arrive") from exc

    @property
    def nodes(self) -> dict[str, Any]:
        return (self.telemetry or {}).get("nodes") or {}

    def node(self, node_id: str) -> dict[str, Any]:
        return self.nodes.get(node_id) or {}
