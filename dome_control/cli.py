"""Production command line interface for the Dome WebSocket API.

The CLI is deliberately a management tool: it can inspect telemetry, validate
commands against ``describe_nodes``, send individual requests and keep an audit
log.  It is not an autonomous agent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from .client import (
    MISSING,
    DomeAuthError,
    DomeClient,
    DomeConfigError,
    DomeConnectionError,
    DomeError,
    DomeSessionRevoked,
    DomeTimeout,
    retry_policy,
)
from .panel import (
    DEFAULT_NAME,
    RULE,
    _client,
    format_chat_messages,
    format_describe,
    format_nodes,
    format_status,
    load_demo_telemetry,
    parse_value,
    resolve_settings,
)
from .domain import (
    DomeValidationError,
    filter_catalog as _filter_catalog,
    filter_nodes as _filter_nodes,
    find_node as _find_node,
    resolve_control_action,
    resolve_query_name,
    telemetry_diff as _telemetry_diff,
    telemetry_node as _telemetry_node,
)

logger = logging.getLogger("dome_control")

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2
EXIT_NETWORK = 3
EXIT_INTERRUPTED = 130

CommandHandler = Callable[["argparse.Namespace", "AuditLogger"], Coroutine[Any, Any, int]]


class CliError(Exception):
    """An expected command failure with an exit code."""

    def __init__(self, message: str, code: int = EXIT_FAILURE) -> None:
        super().__init__(message)
        self.code = code


# --- audit log ---------------------------------------------------------------


class AuditLogger:
    """Append-only JSON Lines audit log.

    The auth key is never accepted as a field; a defensive filter also drops
    anything that looks like a secret.
    """

    _SECRET_FIELDS = {
        "key",
        "dome_key",
        "api_key",
        "deepseek_api_key",
        "password",
        "token",
        "secret",
    }

    def __init__(self, path: str | None, *, url: str | None = None, log_telemetry: bool = False) -> None:
        self.path = path
        self.url = url
        self.role: str | None = None
        self.team: int | None = None
        self.log_telemetry = log_telemetry
        self._handle: Any = None
        self._seq = 0
        if not path:
            return
        try:
            candidate = Path(path)
            if candidate.parent and str(candidate.parent) not in ("", "."):
                candidate.parent.mkdir(parents=True, exist_ok=True)
            self._handle = candidate.open("a", encoding="utf-8", buffering=1)
        except OSError as exc:
            raise CliError(f"cannot open audit log {path!r}: {exc}", EXIT_CONFIG) from exc

    def bind(self, client: DomeClient) -> None:
        self.url = client.url
        self.role = client.role
        self.team = client.team

    def write(self, event: str, **fields: Any) -> None:
        if self._handle is None:
            return
        self._seq += 1
        record: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "seq": self._seq,
            "event": event,
        }
        if self.url:
            record["url"] = self.url
        if self.role is not None:
            record["role"] = self.role
        if self.team is not None:
            record["team"] = self.team
        for name, value in fields.items():
            if value is None or name.casefold() in self._SECRET_FIELDS:
                continue
            record[name] = value
        try:
            self._handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._handle.flush()
        except OSError as exc:
            logger.warning("cannot write audit log: %s", exc)

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.close()
        except OSError as exc:
            logger.warning("cannot close audit log: %s", exc)
        finally:
            self._handle = None


# --- generic helpers ---------------------------------------------------------


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


def _json_dump(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, default=str)


def _emit_json(value: Any) -> None:
    print(_json_dump(value))


def _warn(args: argparse.Namespace, message: str) -> None:
    if not getattr(args, "quiet", False):
        print(message, file=sys.stderr)


def _url_for_audit(args: argparse.Namespace) -> str | None:
    value = str(getattr(args, "url", None) or os.getenv("DOME_URL") or "").strip()
    return value or None


def _settings(args: argparse.Namespace) -> tuple[str, str, str, float]:
    try:
        return resolve_settings(args)
    except SystemExit as exc:
        message = exc.code if isinstance(exc.code, str) else "configuration error"
        raise CliError(str(message), EXIT_CONFIG) from None


def _make_client(
    args: argparse.Namespace,
    *,
    on_message: Any = None,
) -> DomeClient:
    url, key, name, timeout = _settings(args)
    return DomeClient(
        url,
        key,
        name=name,
        request_timeout=timeout,
        open_timeout=min(timeout, 10.0),
        auth_timeout=min(timeout, 10.0),
        on_message=on_message,
    )


def _load_demo(args: argparse.Namespace) -> dict[str, Any]:
    try:
        return load_demo_telemetry(args.demo)
    except SystemExit as exc:
        message = exc.code if isinstance(exc.code, str) else "demo telemetry error"
        raise CliError(str(message), EXIT_CONFIG) from None


def _resolve_control_action(
    args: argparse.Namespace,
    node: dict[str, Any],
    action: str,
    *,
    force: bool,
) -> str:
    try:
        resolved = resolve_control_action(node, action, force=force)
    except DomeValidationError as exc:
        raise CliError(str(exc), EXIT_FAILURE) from None
    if force and resolved == action and action.casefold() not in {
        value.casefold()
        for value in [*(node.get("controls") or []), *(node.get("service_actions") or [])]
    }:
        _warn(args, f"Внимание: действие {action!r} отсутствует в каталоге; отправляю из-за --force.")
    return resolved


def _resolve_query_name(node: dict[str, Any], query: str, *, force: bool) -> str:
    try:
        return resolve_query_name(node, query, force=force)
    except DomeValidationError as exc:
        raise CliError(str(exc), EXIT_FAILURE) from None


async def _confirm(args: argparse.Namespace, payload: dict[str, Any]) -> bool:
    if getattr(args, "yes", False):
        return True
    if not sys.stdin.isatty():
        print(
            "Отправка управляющей команды в неинтерактивном режиме требует --yes.",
            file=sys.stderr,
        )
        return False
    print("Запрос:", _json_dump(payload, indent=None))
    answer = await asyncio.to_thread(input, "Отправить? [y/N] ")
    return answer.strip().casefold() in {"y", "yes", "д", "да"}


def _accepted_text(response: dict[str, Any], *, verified: bool) -> str:
    mode = response.get("mode", "—")
    suffix = (
        "команда принята сервером; проверьте телеметрию по diff ниже."
        if verified
        else "команда принята сервером (accepted); проверка телеметрии отключена."
    )
    return f"OK · mode={mode} · {suffix}"


def _response_error(response: dict[str, Any]) -> str:
    return str(response.get("error") or "server returned success=false")


# --- commands ----------------------------------------------------------------


async def cmd_whoami(args: argparse.Namespace, audit: AuditLogger) -> int:
    start = time.perf_counter()
    async with _client(args) as client:
        audit.bind(client)
        latency = _ms_since(start)
        info = {
            "url": client.url,
            "role": client.role,
            "team": client.team,
            "name": client.name,
        }
        audit.write("whoami", success=True, latency_ms=latency, role=client.role, team=client.team)
        if args.json:
            _emit_json(info)
        else:
            print(f"URL:   {client.url}")
            print(f"role:  {client.role}")
            print(f"team:  {client.team}")
            print(f"name:  {client.name}")
        return EXIT_OK


async def cmd_ping(args: argparse.Namespace, audit: AuditLogger) -> int:
    async with _client(args) as client:
        audit.bind(client)
        start = time.perf_counter()
        response = await client.ping()
        latency = _ms_since(start)
        success = bool(response.get("success"))
        audit.write(
            "ping",
            request_type="ping",
            request_id=response.get("request_id"),
            success=success,
            error=None if success else _response_error(response),
            latency_ms=latency,
        )
        if args.json:
            _emit_json(
                {
                    "success": success,
                    "latency_ms": latency,
                    "role": client.role,
                    "team": client.team,
                    "response": response,
                }
            )
        elif success:
            print(f"PONG · {latency:.1f} ms")
        else:
            print(f"PING FAILED: {_response_error(response)}", file=sys.stderr)
        return EXIT_OK if success else EXIT_FAILURE


async def cmd_catalog(args: argparse.Namespace, audit: AuditLogger) -> int:
    async with _client(args) as client:
        audit.bind(client)
        start = time.perf_counter()
        catalog = await client.describe_nodes()
        latency = _ms_since(start)
        filtered = _filter_catalog(catalog, system=args.system, search=args.search)
        audit.write(
            "catalog",
            request_type="describe_nodes",
            success=True,
            latency_ms=latency,
            nodes=len(filtered),
            system=args.system,
            search=args.search,
        )
        if args.json:
            _emit_json(filtered)
        else:
            print(format_describe(filtered))
        return EXIT_OK


async def cmd_nodes(args: argparse.Namespace, audit: AuditLogger) -> int:
    if args.demo:
        data = _load_demo(args)
        catalog: list[dict[str, Any]] = []
        role: str | None = "demo"
        team: int | None = 3
        latency = 0.0
    else:
        async with _client(args) as client:
            audit.bind(client)
            start = time.perf_counter()
            catalog = await client.describe_nodes()
            data = await client.wait_for_telemetry(
                timeout=args.timeout,
                max_age=min(args.timeout, 5.0),
            )
            latency = _ms_since(start)
            role, team = client.role, client.team

    filtered_catalog = _filter_catalog(catalog, system=args.system, search=args.search)
    nodes = _filter_nodes(
        data.get("nodes") or {},
        catalog,
        system=args.system,
        search=args.search,
    )
    audit.write(
        "nodes",
        request_type="describe_nodes+telemetry",
        success=True,
        latency_ms=latency,
        role=role,
        team=team,
        nodes=len(nodes),
        system=args.system,
        search=args.search,
    )
    if args.json:
        _emit_json({"catalog": filtered_catalog, "nodes": nodes})
    else:
        print(format_nodes(filtered_catalog, {"nodes": nodes}))
    return EXIT_OK


async def cmd_status(args: argparse.Namespace, audit: AuditLogger) -> int:
    if args.demo:
        data = _load_demo(args)
        role: str | None = "demo"
        team: int | None = 3
        latency = 0.0
    else:
        async with _client(args) as client:
            audit.bind(client)
            start = time.perf_counter()
            data = await client.wait_for_telemetry(
                timeout=args.timeout,
                max_age=min(args.timeout, 5.0),
            )
            latency = _ms_since(start)
            role, team = client.role, client.team

    audit.write("status", success=True, latency_ms=latency, role=role, team=team)
    if args.json:
        _emit_json(data)
    else:
        print(format_status(data, role=role, team=team))
        if role == "guest":
            print(
                "\nРоль guest: доступен только meeting_01, телеметрия купола закрыта.",
                file=sys.stderr,
            )
    return EXIT_OK


async def cmd_telemetry(args: argparse.Namespace, audit: AuditLogger) -> int:
    if args.demo:
        data = _load_demo(args)
        latency = 0.0
    else:
        async with _client(args) as client:
            audit.bind(client)
            start = time.perf_counter()
            data = await client.wait_for_telemetry(
                timeout=args.timeout,
                max_age=min(args.timeout, 5.0),
            )
            latency = _ms_since(start)

    selected: Any = data
    node_id: str | None = None
    if args.node:
        found = _telemetry_node(data, args.node)
        if found is None:
            raise CliError(f"Узел {args.node!r} отсутствует в телеметрии", EXIT_FAILURE)
        node_id, selected = found

    fields: dict[str, Any] = {
        "request_type": "telemetry",
        "success": True,
        "latency_ms": latency,
        "node": node_id,
    }
    if args.log_telemetry:
        fields["telemetry"] = selected
    audit.write("telemetry", **fields)

    if args.json:
        _emit_json(selected)
    else:
        print(_json_dump(selected))
    return EXIT_OK


async def _watch_frame(
    args: argparse.Namespace,
    data: dict[str, Any],
    *,
    role: str | None,
    team: int | None,
) -> None:
    payload: Any = data
    if args.node:
        found = _telemetry_node(data, args.node)
        if found is None:
            raise CliError(f"Узел {args.node!r} отсутствует в телеметрии", EXIT_FAILURE)
        _, payload = found

    if args.raw:
        print(_json_dump(payload, indent=None))
    elif args.node:
        print(_json_dump(payload))
    else:
        print(format_status(data, role=role, team=team))
        print(RULE)
    sys.stdout.flush()


async def cmd_watch(args: argparse.Namespace, audit: AuditLogger) -> int:
    interval = max(float(args.interval), 0.1)

    if args.demo:
        data = _load_demo(args)
        await _watch_frame(args, data, role="demo", team=3)
        return EXIT_OK

    client = _make_client(args)
    try:
        await client.connect_with_retry(*retry_policy(args.timeout))
        audit.bind(client)
        first = True
        last_print = 0.0
        while True:
            try:
                if first:
                    data = await client.wait_for_telemetry(timeout=args.timeout)
                    first = False
                else:
                    data = await client.wait_for_telemetry(timeout=max(interval + 15.0, 15.0), next_frame=True)
                now = time.monotonic()
                wait = interval - (now - last_print)
                if wait > 0:
                    await asyncio.sleep(wait)
                await _watch_frame(args, data, role=client.role, team=client.team)
                last_print = time.monotonic()
                audit.write(
                    "watch_frame",
                    request_type="telemetry",
                    success=True,
                    node=args.node,
                    telemetry=data if args.log_telemetry else None,
                )
            except (DomeTimeout, DomeConnectionError, DomeSessionRevoked, DomeError, OSError) as exc:
                if not args.reconnect:
                    raise
                audit.write("reconnect", success=False, error=str(exc))
                _warn(args, f"Соединение потеряно: {exc}. Переподключаюсь...")
                await asyncio.sleep(max(interval, 1.0))
                await client.reconnect()
                audit.bind(client)
                first = True
    finally:
        await client.close()


async def _ensure_catalog(client: DomeClient) -> list[dict[str, Any]]:
    return await client.describe_nodes()


async def cmd_control(args: argparse.Namespace, audit: AuditLogger) -> int:
    value = parse_value(args.value)
    async with _client(args) as client:
        audit.bind(client)
        start = time.perf_counter()
        catalog = await _ensure_catalog(client)
        catalog_latency = _ms_since(start)

        node = _find_node(catalog, args.node_id)
        if node is None:
            available = ", ".join(str(item.get("node_id")) for item in catalog if item.get("node_id"))
            raise CliError(
                f"Узел {args.node_id!r} не найден в каталоге. Доступные: {available or 'нет'}",
                EXIT_FAILURE,
            )
        node_id = str(node.get("node_id"))
        action = _resolve_control_action(args, node, args.action, force=args.force)

        payload: dict[str, Any] = {"type": "control", "node_id": node_id, "action": action}
        if value is not MISSING:
            payload["value"] = value

        if args.dry_run:
            audit.write(
                "control_dry_run",
                request_type="control",
                node=node_id,
                action=action,
                request=payload,
                success=True,
                latency_ms=catalog_latency,
            )
            if args.json:
                _emit_json({"dry_run": True, "request": payload})
            else:
                print("DRY-RUN:", _json_dump(payload, indent=None))
            return EXIT_OK

        before: dict[str, Any] | None = None
        try:
            data = await client.wait_for_telemetry(
                timeout=min(args.timeout, 10.0),
                max_age=min(args.timeout, 5.0),
            )
            found = _telemetry_node(data, node_id)
            before = found[1] if found else None
        except DomeError as exc:
            _warn(args, f"Не удалось снять телеметрию до команды: {exc}")

        if not await _confirm(args, payload):
            audit.write(
                "control_cancelled",
                request_type="control",
                node=node_id,
                action=action,
                success=False,
                error="cancelled by user",
            )
            print("Отменено.", file=sys.stderr)
            return EXIT_FAILURE

        start = time.perf_counter()
        response = await client.control(node_id, action, value)
        latency = _ms_since(start)
        success = bool(response.get("success"))
        audit.write(
            "control",
            request_type="control",
            node=node_id,
            action=action,
            request_id=response.get("request_id"),
            success=success,
            error=None if success else _response_error(response),
            latency_ms=latency,
        )

        after: dict[str, Any] | None = None
        diff: dict[str, Any] | None = None
        if success and not args.no_verify:
            try:
                data = await client.wait_for_telemetry(
                    timeout=min(args.timeout, 10.0),
                    next_frame=True,
                )
            except DomeError as exc:
                _warn(args, f"Проверка по телеметрии не удалась: {exc}")
            else:
                found = _telemetry_node(data, node_id)
                after = found[1] if found else None
                diff = _telemetry_diff(before, after)

        if args.json:
            _emit_json(
                {
                    "success": success,
                    "accepted": success,
                    "request": payload,
                    "response": response,
                    "before": before,
                    "after": after,
                    "diff": diff,
                }
            )
        elif success:
            print(_accepted_text(response, verified=not args.no_verify))
            if diff:
                print("\nИзменения телеметрии:")
                for key, change in diff.items():
                    print(
                        f"  {key}: {_json_dump(change['before'], indent=None)}"
                        f" -> {_json_dump(change['after'], indent=None)}"
                    )
            elif args.no_verify:
                print("Телеметрия не проверялась (--no-verify).")
            else:
                print("Изменений в снятом узле не видно; проверьте телеметрию вручную.")
        else:
            print(f"ОШИБКА: {_response_error(response)}", file=sys.stderr)
        return EXIT_OK if success else EXIT_FAILURE


async def cmd_query(args: argparse.Namespace, audit: AuditLogger) -> int:
    if args.params:
        params = parse_value(args.params)
        if not isinstance(params, dict):
            raise CliError("params должен быть JSON-объектом", EXIT_FAILURE)
    else:
        params = {}

    async with _client(args) as client:
        audit.bind(client)
        start = time.perf_counter()
        catalog = await client.describe_nodes()
        catalog_latency = _ms_since(start)

        node = _find_node(catalog, args.node_id)
        if node is None:
            available = ", ".join(str(item.get("node_id")) for item in catalog if item.get("node_id"))
            raise CliError(
                f"Узел {args.node_id!r} не найден в каталоге. Доступные: {available or 'нет'}",
                EXIT_FAILURE,
            )
        node_id = str(node.get("node_id"))
        query = _resolve_query_name(node, args.query, force=args.force)

        start = time.perf_counter()
        response = await client.query(node_id, query, params)
        latency = _ms_since(start)
        success = bool(response.get("success"))
        audit.write(
            "query",
            request_type="query",
            node=node_id,
            query=query,
            request_id=response.get("request_id"),
            success=success,
            error=None if success else _response_error(response),
            latency_ms=latency,
            catalog_latency_ms=catalog_latency,
        )
        if args.json:
            _emit_json(response)
        elif success:
            print(_json_dump(response.get("result")))
        else:
            print(f"ОШИБКА: {_response_error(response)}", file=sys.stderr)
        return EXIT_OK if success else EXIT_FAILURE


async def cmd_chat(args: argparse.Namespace, audit: AuditLogger) -> int:
    async with _client(args) as client:
        audit.bind(client)
        start = time.perf_counter()
        messages = await client.read_messages(since_id=args.since_id, limit=args.limit)
        latency = _ms_since(start)
        audit.write(
            "chat",
            request_type="query",
            node="meeting_01",
            query="read_messages",
            success=True,
            latency_ms=latency,
            messages=len(messages),
        )
        if args.json:
            _emit_json(messages)
        else:
            print(format_chat_messages(messages))
        return EXIT_OK


async def cmd_say(args: argparse.Namespace, audit: AuditLogger) -> int:
    async with _client(args) as client:
        audit.bind(client)
        start = time.perf_counter()
        response = await client.post_message(args.text, to=args.to, name=args.name)
        latency = _ms_since(start)
        success = bool(response.get("success"))
        audit.write(
            "say",
            request_type="control",
            node="meeting_01",
            action="post_message",
            request_id=response.get("request_id"),
            success=success,
            error=None if success else _response_error(response),
            latency_ms=latency,
            to=args.to,
        )
        if args.json:
            _emit_json(response)
        elif success:
            print(f"OK · сообщение отправлено (to={args.to or 'all'})")
        else:
            print(f"ОШИБКА: {_response_error(response)}", file=sys.stderr)
        return EXIT_OK if success else EXIT_FAILURE


async def cmd_send(args: argparse.Namespace, audit: AuditLogger) -> int:
    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError as exc:
        raise CliError(f"Некорректный JSON: {exc}", EXIT_FAILURE) from exc
    if not isinstance(payload, dict):
        raise CliError("payload должен быть JSON-объектом", EXIT_FAILURE)

    async with _client(args) as client:
        audit.bind(client)
        if str(payload.get("type") or "").casefold() == "control" and not await _confirm(args, payload):
            audit.write(
                "send_cancelled",
                request_type="control",
                node=payload.get("node_id"),
                action=payload.get("action"),
                success=False,
                error="cancelled by user",
            )
            print("Отменено.", file=sys.stderr)
            return EXIT_FAILURE
        start = time.perf_counter()
        response = await client.request_raw(payload)
        latency = _ms_since(start)
        success = bool(response.get("success", True))
        audit.write(
            "send",
            request_type=str(payload.get("type") or "raw"),
            node=payload.get("node_id"),
            action=payload.get("action"),
            query=payload.get("query"),
            request_id=response.get("request_id", payload.get("request_id")),
            success=success,
            error=None if success else _response_error(response),
            latency_ms=latency,
        )
        print(_json_dump(response))
        return EXIT_OK if success else EXIT_FAILURE


async def cmd_doctor(args: argparse.Namespace, audit: AuditLogger) -> int:
    steps: list[dict[str, Any]] = []
    exit_code = EXIT_OK

    def add_step(name: str, ok: bool, latency: float | None = None, detail: str = "") -> None:
        steps.append(
            {
                "name": name,
                "ok": ok,
                "latency_ms": latency,
                "detail": detail,
            }
        )

    try:
        url, key, name, timeout = _settings(args)
    except CliError as exc:
        add_step("url/key", False, None, str(exc))
        return _finish_doctor(args, audit, steps, EXIT_CONFIG)

    url_ok = bool(url)
    key_ok = bool(key)
    add_step(
        "url/key",
        url_ok and key_ok,
        None,
        f"url={'set' if url_ok else 'missing'}, key={'set' if key_ok else 'missing'}",
    )
    if not (url_ok and key_ok):
        return _finish_doctor(args, audit, steps, EXIT_CONFIG)

    client = DomeClient(
        url,
        key,
        name=name,
        request_timeout=timeout,
        open_timeout=min(timeout, 10.0),
        auth_timeout=min(timeout, 10.0),
    )
    try:
        start = time.perf_counter()
        try:
            auth = await client.connect_with_retry(*retry_policy(timeout))
        except DomeConfigError as exc:
            add_step("connect", False, _ms_since(start), str(exc))
            return _finish_doctor(args, audit, steps, EXIT_CONFIG)
        except DomeAuthError as exc:
            add_step("connect", False, _ms_since(start), str(exc))
            add_step("auth", False, None, str(exc))
            return _finish_doctor(args, audit, steps, EXIT_CONFIG)
        except (DomeTimeout, DomeConnectionError, OSError) as exc:
            add_step("connect", False, _ms_since(start), str(exc))
            return _finish_doctor(args, audit, steps, EXIT_NETWORK)
        connect_latency = _ms_since(start)
        add_step("connect", True, connect_latency, "websocket established")
        audit.bind(client)

        add_step(
            "auth",
            bool(client.role),
            None,
            f"role={client.role}, team={client.team}, mode={auth.get('mode', '—')}",
        )
        if not client.role:
            return _finish_doctor(args, audit, steps, EXIT_CONFIG)

        start = time.perf_counter()
        try:
            response = await client.ping()
        except DomeTimeout as exc:
            add_step("ping", False, _ms_since(start), str(exc))
            exit_code = EXIT_NETWORK
        except (DomeConnectionError, OSError) as exc:
            add_step("ping", False, _ms_since(start), str(exc))
            exit_code = EXIT_NETWORK
        except DomeError as exc:
            add_step("ping", False, _ms_since(start), str(exc))
            exit_code = EXIT_FAILURE
        else:
            latency = _ms_since(start)
            ok = bool(response.get("success"))
            add_step("ping", ok, latency, f"request_id={response.get('request_id')}")
            if not ok:
                exit_code = EXIT_FAILURE

        start = time.perf_counter()
        try:
            catalog = await client.describe_nodes()
        except DomeTimeout as exc:
            add_step("describe_nodes", False, _ms_since(start), str(exc))
            exit_code = EXIT_NETWORK
        except (DomeConnectionError, OSError) as exc:
            add_step("describe_nodes", False, _ms_since(start), str(exc))
            exit_code = EXIT_NETWORK
        except DomeError as exc:
            add_step("describe_nodes", False, _ms_since(start), str(exc))
            if exit_code == EXIT_OK:
                exit_code = EXIT_FAILURE
        else:
            latency = _ms_since(start)
            add_step("describe_nodes", True, latency, f"nodes={len(catalog)}")

        start = time.perf_counter()
        try:
            telemetry = await client.wait_for_telemetry(timeout=min(timeout, 10.0))
        except DomeTimeout as exc:
            add_step("telemetry", False, _ms_since(start), str(exc))
            exit_code = EXIT_NETWORK
        except (DomeConnectionError, OSError) as exc:
            add_step("telemetry", False, _ms_since(start), str(exc))
            exit_code = EXIT_NETWORK
        except DomeError as exc:
            add_step("telemetry", False, _ms_since(start), str(exc))
            if exit_code == EXIT_OK:
                exit_code = EXIT_FAILURE
        else:
            latency = _ms_since(start)
            nodes = telemetry.get("nodes") or {}
            add_step("telemetry", True, latency, f"nodes={len(nodes)}")

        return _finish_doctor(args, audit, steps, exit_code)
    finally:
        await client.close()


def _finish_doctor(
    args: argparse.Namespace,
    audit: AuditLogger,
    steps: list[dict[str, Any]],
    exit_code: int,
) -> int:
    ok = all(bool(step["ok"]) for step in steps) and exit_code == EXIT_OK
    audit.write(
        "doctor",
        success=ok,
        steps=[{"name": step["name"], "ok": step["ok"], "latency_ms": step["latency_ms"]} for step in steps],
        exit_code=exit_code,
    )
    if args.json:
        _emit_json({"ok": ok, "steps": steps})
    else:
        for step in steps:
            status = "PASS" if step["ok"] else "FAIL"
            latency = f" ({step['latency_ms']:.1f} ms)" if step["latency_ms"] is not None else ""
            detail = f" — {step['detail']}" if step["detail"] else ""
            print(f"{status:<4} {step['name']}{latency}{detail}")
        print("ИТОГО:", "PASS" if ok else "FAIL")
    return exit_code if not ok else EXIT_OK


# --- REPL --------------------------------------------------------------------


HELP_TEXT = """Команды:
  help                              эта справка
  whoami                            роль/команда/URL
  ping                              проверить задержку
  status                            сводка купола
  nodes [f]                         каталог + телеметрия, фильтр f
  catalog [f]                       каталог узлов, фильтр f
  telemetry [node]                  сырая телеметрия (или один узел)
  control NODE ACTION [VALUE]       команда с проверкой каталога
  query NODE QUERY [PARAMS]         запрос данных узла
  chat [limit]                      прочитать сообщения meeting_01
  say TEXT                          отправить сообщение в meeting_01
  send JSON                         сырой запрос (request_id добавится)
  quit | exit                       выход
"""


def _shell_tail(text: str, words: int) -> str:
    """Return *text* after skipping *words* shell words, preserving quoting."""
    index = 0
    length = len(text)
    skipped = 0
    while index < length and skipped < words:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        quote: str | None = None
        while index < length:
            char = text[index]
            if quote:
                if char == quote:
                    quote = None
                elif char == "\\" and quote == '"':
                    index += 1
            elif char in {"'", '"'}:
                quote = char
            elif char.isspace():
                break
            index += 1
        skipped += 1
    return text[index:].strip()


def _parse_repl_value(line: str, skip_words: int) -> Any:
    tail = _shell_tail(line, skip_words)
    if not tail:
        return MISSING
    if tail[0] in "[{":
        return parse_value(tail)
    try:
        tokens = shlex.split(tail)
    except ValueError:
        return parse_value(tail)
    if len(tokens) == 1:
        return parse_value(tokens[0])
    return parse_value(tail)


def _json_repl_argument(line: str, skip_words: int) -> str:
    tail = _shell_tail(line, skip_words)
    if not tail:
        return ""
    if tail[0] in "[{":
        return tail
    try:
        tokens = shlex.split(tail)
    except ValueError:
        return tail
    if len(tokens) == 1:
        return tokens[0]
    return tail


async def _repl_catalog(client: DomeClient, cache: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if cache is not None:
        return cache
    return await client.describe_nodes()


async def cmd_repl(args: argparse.Namespace, audit: AuditLogger) -> int:
    client = _make_client(args)
    await client.connect_with_retry(*retry_policy(args.timeout))
    audit.bind(client)
    catalog: list[dict[str, Any]] | None = None
    prompt = "dome> " if sys.stdin.isatty() else ""
    try:
        while True:
            try:
                line = await asyncio.to_thread(input, prompt)
            except EOFError:
                if prompt:
                    print()
                break
            except KeyboardInterrupt:
                if prompt:
                    print()
                raise

            try:
                parts = shlex.split(line)
            except ValueError as exc:
                print(f"Ошибка разбора строки: {exc}", file=sys.stderr)
                continue
            if not parts:
                continue

            command = parts[0].casefold()
            rest = parts[1:]
            try:
                if command in {"quit", "exit"}:
                    break
                if command in {"help", "?"}:
                    print(HELP_TEXT, end="")
                    continue
                if command == "whoami":
                    print(
                        f"URL: {client.url}\nrole: {client.role}\nteam: {client.team}\nname: {client.name}"
                    )
                    continue
                if command == "ping":
                    start = time.perf_counter()
                    response = await client.ping()
                    latency = _ms_since(start)
                    success = bool(response.get("success"))
                    audit.write(
                        "ping",
                        request_type="ping",
                        success=success,
                        latency_ms=latency,
                        error=None if success else _response_error(response),
                    )
                    print(f"PONG · {latency:.1f} ms" if success else f"FAIL: {_response_error(response)}")
                    continue
                if command == "status":
                    data = await client.wait_for_telemetry(
                        timeout=args.timeout,
                        max_age=min(args.timeout, 5.0),
                    )
                    print(format_status(data, role=client.role, team=client.team))
                    continue
                if command == "catalog":
                    catalog = await _repl_catalog(client, catalog)
                    filtered = _filter_catalog(catalog, search=" ".join(rest) if rest else None)
                    print(format_describe(filtered))
                    continue
                if command == "nodes":
                    catalog = await _repl_catalog(client, catalog)
                    data = await client.wait_for_telemetry(
                        timeout=args.timeout,
                        max_age=min(args.timeout, 5.0),
                    )
                    filtered = _filter_catalog(catalog, search=" ".join(rest) if rest else None)
                    print(format_nodes(filtered, data))
                    continue
                if command == "telemetry":
                    data = await client.wait_for_telemetry(
                        timeout=args.timeout,
                        max_age=min(args.timeout, 5.0),
                    )
                    payload: Any = data
                    if rest:
                        found = _telemetry_node(data, rest[0])
                        if found is None:
                            raise CliError(f"Узел {rest[0]!r} отсутствует в телеметрии", EXIT_FAILURE)
                        _, payload = found
                    print(_json_dump(payload))
                    continue
                if command == "control":
                    if len(rest) < 2:
                        raise CliError("использование: control NODE ACTION [VALUE]", EXIT_FAILURE)
                    node_name, action_name = rest[0], rest[1]
                    value = _parse_repl_value(line, 3)
                    catalog = await _repl_catalog(client, catalog)
                    node = _find_node(catalog, node_name)
                    if node is None:
                        raise CliError(f"Узел {node_name!r} не найден в каталоге", EXIT_FAILURE)
                    node_id = str(node.get("node_id"))
                    action = _resolve_control_action(args, node, action_name, force=False)
                    payload = {"type": "control", "node_id": node_id, "action": action}
                    if value is not MISSING:
                        payload["value"] = value
                    if not await _confirm(args, payload):
                        print("Отменено.", file=sys.stderr)
                        continue
                    try:
                        before_data = await client.wait_for_telemetry(
                            timeout=min(args.timeout, 10.0),
                            max_age=min(args.timeout, 5.0),
                        )
                    except DomeError:
                        before_data = client.telemetry or {}
                    before_found = _telemetry_node(before_data, node_id)
                    before = before_found[1] if before_found else None
                    start = time.perf_counter()
                    response = await client.control(node_id, action, value)
                    latency = _ms_since(start)
                    success = bool(response.get("success"))
                    audit.write(
                        "control",
                        request_type="control",
                        node=node_id,
                        action=action,
                        success=success,
                        latency_ms=latency,
                        error=None if success else _response_error(response),
                    )
                    if not success:
                        print(f"ОШИБКА: {_response_error(response)}", file=sys.stderr)
                        continue
                    print(_accepted_text(response, verified=True))
                    try:
                        data = await client.wait_for_telemetry(
                            timeout=min(args.timeout, 10.0),
                            next_frame=True,
                        )
                    except DomeError as exc:
                        print(f"Проверка не удалась: {exc}", file=sys.stderr)
                        continue
                    after_found = _telemetry_node(data, node_id)
                    diff = _telemetry_diff(before, after_found[1] if after_found else None)
                    if diff:
                        for key, change in diff.items():
                            print(
                                f"  {key}: {_json_dump(change['before'], indent=None)}"
                                f" -> {_json_dump(change['after'], indent=None)}"
                            )
                    continue
                if command == "query":
                    if len(rest) < 2:
                        raise CliError("использование: query NODE QUERY [PARAMS]", EXIT_FAILURE)
                    node_name, query_name = rest[0], rest[1]
                    params = _parse_repl_value(line, 3)
                    if params is MISSING:
                        params = {}
                    if not isinstance(params, dict):
                        raise CliError("params должен быть JSON-объектом", EXIT_FAILURE)
                    catalog = await _repl_catalog(client, catalog)
                    node = _find_node(catalog, node_name)
                    if node is None:
                        raise CliError(f"Узел {node_name!r} не найден в каталоге", EXIT_FAILURE)
                    node_id = str(node.get("node_id"))
                    query = _resolve_query_name(node, query_name, force=False)
                    response = await client.query(node_id, query, params)
                    if response.get("success"):
                        print(_json_dump(response.get("result")))
                    else:
                        print(f"ОШИБКА: {_response_error(response)}", file=sys.stderr)
                    continue
                if command == "chat":
                    limit = 50
                    if rest:
                        try:
                            limit = int(rest[0])
                        except ValueError as exc:
                            raise CliError(f"limit должен быть числом: {exc}", EXIT_FAILURE) from exc
                    messages = await client.read_messages(limit=limit)
                    print(format_chat_messages(messages))
                    continue
                if command == "say":
                    text = " ".join(rest)
                    if not text:
                        raise CliError("использование: say TEXT", EXIT_FAILURE)
                    to = getattr(args, "to", "operator")
                    name = getattr(args, "name", None)
                    start = time.perf_counter()
                    response = await client.post_message(text, to=to, name=name)
                    latency = _ms_since(start)
                    success = bool(response.get("success"))
                    audit.write(
                        "say",
                        request_type="control",
                        node="meeting_01",
                        action="post_message",
                        success=success,
                        latency_ms=latency,
                        to=to,
                        error=None if success else _response_error(response),
                    )
                    print("OK · отправлено" if success else f"ОШИБКА: {_response_error(response)}")
                    continue
                if command == "send":
                    raw = _json_repl_argument(line, 1)
                    if not raw:
                        raise CliError("использование: send JSON", EXIT_FAILURE)
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise CliError(f"Некорректный JSON: {exc}", EXIT_FAILURE) from exc
                    if not isinstance(payload, dict):
                        raise CliError("payload должен быть JSON-объектом", EXIT_FAILURE)
                    if str(payload.get("type") or "").casefold() == "control" and not await _confirm(
                        args, payload
                    ):
                        print("Отменено.", file=sys.stderr)
                        continue
                    response = await client.request_raw(payload)
                    print(_json_dump(response))
                    continue
                print(f"Неизвестная команда: {parts[0]}. Введите help.", file=sys.stderr)
            except CliError as exc:
                print(f"Ошибка: {exc}", file=sys.stderr)
            except DomeSessionRevoked as exc:
                print(f"Сессия отозвана: {exc}", file=sys.stderr)
                audit.write("revoked", success=False, error=str(exc))
                break
            except DomeTimeout as exc:
                print(f"Таймаут: {exc}", file=sys.stderr)
            except DomeConnectionError as exc:
                print(f"Соединение потеряно: {exc}", file=sys.stderr)
                break
            except DomeAuthError as exc:
                print(f"Ошибка авторизации: {exc}", file=sys.stderr)
                break
            except DomeError as exc:
                print(f"Ошибка: {exc}", file=sys.stderr)
        return EXIT_OK
    finally:
        await client.close()


# --- argument parsing --------------------------------------------------------


def _add_common_options(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument("--url", default=default, help="ws://host:8765 (or DOME_URL)")
    parser.add_argument("--key", default=default, help="team key (or DOME_KEY)")
    parser.add_argument(
        "--key-file",
        default=default,
        help="file containing 'team: <key>' (or DOME_KEY_FILE)",
    )
    parser.add_argument("--env-file", default=default, help="path to .env")
    parser.add_argument("--timeout", type=float, default=default, help="request timeout, seconds")
    parser.add_argument("--json", action="store_true", default=default, help="pure JSON output")
    parser.add_argument("--log-file", default=default, help="append a JSONL audit log")
    parser.add_argument("--log-telemetry", action="store_true", default=default, help="include telemetry in the audit log")
    parser.add_argument("--verbose", action="store_true", default=default, help="verbose diagnostics")
    parser.add_argument("--quiet", action="store_true", default=default, help="suppress warnings")
    parser.add_argument(
        "--demo",
        default=default,
        metavar="TELEMETRY_JSONL",
        help="offline demo from a telemetry.jsonl file",
    )


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    _add_common_options(common, suppress=True)

    parser = argparse.ArgumentParser(
        prog="dome-panel",
        description="Dome WebSocket API management CLI (Team 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_options(parser, suppress=False)
    parser.set_defaults(
        url=None,
        key=None,
        key_file=None,
        env_file=".env",
        timeout=None,
        json=False,
        log_file=None,
        log_telemetry=False,
        verbose=False,
        quiet=False,
        demo=None,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("whoami", parents=[common], help="connect and print role/team/URL")
    subparsers.add_parser("ping", parents=[common], help="authenticate and ping")
    subparsers.add_parser("doctor", parents=[common], help="run connection diagnostics")
    for name in ("repl", "shell"):
        repl = subparsers.add_parser(
            name,
            parents=[common],
            help="interactive shell over one connection" if name == "repl" else "alias for repl",
        )
        repl.add_argument("--yes", action="store_true", help="skip confirmation for control commands")

    for name in ("catalog", "describe"):
        catalog = subparsers.add_parser(
            name,
            parents=[common],
            help="show the describe_nodes catalog",
        )
        catalog.add_argument("--system", default=None, help="filter by system/category")
        catalog.add_argument("--search", default=None, help="case-insensitive text filter")

    nodes = subparsers.add_parser("nodes", parents=[common], help="catalog and live telemetry")
    nodes.add_argument("--system", default=None, help="filter by system/category")
    nodes.add_argument("--search", default=None, help="case-insensitive text filter")

    subparsers.add_parser("status", parents=[common], help="dome dashboard")

    telemetry = subparsers.add_parser("telemetry", parents=[common], help="raw telemetry")
    telemetry.add_argument("--node", default=None, help="only this node")

    watch = subparsers.add_parser("watch", parents=[common], help="continuously print telemetry")
    watch.add_argument("--raw", action="store_true", help="print raw JSON frames")
    watch.add_argument("--node", default=None, help="only this node")
    watch.add_argument("--interval", type=float, default=2.0, help="minimum seconds between frames")
    watch.add_argument("--reconnect", action="store_true", help="reconnect automatically")

    control = subparsers.add_parser("control", parents=[common], help="send a validated control action")
    control.add_argument("node_id")
    control.add_argument("action")
    control.add_argument("value", nargs="?", help="JSON value; bare text is a string")
    control.add_argument("--dry-run", action="store_true", help="print the request and send nothing")
    control.add_argument("--yes", action="store_true", help="skip interactive confirmation")
    control.add_argument("--no-verify", action="store_true", help="do not wait for the next telemetry frame")
    control.add_argument("--force", action="store_true", help="allow an action missing from the catalog")

    query = subparsers.add_parser("query", parents=[common], help="run a validated node query")
    query.add_argument("node_id")
    query.add_argument("query")
    query.add_argument("params", nargs="?", help="JSON object")
    query.add_argument("--force", action="store_true", help="allow a query missing from the catalog")

    chat = subparsers.add_parser("chat", parents=[common], help="read meeting_01 messages")
    chat.add_argument("--limit", type=int, default=50)
    chat.add_argument("--since-id", type=int, default=None)

    say = subparsers.add_parser("say", parents=[common], help="post a meeting_01 message")
    say.add_argument("text")
    say.add_argument("--to", default="operator")
    say.add_argument("--name", default=None)

    send = subparsers.add_parser("send", parents=[common], help="send a raw JSON request")
    send.add_argument("payload")
    send.add_argument("--yes", action="store_true", help="skip confirmation for raw control requests")

    return parser


# --- entry point -------------------------------------------------------------


HANDLERS: dict[str, CommandHandler] = {
    "whoami": cmd_whoami,
    "ping": cmd_ping,
    "doctor": cmd_doctor,
    "repl": cmd_repl,
    "shell": cmd_repl,
    "catalog": cmd_catalog,
    "describe": cmd_catalog,
    "nodes": cmd_nodes,
    "status": cmd_status,
    "telemetry": cmd_telemetry,
    "watch": cmd_watch,
    "control": cmd_control,
    "query": cmd_query,
    "chat": cmd_chat,
    "say": cmd_say,
    "send": cmd_send,
}


async def _dispatch(args: argparse.Namespace, audit: AuditLogger) -> int:
    handler = HANDLERS.get(args.command)
    if handler is None:
        raise CliError(f"Неизвестная команда: {args.command}", EXIT_CONFIG)
    return await handler(args, audit)


def _configure_logging(args: argparse.Namespace) -> None:
    if args.quiet:
        level = logging.ERROR
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _emit_failure(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "json", False):
        print(_json_dump({"success": False, "error": message}))
    else:
        print(message, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timeout is None:
        raw_timeout = os.getenv("DOME_TIMEOUT", "30")
        try:
            args.timeout = float(raw_timeout)
        except ValueError:
            parser.error(f"DOME_TIMEOUT должен быть числом, получено: {raw_timeout!r}")
    _configure_logging(args)

    audit: AuditLogger | None = None
    try:
        audit = AuditLogger(
            args.log_file,
            url=_url_for_audit(args),
            log_telemetry=args.log_telemetry,
        )
        return asyncio.run(_dispatch(args, audit))
    except KeyboardInterrupt:
        if audit is not None:
            audit.write("interrupted", success=False, error="KeyboardInterrupt")
        print("\nОстановлено.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except CliError as exc:
        if audit is not None:
            audit.write("error", success=False, error=str(exc), exit_code=exc.code)
        _emit_failure(args, str(exc))
        return exc.code
    except SystemExit as exc:
        if exc.code in (None, 0):
            return EXIT_OK
        message = str(exc.code)
        if audit is not None:
            audit.write("config_error", success=False, error=message)
        _emit_failure(args, message)
        return EXIT_CONFIG
    except DomeConfigError as exc:
        if audit is not None:
            audit.write("config_error", success=False, error=str(exc))
        _emit_failure(args, f"Ошибка конфигурации: {exc}")
        return EXIT_CONFIG
    except DomeAuthError as exc:
        if audit is not None:
            audit.write("auth_error", success=False, error=str(exc))
        _emit_failure(args, f"Ошибка авторизации: {exc}")
        return EXIT_CONFIG
    except DomeSessionRevoked as exc:
        if audit is not None:
            audit.write("revoked", success=False, error=str(exc))
        _emit_failure(args, f"Сессия отозвана сервером: {exc}")
        return EXIT_FAILURE
    except (DomeTimeout, DomeConnectionError, OSError, asyncio.TimeoutError) as exc:
        if audit is not None:
            audit.write("network_error", success=False, error=str(exc))
        _emit_failure(args, f"Сетевая ошибка: {exc}")
        return EXIT_NETWORK
    except DomeError as exc:
        if audit is not None:
            audit.write("api_error", success=False, error=str(exc))
        _emit_failure(args, f"Ошибка API: {exc}")
        return EXIT_FAILURE
    except Exception as exc:
        if audit is not None:
            audit.write("internal_error", success=False, error=f"{type(exc).__name__}: {exc}")
        logger.exception("unexpected command failure")
        _emit_failure(args, f"Внутренняя ошибка: {type(exc).__name__}: {exc}")
        return EXIT_FAILURE
    finally:
        if audit is not None:
            audit.close()


if __name__ == "__main__":
    raise SystemExit(main())
