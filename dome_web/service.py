"""Browser bridge. Credentials and the sole Dome connection stay in Python.

Demo needs only the standard library. Live mode reuses the project's existing
client, command gate, agent, verification predicates and JSONL audit writer.
"""
from __future__ import annotations

import asyncio
import copy
import importlib
import json
import math
import os
import re
import secrets
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

STATIC = Path(__file__).parent / "static"
HARD_DENY = {"block_automation", "block_interlock", "set_thresholds", "unlock_door", "remove_part", "shutdown", "reboot"}


class WebError(RuntimeError):
    def __init__(self, text: str, status: int = 400):
        super().__init__(text)
        self.status = status


def iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class DashboardService:
    def __init__(self, root: Path, *, demo: bool = False, env_file: str = ".env"):
        self.root = Path(root).resolve()
        self.demo = demo
        self.csrf = secrets.token_urlsafe(32)
        self.loop = asyncio.new_event_loop()
        self.thread: threading.Thread | None = None
        self.monitor: asyncio.Task | None = None
        self.agent = None
        self.agent_task: asyncio.Task | None = None
        self.api = None
        self.audit = None
        self.config = None
        self.want_connection = False
        self.connection = "demo" if demo else "disconnected"
        self.last_error: str | None = None
        self.catalog: list[dict] = []
        self.messages: deque[dict] = deque(maxlen=300)
        self.events: deque[dict] = deque(maxlen=1500)
        self.history: deque[dict] = deque(maxlen=600)
        self.telemetry: dict = {"nodes": {}, "game_time": {}}
        self._mutation_lock = None
        self._readings_at: dict[str, tuple[Any, float]] = {}
        self._catalog_at = 0.0
        self._chat_at = 0.0
        self._history_at = 0.0
        self._retry_at = 0.0
        self._demo_started = time.monotonic()
        self._demo_agent = False
        self._demo_task = None
        self._demo_stats = {"decisions": 0, "actions": 0, "action_failures": 0, "reconnects": 0}
        self._kb_chunks = None
        from dome_agent.knowledge import KnowledgeService
        from .conversation import Conversation
        self.conversation = Conversation(self)
        self.env_file = str((self.root / env_file).resolve()) if not Path(env_file).is_absolute() else env_file
        values = {}
        env_path = Path(self.env_file)
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8-sig").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    values[k.strip()] = v.strip().strip("\"'")
        values.update(os.environ)
        truth = lambda k, default=False: str(values.get(k, str(default))).lower() in {"1", "true", "yes", "on"}
        kb_dir = Path(values.get("AGENT_KB_DIR") or self.root / "dome_rag").expanduser()
        if not kb_dir.is_absolute(): kb_dir = self.root / kb_dir
        kb_db = Path(values.get("AGENT_KB_DB") or kb_dir / "build/kb.sqlite").expanduser()
        if not kb_db.is_absolute(): kb_db = self.root / kb_db
        self.knowledge = KnowledgeService(kb_dir, db_path=kb_db,
            enabled=truth("AGENT_KB_ENABLED", truth("AGENT_KB", True)),
            max_cards=int(values.get("AGENT_KB_MAX_CARDS", 6)),
            max_chars=int(values.get("AGENT_KB_MAX_CHARS", 6000)))
        self.settings = {
            "url": values.get("DOME_URL", ""), "environment": values.get("DOME_ENVIRONMENT", "custom"), "key": values.get("DOME_KEY", ""),
            "key_file": values.get("DOME_KEY_FILE", ""), "name": values.get("AGENT_NAME", "team3-agent"),
            "model": values.get("DEEPSEEK_MODEL", "deepseek-chat"),
            "base_url": values.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "api_key": values.get("DEEPSEEK_API_KEY", ""), "mode": values.get("AGENT_MODE", "normal"),
            "control_enabled": truth("AGENT_ALLOW_CONTROL"), "dry_run": truth("AGENT_DRY_RUN"),
            "llm_enabled": truth("AGENT_LLM", True),
        }
        policy_file = Path(values.get("AGENT_POLICY_FILE", "dome_policy.json"))
        self.policy_path = policy_file if policy_file.is_absolute() else self.root / policy_file
        self.log_dir = self.root / "logs"
        if demo:
            data = json.loads((STATIC / "demo.json").read_text(encoding="utf-8"))
            self.catalog = data.pop("catalog")
            self.telemetry = data
            self.settings.update(url="demo://local", name="Купол · демо")
            self.messages.append({"id": 1, "time": time.time(), "sender": {"role": "operator", "name": "Оператор"}, "text": "Панель готова. Здесь можно проверить интерфейс без подключения к оборудованию.", "to": None})
            self.event("connection", event="demo", details="Открыт демонстрационный стенд. Все изменения локальные.")

    def start(self):
        def run():
            asyncio.set_event_loop(self.loop)
            self._mutation_lock = asyncio.Lock()
            self.monitor = self.loop.create_task(self._monitor())
            self.loop.run_forever()
        self.thread = threading.Thread(target=run, daemon=True, name="dome-web-loop")
        self.thread.start()

    def call(self, coro, timeout: float = 45):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout)
        except TimeoutError as exc:
            # A response timeout must never automatically repeat a control.
            raise WebError("Ожидание ответа истекло. Проверьте журнал и телеметрию перед повтором.", 504) from exc

    def close(self):
        self.knowledge.close()
        if not self.thread:
            if not self.loop.is_closed():
                self.loop.close()
            return
        try:
            self.call(self._close(), timeout=15)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=3)
            if not self.thread.is_alive() and not self.loop.is_closed():
                self.loop.close()

    async def _close(self):
        self.want_connection = False
        if self.monitor:
            self.monitor.cancel()
            await asyncio.gather(self.monitor, return_exceptions=True)
        await self._stop_agent()
        if self.api:
            await self.api.close()
        if self.audit and not self.audit.closed:
            self.audit.write("stop", reason="web_shutdown")
            self.audit.close()

    def redact(self, value):
        if isinstance(value, dict):
            return {str(k): ("[REDACTED]" if re.search(r"(?:key|token|password|secret)$", str(k), re.I) else self.redact(v)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact(v) for v in value]
        if isinstance(value, str):
            known = (self.settings.get("key"), self.settings.get("api_key"))
            if self.config:
                known += self.config.secrets()
            for key in known:
                if key:
                    value = value.replace(key, "[REDACTED]")
            return value
        return value

    def event(self, kind: str, **fields):
        row = self.redact({"ts": iso(), "kind": kind, **fields})
        self.events.append(row)
        return row

    def _load_backend(self):
        try:
            self.backend = importlib.import_module("dome_agent.agent")
            self.config_class = importlib.import_module("dome_agent.config").AgentConfig
            self.api_class = importlib.import_module("dome_control.agent_api").DomeAgentAPI
            self.gate_class = importlib.import_module("dome_agent.gate").CommandGate
            self.log_class = importlib.import_module("dome_agent.logging").AgentLog
        except ImportError as exc:
            raise WebError("Для подключения установите зависимости проекта: python -m pip install -r requirements.txt", 503) from exc

    def _make_config(self):
        self._load_backend()
        s = self.settings
        key = s["key"]
        if not key and s["key_file"]:
            loader = importlib.import_module("dome_control.panel").read_team_key
            key = loader(s["key_file"]) or ""
        config = self.config_class(
            dome_url=s["url"], dome_key=key, agent_name=s["name"],
            deepseek_api_key=s["api_key"] or None, deepseek_base_url=s["base_url"],
            deepseek_model=s["model"], mode=s["mode"], dry_run=s["dry_run"],
            control_enabled=s["control_enabled"], llm_enabled=s["llm_enabled"],
            policy_file=self.policy_path, log_dir=self.log_dir, state_file=self.log_dir / "agent_state.json",
        )
        # Validate the exact policy before any connection/agent launch.
        self.gate_class(config.policy_file, control_enabled=config.control_enabled, yolo=config.yolo)
        return config

    async def _open(self):
        self.config = self._make_config()
        self.api = self.api_class(self.config.dome_url, self.config.dome_key, name=self.config.agent_name, timeout=8, min_send_gap=0.35)
        if not self.audit or self.audit.closed:
            self.audit = self.log_class(self.log_dir, run_id="web_" + datetime.now().strftime("%Y%m%d-%H%M%S"), secrets=self.config.secrets())
            self.audit.write("start", agent="dome-web", version="1.0.0", models=[], config=self.config.safe_dict())
        self.connection = "connecting"
        await self.api.connect()
        self.catalog = await self.api.catalog(refresh=True)
        self._catalog_at = time.monotonic()
        self.connection = "connected"
        self.last_error = None
        self.event("connection", event="connected", details="Подключение к Куполу установлено")

    async def _monitor(self):
        while True:
            async with self._mutation_lock:
                try:
                    if self.demo:
                        self._demo_tick()
                    elif self.want_connection:
                        running = bool(self.agent_task and not self.agent_task.done())
                        if not self.api and time.monotonic() >= self._retry_at:
                            await self._open()
                        if self.api:
                            client = self.api.client
                            if not getattr(client, "_closed", True):
                                self.connection = "connected"
                                if client.telemetry:
                                    self.telemetry = copy.deepcopy(client.telemetry)
                                if time.monotonic() - self._chat_at > 5:
                                    self._chat_at = time.monotonic()
                                    try:
                                        messages = await self.api.read_messages(limit=100)
                                        if isinstance(messages, list):
                                            self.messages = deque(messages, maxlen=300)
                                    except Exception:
                                        pass
                                if time.monotonic() - self._catalog_at > 60:
                                    self.catalog = await self.api.catalog(refresh=True)
                                    self._catalog_at = time.monotonic()
                            elif not running:
                                await self.api.close()
                                self.api = None
                                self.connection = "reconnecting"
                                self._retry_at = time.monotonic() + 5
                            else:
                                self.connection = "reconnecting"
                    self._record_history()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.last_error = self.redact(str(exc))
                    self.connection = "error"
                    self.event("error", where="connection", error=self.last_error)
                    if self.api and not (self.agent_task and not self.agent_task.done()):
                        await self.api.close()
                        self.api = None
                    self._retry_at = time.monotonic() + 8
            await asyncio.sleep(1)

    def _demo_tick(self):
        t = time.monotonic() - self._demo_started
        nodes = self.telemetry["nodes"]
        for node in nodes.values():
            node["updated_at"] = time.time()
        if not nodes:
            return
        solar = round(850 + math.sin(t / 18) * 90, 1)
        nodes["solar_panels_01"]["power_w"] = solar
        nodes["solar_inverter_01"]["pv_power_w"] = solar * .97
        self.telemetry["game_time"] = {"day": 1, "hour": 10.5 + t / 3600}
        for key in ("printer_3d_01", "cnc_01"):
            n = nodes[key]
            if n.get("state") in {"PRINTING", "RUNNING"}:
                n["progress_pct"] = min(100, float(n.get("progress_pct", 0)) + .025)
                if n["progress_pct"] >= 100:
                    n["state"] = "COMPLETED"
                    if key == "printer_3d_01":
                        n["bed_clear"] = False
        if self._demo_agent:
            self._demo_stats["decisions"] += 1

    def _record_history(self):
        if time.monotonic() - self._history_at < 3 or not self.telemetry.get("nodes"):
            return
        self._history_at = time.monotonic()
        ns = self.telemetry["nodes"]
        self.history.append({"time": time.time(), "solar": ns.get("solar_panels_01", {}).get("power_w"), "load": ns.get("solar_inverter_01", {}).get("load_power_w"), "soc": ns.get("battery_01", {}).get("soc_pct"), "co2": ns.get("climate_sensor_01", {}).get("co2_ppm")})
        for k, n in ns.items():
            stamp = n.get("updated_at")
            old = self._readings_at.get(k)
            if old is None or old[0] != stamp:
                self._readings_at[k] = (stamp, time.monotonic())

    def _policy(self):
        try:
            return json.loads(self.policy_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"default": "deny", "allow": {}, "error": "Политика недоступна"}

    def _permission(self, node_id, action, value=None):
        if str(action).lower() in HARD_DENY:
            return False, "Действие запрещено существующим защитным списком агента"
        if not self.settings["control_enabled"] and self.settings["mode"] != "yolo":
            return False, "Управление выключено в настройках"
        if self.settings["mode"] == "yolo":
            return True, "Экспериментальный режим"
        policy = self._policy()
        if action in policy.get("deny_actions", []) or node_id in policy.get("deny_nodes", []):
            return False, "Запрещено политикой"
        rule = policy.get("allow", {}).get(node_id, {})
        actions = rule.get("actions", []) if isinstance(rule, dict) else rule
        if action not in actions:
            return False, "Действие отсутствует в разрешениях dome_policy.json"
        values = rule.get("values", {}) if isinstance(rule, dict) else {}
        if action in values and isinstance(values[action], list) and value not in values[action]:
            return False, "Параметр не разрешён политикой"
        return True, "Разрешено политикой"

    def _alerts(self):
        if not self.demo:
            try:
                return self.backend.Snapshot(self.telemetry).alerts()
            except Exception:
                pass
        ns = self.telemetry.get("nodes", {})
        alerts = []
        for k, n in ns.items():
            if n.get("online") is False:
                alerts.append({"node_id": k, "severity": "warning", "code": "OFFLINE", "message": "Узел не на связи"})
            if n.get("alarm_code") or n.get("error_code") or n.get("smoke_detected"):
                alerts.append({"node_id": k, "severity": "critical", "code": n.get("alarm_code") or n.get("error_code") or "SMOKE", "message": "Требуется проверка оборудования"})
        return alerts

    async def snapshot(self):
        running = self._demo_agent if self.demo else bool(self.agent_task and not self.agent_task.done())
        role = "team" if self.demo else getattr(getattr(self.api, "client", None), "role", None)
        age = 0 if self.demo else getattr(getattr(self.api, "client", None), "telemetry_age", None)
        settings = {k: v for k, v in self.settings.items() if k not in {"key", "api_key", "key_file"}}
        settings.update(key_saved=bool(self.settings["key"] or self.settings["key_file"]), api_key_saved=bool(self.settings["api_key"]))
        task = self._demo_task if self.demo else (self.agent.state.task if self.agent else None)
        pending = self.agent._pending.as_dict() if self.agent and self.agent._pending else None
        if pending is not None:
            pending["signature"] = list(self.agent._pending.signature)
            pending["age_s"] = max(0, time.time() - self.agent._pending.created_at)
        events = list(self.events)[-250:]
        quality = {}
        for k, n in self.telemetry.get("nodes", {}).items():
            updated_age = time.monotonic() - self._readings_at.get(k, (None, time.monotonic()))[1]
            quality[k] = {"unchanged_for_s": round(updated_age, 1), "stale": bool((age is not None and age > 10) or updated_age > 15 or n.get("status") == "UNKNOWN"), "reading_delay_s": n.get("reading_delay_s")}
        return self.redact({"csrf": self.csrf, "demo": self.demo, "connection": {"status": self.connection, "role": role, "team": 3 if self.demo else getattr(getattr(self.api, "client", None), "team", None), "age_s": age, "error": self.last_error}, "settings": settings, "catalog": self.catalog, "telemetry": self.telemetry, "quality": quality, "alerts": self._alerts(), "messages": list(self.messages), "events": events, "history": list(self.history), "agent_chat": list(self.conversation.messages), "agent_chat_busy": self.conversation.busy, "agent": {"running": running, "mode": self.settings["mode"], "stats": self._demo_stats if self.demo else (self.agent.stats if self.agent else {}), "task": task, "pending": pending}, "policy": self._policy(), "server_time": iso()})

    async def dispatch(self, action: str, data: dict):
        # LLM waits must not hold the equipment/settings lock or delay Stop.
        if action == "agent/chat":
            return await self.conversation.ask(data.get("text", ""))
        async with self._mutation_lock:
            if action == "settings":
                if self.agent_task and not self.agent_task.done() or self._demo_agent:
                    raise WebError("Остановите агента перед изменением режима или подключения", 409)
                new = dict(self.settings)
                for key in {"url", "environment", "name", "model", "base_url", "mode", "dry_run", "control_enabled", "llm_enabled"}:
                    if key in data:
                        if key in {"dry_run", "control_enabled", "llm_enabled"} and not isinstance(data[key], bool):
                            raise WebError("Параметры переключателей должны быть true/false")
                        new[key] = data[key]
                for key in ("key", "api_key"):
                    if data.get(key):
                        new[key] = str(data[key]).strip()
                if new["mode"] not in {"normal", "auto", "yolo"}:
                    raise WebError("Неизвестный режим")
                if new["mode"] == "yolo" and not data.get("unsafe_ack"):
                    raise WebError("Экспериментальный режим требует явного подтверждения")
                if not self.demo and new["url"] and urlsplit(str(new["url"])).scheme not in {"ws", "wss"}:
                    raise WebError("Адрес Купола должен начинаться с ws:// или wss://")
                if len(str(new["name"])) > 80 or len(str(new["url"])) > 1000:
                    raise WebError("Слишком длинное значение настройки")
                if new["environment"] not in {"test", "real", "custom"}:
                    raise WebError("Неизвестный тип подключения")
                if new["url"] != self.settings["url"]:
                    # Never reuse a previous complex's credentials on a new host.
                    new["key"] = str(data.get("key", "")).strip()
                    new["key_file"] = ""
                    new["control_enabled"] = False
                    new["mode"] = "normal"
                    self.telemetry = {"nodes": {}, "game_time": {}}
                    self.catalog = []
                    self.history.clear()
                    self.messages.clear()
                    self._readings_at.clear()
                    self.conversation.messages.clear()
                if self.api:
                    await self.api.close()
                    self.api = None
                self.settings = new
                self.want_connection = False
                if not self.demo:
                    self.connection = "disconnected"
                self.event("decision", situation="Настройки обновлены", source="web")
                return {"ok": True, "message": "Настройки применены. Ключи хранятся только в памяти сервера."}
            if action == "connect":
                if self.demo:
                    return {"ok": True, "message": "Демонстрационный стенд работает локально"}
                if not self.settings["url"]:
                    raise WebError("Укажите адрес Купола в настройках")
                self.want_connection = True
                if not self.api:
                    await self._open()
                return {"ok": True, "message": "Подключение установлено"}
            if action == "disconnect":
                self.want_connection = False
                await self._stop_agent()
                if self.api:
                    await self.api.close()
                self.api = None
                self.connection = "demo" if self.demo else "disconnected"
                return {"ok": True}
            if action == "agent/start":
                if self.demo:
                    self._demo_agent = True
                    self.event("start", agent="Демонстрационный агент", mode=self.settings["mode"])
                    return {"ok": True}
                if not self.api:
                    raise WebError("Сначала подключитесь к Куполу", 409)
                if self.agent_task and not self.agent_task.done():
                    return {"ok": True}
                await self._start_agent()
                return {"ok": True}
            if action == "agent/stop":
                await self._stop_agent()
                return {"ok": True, "message": "Программа агента остановлена. Состояние оборудования не изменялось."}
            if action == "proposal":
                if not self.agent or not self.agent._pending:
                    raise WebError("Нет ожидающего предложения", 409)
                signature = data.get("signature")
                if signature != list(self.agent._pending.signature):
                    raise WebError("Предложение изменилось. Обновите страницу.", 409)
                if data.get("reply") not in {"confirm", "cancel", "force"}:
                    raise WebError("Неизвестный ответ")
                return await self.agent.web_reply(data["reply"], signature)
            if action == "chat":
                text = str(data.get("text", "")).strip()
                if not text or len(text) > 4000:
                    raise WebError("Сообщение должно содержать от 1 до 4000 символов")
                if self.settings["dry_run"]:
                    raise WebError("В режиме имитации сообщения не отправляются", 409)
                to = str(data.get("to", "operator"))[:64]
                if self.demo:
                    message = {"id": len(self.messages) + 1, "time": time.time(), "sender": {"role": "team", "name": self.settings["name"]}, "to": to, "text": text}
                    self.messages.append(message)
                else:
                    self._require_connected()
                    message = await self.api.post_message(text, to=to, name=self.settings["name"])
                    self._chat_at = 0
                self.event("message", direction="out", text=text, to=to)
                return {"ok": True, "message": message}
            if action == "ping":
                started = time.perf_counter()
                if not self.demo:
                    self._require_connected()
                    await self.api.ping()
                return {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
            if action == "query":
                node_id, query = str(data.get("node_id", "")), str(data.get("query", ""))
                params = data.get("params", {})
                if not isinstance(params, dict):
                    raise WebError("Параметры запроса должны быть JSON-объектом")
                cat = next((n for n in self.catalog if n["node_id"] == node_id), None)
                if not cat or query not in cat.get("queries", []):
                    raise WebError("Запрос недоступен в актуальном каталоге", 403)
                if self.demo:
                    files = self.telemetry["nodes"].get(node_id, {}).get("files", [])
                    result = [{"name": f, "size_bytes": 120000, "valid": True} for f in files] if query == "list_files" else list(self.messages)
                else:
                    self._require_connected()
                    result = await self.api.query_device(node_id, query, params)
                self.event("action", request={"type": "query", "node_id": node_id, "query": query, "params": params}, response={"success": True})
                return {"ok": True, "result": result}
            if action == "control":
                return await self._control(data)
            if action == "knowledge":
                return self.search(str(data.get("query", "")))
            raise WebError("Неизвестная операция", 404)

    def _require_connected(self):
        if not self.api or self.connection != "connected":
            raise WebError("Нет подключения к Куполу", 409)

    async def _control(self, data):
        if self._demo_agent or self.agent_task and not self.agent_task.done():
            raise WebError("Агент управляет комплексом. Остановите программу агента перед ручной командой.", 409)
        node_id, action = str(data.get("node_id", "")), str(data.get("action", ""))
        node = next((n for n in self.catalog if n["node_id"] == node_id), None)
        if not node or action not in node.get("controls", []):
            raise WebError("Команда недоступна в актуальном каталоге", 403)
        allowed, reason = self._permission(node_id, action, data.get("value"))
        if not allowed:
            self.event("action_blocked", reason=reason, request=data)
            raise WebError(reason, 403)
        if self.settings["dry_run"]:
            self.event("action", request=data, response={"dry_run": True})
            return {"ok": True, "dry_run": True, "message": "Имитация: команда не отправлялась"}
        if self.demo:
            result = self._demo_control(node_id, action, data.get("value"))
            self._demo_stats["actions"] += 1
            self.event("action", request=data, response={"success": True}, source="demo")
            self.event("verification", ok=True, expected=action, observed=result, source="demo")
            return {"ok": True, "accepted": True, "verified": True, "result": result}
        self._require_connected()
        if self.api.client.role != "team":
            raise WebError("Управление доступно только активной команде", 403)
        config = self._make_config()
        from dome_agent.rules import ActionSpec, NO_VALUE
        from dome_agent.control import DomeTool, ModelCommand
        from dome_agent.verification import expected_effect
        spec = ActionSpec(key="web:" + secrets.token_hex(5), node_id=node_id, action=action, value=data.get("value", NO_VALUE), reason="Команда из локальной веб-панели")
        checked = self.gate_class(config.policy_file, control_enabled=config.control_enabled, yolo=config.yolo).check(spec)
        if not checked.allowed:
            raise WebError(checked.reason, 403)
        ref = self.audit.write("decision", situation="Ручная команда из панели", cause="Запрос пользователя", confidence=1, evidence=[], plan=[action], source="rules")
        result = await DomeTool(self.api).execute(ModelCommand.from_action(spec), verify=True, force=False)
        action_ref = self.audit.write("action", request=spec.request(), response={"success": bool(result.get("ok")), "error": result.get("error")}, ref=ref)
        payload = result.get("result", {})
        after = payload.get("after") or {}
        effect = expected_effect(node_id, action, data.get("value"), has_value="value" in data)
        verified = effect.check(after) is True
        observed = effect.observe(after)
        self.audit.write("verification", ref=action_ref, expected=effect.description, observed=observed, ok=verified)
        self.event("action", request=data, response={"success": bool(result.get("ok")), "verified": verified}, error=result.get("error"))
        return {"ok": bool(result.get("ok")), "accepted": bool(result.get("ok")), "verified": verified, "result": result, "message": "Результат подтверждён" if verified else "Ответ получен; физический результат пока не подтверждён"}

    def _demo_control(self, node_id, action, value):
        n = self.telemetry["nodes"][node_id]
        if action in {"line_on", "line_off", "reset_protection"}:
            num = int(value.get("line")) if isinstance(value, dict) else int(value)
            line = next((x for x in n.get("lines", []) if x.get("line") == num), None)
            if line is None:
                raise WebError("Линия не найдена")
            line["state"] = "OFF" if action == "line_off" else "ON"
            if action == "reset_protection":
                line["protection_state"] = "ARMED"
            return line
        if action == "start_job":
            filename = value.get("file_name") if isinstance(value, dict) else value
            if filename not in n.get("files", []):
                raise WebError("Файла нет на устройстве")
            if n.get("state") in {"PRINTING", "RUNNING"}:
                raise WebError("Устройство уже выполняет задание", 409)
            if node_id == "printer_3d_01" and n.get("bed_clear") is not True:
                raise WebError("Стол принтера занят", 409)
            n.update(file_name=filename, state="PRINTING" if node_id == "printer_3d_01" else "RUNNING", progress_pct=0, paused=False)
        elif action == "resume":
            if n.get("hold_reason"):
                raise WebError("Причина удержания ещё не устранена", 409)
            n.update(state="PRINTING" if node_id == "printer_3d_01" else "RUNNING", paused=False)
        elif action == "home":
            if n.get("state") in {"PRINTING", "RUNNING"}:
                raise WebError("Нельзя выполнять home во время задания", 409)
            n["homed"] = True
        elif action in {"turn_on", "turn_off"}:
            n.update(state="ON" if action == "turn_on" else "OFF", enabled=action == "turn_on")
        elif action == "set_fan":
            n.update(fan_mode=value, fan_state=value, fan_current_a=.25 if value == "ON" else 0)
        elif action == "set_mode":
            n["mode"] = value
        elif action == "reset_error":
            n.update(error=False, error_code=None)
        elif action == "reconnect":
            n.update(online=True, status="OK")
        else:
            raise WebError("Эта команда доступна после подключения к настоящему API; в демо её эффект не моделируется", 422)
        return copy.deepcopy(n)

    async def _start_agent(self):
        config = self._make_config()
        from .integration import BrowserAgent, BrowserLog
        log = BrowserLog(config.log_dir, secrets=config.secrets(), callback=self.events.append)
        self.agent = BrowserAgent(config, api=self.api, log=log)
        async def run():
            try:
                await self.agent.run()
            except Exception as exc:
                self.event("error", where="agent", error=str(exc))
            finally:
                self.api = None
                self._retry_at = time.monotonic() + 2
        self.agent_task = asyncio.create_task(run(), name="web-owned-agent")

    async def _stop_agent(self):
        if self.demo:
            self._demo_agent = False
            self.event("stop", reason="Остановлен демонстрационный агент")
        if self.agent_task and not self.agent_task.done():
            self.agent.request_stop("web_user")
            try:
                await asyncio.wait_for(asyncio.shield(self.agent_task), timeout=10)
            except asyncio.TimeoutError:
                self.agent_task.cancel()
                await asyncio.gather(self.agent_task, return_exceptions=True)

    def search(self, query: str):
        query = query.strip()
        if not 2 <= len(query) <= 500:
            raise WebError("Введите вопрос от 2 до 500 символов")
        result = self.knowledge.query(query)
        if result is not None:
            return {"ok": True, "engine": "DomeKB", "answer": result}
        # A missing index is a reported state, never a silent weaker answer.
        return {"ok": False, "engine": "DomeKB", "error": self.knowledge.error or "База знаний недоступна", "results": []}
    def log_download(self):
        # A single exported session; no arbitrary filesystem path from the browser.
        return "\n".join(json.dumps(self.redact(x), ensure_ascii=False, default=str) for x in list(self.events)) + "\n"
