"""Local HTTP server and CLI. No Node build or additional server dependency."""
from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .service import DashboardService, STATIC, WebError

ASSETS = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/styles.css": "styles.css", "/favicon.svg": "favicon.svg"}


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, service):
        self.service = service
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "DomePanel/1.0"

    def log_message(self, *args):
        pass

    def _host_allowed(self):
        try:
            parsed = urlsplit("http://" + self.headers.get("Host", ""))
            return parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port == self.server.server_port
        except ValueError:
            return False

    def _send(self, body, status=200, content_type="application/json; charset=utf-8", download=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="dome_web_session.jsonl"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if not self._host_allowed():
            return self._send({"error": "Недопустимый адрес панели"}, 403)
        route = urlsplit(self.path).path
        if route in ASSETS:
            p = STATIC / ASSETS[route]
            if not p.is_file():
                return self._send({"error": "Ресурс не найден"}, 404)
            types = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".html": "text/html; charset=utf-8", ".svg": "image/svg+xml"}
            return self._send(p.read_bytes(), content_type=types.get(p.suffix, "application/octet-stream"))
        if route == "/api/state":
            try:
                return self._send(self.server.service.call(self.server.service.snapshot(), timeout=5))
            except Exception as exc:
                return self._send({"error": self.server.service.redact(str(exc))}, 503)
        if route == "/api/logs/download":
            return self._send(self.server.service.log_download(), content_type="application/x-ndjson; charset=utf-8", download=True)
        self._send({"error": "Страница не найдена"}, 404)

    def do_POST(self):
        if not self._host_allowed():
            return self._send({"error": "Недопустимый адрес панели"}, 403)
        origin = self.headers.get("Origin")
        expected = "http://" + self.headers.get("Host", "")
        if origin and origin != expected:
            return self._send({"error": "Запрос с другого сайта запрещён"}, 403)
        if not secrets.compare_digest(self.headers.get("X-Dome-CSRF", ""), self.server.service.csrf):
            return self._send({"error": "Обновите страницу перед выполнением действия"}, 403)
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            return self._send({"error": "Ожидается JSON"}, 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 18 * 1024 * 1024:
                raise WebError("Недопустимый размер запроса", 413)
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise WebError("Ожидается JSON-объект")
            route = urlsplit(self.path).path
            if not route.startswith("/api/"):
                raise WebError("Операция не найдена", 404)
            result = self.server.service.call(self.server.service.dispatch(route[5:], data), timeout=45)
            self._send(result)
        except WebError as exc:
            self._send({"ok": False, "error": self.server.service.redact(str(exc))}, exc.status)
        except (ValueError, KeyError, TypeError) as exc:
            self._send({"ok": False, "error": self.server.service.redact(str(exc))}, 400)
        except Exception as exc:
            self._send({"ok": False, "error": self.server.service.redact(str(exc))}, 503)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Локальная веб-панель Купола")
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--demo", action="store_true", help="Локальные демонстрационные данные, без сетевых обращений")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--url", help="Адрес WebSocket Купола")
    parser.add_argument("--key-file", help="Файл ключа; содержимое не передаётся браузеру")
    parser.add_argument("--environment", choices=["test", "real", "custom"], help="Подпись источника: тестовый сервер, настоящий Купол или другой сервер")
    parser.add_argument("--connect", action="store_true", help="Подключиться в наблюдении при запуске")
    args = parser.parse_args(argv)
    root = Path.cwd()
    service = DashboardService(root, demo=args.demo, env_file=args.env_file)
    if args.url:
        service.settings["url"] = args.url
    if args.key_file:
        service.settings["key_file"] = args.key_file
        service.settings["key"] = ""
    if args.environment:
        service.settings["environment"] = args.environment
    if args.connect:
        # Opening a demo/live viewer never starts the agent or enables control.
        service.settings.update(control_enabled=False, mode="normal")
        service.want_connection = not args.demo
    service.start()
    server = DashboardHTTPServer((args.host, args.port), service)
    print(f"Купол: http://{args.host}:{server.server_port}", flush=True)
    print("Демонстрационные данные" if args.demo else "Подключение к API — в режиме наблюдения", flush=True)
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, stop)
        except (ValueError, OSError):
            pass
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close()
        service.close()
    return 0
