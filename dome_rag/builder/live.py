"""Минимальный клиент живого API купола на стандартной библиотеке.

Протокол — WebSocket + JSON (см. ``01_API.md``). Штатный пример использует
``websockets``, но в этой среде пакета нет, а ставить зависимости нельзя.
Здесь реализован достаточный для агента синхронный клиент: авторизация,
чтение телеметрии, ``describe_nodes``, ``query``, ``control``.

Живые данные НЕ индексируются и НЕ попадают в базу знаний: они нужны только
для ответа на вопрос «что сейчас». После сессии остаётся журнал действий, а
не поток чисел.
"""
from __future__ import annotations

import base64
import hashlib
import itertools
import json
import os
import socket
import struct
import time

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WSError(RuntimeError):
    pass


class WebSocket:
    def __init__(self, host: str, port: int = 8765, path: str = "/", timeout: float = 15.0):
        self.host, self.port, self.path = host, port, path
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self._buf = b""

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise WSError("Сервер закрыл соединение при рукопожатии")
            head += chunk
        header, _, rest = head.partition(b"\r\n\r\n")
        self._buf = rest
        text = header.decode("latin-1")
        if "101" not in text.splitlines()[0]:
            raise WSError(f"Рукопожатие не удалось: {text.splitlines()[0]}")
        expected = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        if f"sec-websocket-accept: {expected}".lower() not in text.lower():
            raise WSError("Неверный Sec-WebSocket-Accept")

    def close(self) -> None:
        if self.sock:
            try:
                self._send_frame(0x8, b"")
            except OSError:
                pass
            self.sock.close()
            self.sock = None

    # -------------------------------------------------------------- кадры
    def _recv_exact(self, n: int) -> bytes:
        assert self.sock
        out = self._buf[:n]
        self._buf = self._buf[n:]
        while len(out) < n:
            chunk = self.sock.recv(n - len(out))
            if not chunk:
                raise WSError("Соединение закрыто")
            out += chunk
        return out

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        assert self.sock
        header = bytearray([0x80 | opcode])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = os.urandom(4)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode("utf-8"))

    def recv_text(self) -> str:
        assert self.sock
        fragments = b""
        while True:
            b1, b2 = self._recv_exact(2)
            fin = bool(b1 & 0x80)
            opcode = b1 & 0x0F
            masked = bool(b2 & 0x80)
            length = b2 & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if masked else b""
            payload = self._recv_exact(length)
            if masked:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

            if opcode == 0x8:
                raise WSError("Сервер закрыл сессию")
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode in (0x1, 0x0):
                fragments += payload
                if fin:
                    return fragments.decode("utf-8", errors="replace")

    # ------------------------------------------------------------- запросы
    def recv_json(self) -> dict:
        return json.loads(self.recv_text())


class DomeClient:
    def __init__(self, host: str | None = None, port: int | None = None,
                 key: str | None = None, path: str = "/", timeout: float = 15.0):
        self.host = host or os.environ.get("DOME_HOST", "")
        self.port = int(port or os.environ.get("DOME_PORT", "8765"))
        self.key = key or os.environ.get("DOME_KEY", "")
        self.ws = WebSocket(self.host, self.port, path, timeout)
        self.ids = itertools.count(1)
        self.telemetry: dict | None = None
        self.messages: list[dict] = []
        self.role = None
        self.team = None

    def connect(self) -> dict:
        if not self.host:
            raise WSError("Не задан адрес купола: переменная DOME_HOST (например, 192.168.8.10)")
        if not self.key:
            raise WSError("Не задан ключ команды: переменная DOME_KEY")
        self.ws.connect()
        self.ws.send_text(json.dumps({"type": "auth", "key": self.key}))
        auth = self._wait(lambda m: m.get("type") not in ("telemetry", "meeting_message"), limit=5)
        if not auth.get("success"):
            raise WSError(f"Авторизация отклонена: {auth.get('error')}")
        self.role = auth.get("role")
        self.team = auth.get("team")
        return auth

    def _pump(self) -> dict:
        msg = self.ws.recv_json()
        kind = msg.get("type")
        if kind == "telemetry":
            self.telemetry = msg.get("data")
        elif kind == "meeting_message":
            self.messages.append(msg.get("message") or {})
        return msg

    def _wait(self, predicate, limit: int = 200) -> dict:
        for _ in range(limit):
            msg = self._pump()
            if predicate(msg):
                return msg
        raise WSError("Ответ не получен")

    def request(self, **payload) -> dict:
        rid = str(next(self.ids))
        self.ws.send_text(json.dumps({**payload, "request_id": rid}))
        return self._wait(lambda m: str(m.get("request_id")) == rid)

    def describe_nodes(self) -> dict:
        return self.request(type="describe_nodes")

    def query(self, node_id: str, query: str, params: dict | None = None) -> dict:
        return self.request(type="query", node_id=node_id, query=query, params=params or {})

    def control(self, node_id: str, action: str, value=None) -> dict:
        payload = {"type": "control", "node_id": node_id, "action": action}
        if value is not None:
            payload["value"] = value
        return self.request(**payload)

    def say(self, text: str, to: str = "operator", name: str = "dome-kb") -> dict:
        return self.control("meeting_01", "post_message", {"text": text, "to": to, "name": name})

    def snapshot(self, wait_s: float = 10.0) -> dict:
        """Дождаться ближайшего пакета телеметрии и вернуть его."""
        deadline = time.time() + wait_s
        while self.telemetry is None:
            if time.time() > deadline:
                raise WSError("Телеметрия не пришла за отведённое время")
            self._pump()
        return self.telemetry

    def close(self) -> None:
        self.ws.close()
