"""Operator chat parsing and completion guards.

The operator speaks through ``meeting_01``.  Messages from other teams are
never treated as commands.  Parsing is deliberately conservative: when a
request is ambiguous the agent asks the operator instead of guessing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

FILE_RE = re.compile(r"([A-Za-zА-Яа-я0-9_.\-/]+\.(?:gcode|gco|nc|ngc|tap))", re.IGNORECASE)
LINE_RE = re.compile(r"(?:лини[яию]|line)\s*[№#]?\s*(\d+)", re.IGNORECASE)
QUOTED_RE = re.compile(r"[\"'«»]([^\"'«»]{3,120})[\"'«»]")
FILENAME_TOKEN_RE = re.compile(r"(?:файл|file)\s*[:\-]?\s*([A-Za-zА-Яа-я0-9_.\-/]+)", re.IGNORECASE)

PRINT_WORDS = ("напечата", "распечата", "печать", "принтер", "print", "3d")
CNC_WORDS = ("фрезер", "чпу", "cnc", "mill", "гравир", "плат")
POWER_WORDS = ("лини", "лайн", "line", "обесточ", "запитан", "щит")
ON_WORDS = ("включ", "подай", "запитан", "on", "восстанов")
OFF_WORDS = ("выключ", "отключ", "обесточ", "сними", "off")
REPORT_WORDS = ("проверь", "доложи", "статус", "отчёт", "отчет", "report", "что происходит", "осмотри")
MONITOR_WORDS = ("смена", "следи", "поддерживай", "работай", "дежурь", "shift", "monitor")
TASK_WORDS = ("задание", "task", "нужно", "надо", "прошу", "please")


def _lower(value: Any) -> str:
    return str(value or "").strip().casefold()


def _sender(raw: Mapping[str, Any]) -> dict[str, Any]:
    sender = raw.get("sender")
    return sender if isinstance(sender, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class ChatMessage:
    id: int
    text: str
    sender_role: str
    sender_name: str
    sender_team: int | None
    to: str | None
    time: float | None

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "ChatMessage":
        sender = _sender(raw)
        team_value = sender.get("team", raw.get("team"))
        team: int | None
        try:
            team = int(team_value) if team_value is not None else None
        except (TypeError, ValueError):
            team = None
        return cls(
            id=_as_int(raw.get("id")),
            text=str(raw.get("text") or "").strip(),
            sender_role=_lower(sender.get("role") or raw.get("role")),
            sender_name=str(sender.get("name") or raw.get("name") or "").strip(),
            sender_team=team,
            to=(str(raw.get("to")).strip() if raw.get("to") is not None else None),
            time=float(raw["time"]) if isinstance(raw.get("time"), (int, float)) else None,
        )

    @property
    def is_operator(self) -> bool:
        role = self.sender_role
        name = self.sender_name.casefold()
        return role == "operator" or "оператор" in name or role == "admin"

    def is_for_us(self, agent_name: str = "team3-agent", team: int = 3) -> bool:
        target = _lower(self.to)
        if not target or target in {"all", "team", "teams", "everyone"}:
            return True
        if target in {
            _lower(agent_name),
            f"team{team}",
            f"team_{team}",
            f"team {team}",
            "agent",
        }:
            return True
        # Chat is shared between teams: never consume a command addressed to
        # another team or another named agent.
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "time": self.time,
            "sender": {
                "role": self.sender_role,
                "team": self.sender_team,
                "name": self.sender_name,
            },
            "to": self.to,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class TaskItem:
    kind: str  # print | cnc | line_on | line_off | report | inspect | message
    text: str
    file: str | None = None
    line: int | None = None

    @property
    def actionable(self) -> bool:
        return self.kind in {"print", "cnc", "line_on", "line_off"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "file": self.file,
            "line": self.line,
            "actionable": self.actionable,
        }


@dataclass(frozen=True, slots=True)
class OperatorTask:
    message_id: int
    text: str
    items: tuple[TaskItem, ...]
    kind: str
    created_at: float
    sender: str = "operator"

    @property
    def actionable(self) -> bool:
        return any(item.actionable for item in self.items)

    @property
    def needs_info(self) -> bool:
        return any(
            item.kind in {"print", "cnc"} and not item.file
            or item.kind in {"line_on", "line_off"} and item.line is None
            for item in self.items
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "text": self.text,
            "kind": self.kind,
            "created_at": self.created_at,
            "sender": self.sender,
            "items": [item.as_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OperatorTask | None":
        if not raw:
            return None
        try:
            message_id = int(raw.get("message_id") or raw.get("id") or 0)
        except (TypeError, ValueError):
            message_id = 0
        items: list[TaskItem] = []
        for item in raw.get("items") or []:
            if not isinstance(item, dict):
                continue
            try:
                line = int(item["line"]) if item.get("line") is not None else None
            except (TypeError, ValueError):
                line = None
            items.append(
                TaskItem(
                    kind=str(item.get("kind") or "message"),
                    text=str(item.get("text") or ""),
                    file=(str(item["file"]) if item.get("file") else None),
                    line=line,
                )
            )
        if not message_id and not items:
            return None
        return cls(
            message_id=message_id,
            text=str(raw.get("text") or ""),
            items=tuple(items),
            kind=str(raw.get("kind") or "chat"),
            created_at=float(raw.get("created_at") or 0.0),
            sender=str(raw.get("sender") or "operator"),
        )


def split_clauses(text: str) -> list[str]:
    """Split a shift task into independently actionable clauses."""
    normalized = re.sub(r"[ \t]+", " ", text.replace("•", "\n"))
    parts = re.split(r"(?:\n+|;|,|\s+(?:и|затем|потом|also|then)\s+)", normalized, flags=re.IGNORECASE)
    return [part.strip(" .-—") for part in parts if part.strip(" .-—")]


def extract_file(clause: str) -> str | None:
    match = FILE_RE.search(clause)
    if match:
        return match.group(1).replace("\\", "/").rsplit("/", 1)[-1]
    match = QUOTED_RE.search(clause)
    if match:
        candidate = match.group(1).strip()
        if "." in candidate:
            return candidate.rsplit("/", 1)[-1]
    match = FILENAME_TOKEN_RE.search(clause)
    if match:
        candidate = match.group(1).strip(" .-—")
        if candidate and candidate.casefold() not in {"на", "для", "с", "the"}:
            return candidate
    return None


def extract_line(clause: str) -> int | None:
    match = LINE_RE.search(clause)
    if not match:
        return None
    try:
        number = int(match.group(1))
    except ValueError:
        return None
    return number if 1 <= number <= 8 else None


def _classify_clause(clause: str) -> TaskItem | None:
    value = clause.casefold()
    file = extract_file(clause)
    line = extract_line(clause)
    has_action_word = any(word in value for word in TASK_WORDS)
    if any(word in value for word in PRINT_WORDS):
        if file or any(word in value for word in ("напечата", "распечата", "печать", "print")):
            return TaskItem(kind="print", text=clause, file=file)
    if any(word in value for word in CNC_WORDS):
        if file or any(word in value for word in ("фрезер", "чпу", "cnc", "mill")):
            return TaskItem(kind="cnc", text=clause, file=file)
    if any(word in value for word in POWER_WORDS):
        if line is not None:
            if any(word in value for word in OFF_WORDS):
                return TaskItem(kind="line_off", text=clause, line=line)
            if any(word in value for word in ON_WORDS):
                return TaskItem(kind="line_on", text=clause, line=line)
            return TaskItem(kind="inspect", text=clause, line=line)
    if any(word in value for word in REPORT_WORDS):
        return TaskItem(kind="report", text=clause)
    if has_action_word and file:
        # Extension tells us the likely machine when the verb was omitted.
        suffix = file.casefold().rsplit(".", 1)[-1]
        if suffix in {"gcode", "gco"}:
            return TaskItem(kind="print", text=clause, file=file)
        if suffix in {"nc", "ngc", "tap"}:
            return TaskItem(kind="cnc", text=clause, file=file)
    return None


def parse_operator_task(message: ChatMessage | Mapping[str, Any]) -> OperatorTask | None:
    """Parse one operator message into zero or more actionable items."""
    if not isinstance(message, ChatMessage):
        message = ChatMessage.from_raw(message)
    if not message.text or not message.is_operator:
        return None
    value = message.text.casefold()
    items: list[TaskItem] = []
    for clause in split_clauses(message.text):
        item = _classify_clause(clause)
        if item is not None and item not in items:
            items.append(item)
    if not items and any(word in value for word in MONITOR_WORDS):
        items.append(TaskItem(kind="monitor", text=message.text))
    kind = items[0].kind if len(items) == 1 else ("multi" if items else "chat")
    if kind == "monitor":
        kind = "shift"
    return OperatorTask(
        message_id=message.id,
        text=message.text,
        items=tuple(items),
        kind=kind,
        created_at=time.time() if message.time is None else message.time,
        sender=message.sender_name or "operator",
    )


def new_operator_messages(
    messages: Iterable[Mapping[str, Any]],
    *,
    after_id: int,
    agent_name: str = "team3-agent",
    team: int = 3,
) -> list[ChatMessage]:
    result: list[ChatMessage] = []
    for raw in messages:
        if not isinstance(raw, Mapping):
            continue
        message = ChatMessage.from_raw(raw)
        if message.id <= after_id:
            continue
        if message.is_operator and message.is_for_us(agent_name=agent_name, team=team):
            result.append(message)
    result.sort(key=lambda item: item.id)
    return result


@dataclass(frozen=True, slots=True)
class CompletionCheck:
    ok: bool
    reason: str
    observed: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason, "observed": self.observed}


def check_print_completion(node: Mapping[str, Any]) -> CompletionCheck:
    state = _lower(node.get("state"))
    real_state = _lower(node.get("real_state") or node.get("state"))
    webhook = _lower(node.get("webhook_state"))
    progress = node.get("progress_pct")
    try:
        progress_value = float(progress) if progress is not None else None
    except (TypeError, ValueError):
        progress_value = None
    z_value = node.get("z_mm")
    try:
        z_mm = float(z_value) if z_value is not None else None
    except (TypeError, ValueError):
        z_mm = None
    observed = {
        "state": state,
        "real_state": real_state,
        "webhook_state": webhook,
        "progress_pct": progress_value,
        "z_mm": z_mm,
        "bed_clear": node.get("bed_clear"),
        "paused": node.get("paused"),
    }
    if webhook == "completed" and state not in {"completed", "idle"}:
        return CompletionCheck(False, "webhook says COMPLETED but state does not", observed)
    if state not in {"completed", "idle"} and real_state not in {"completed", "idle"}:
        return CompletionCheck(False, "printer is not finished", observed)
    if progress_value is not None and progress_value < 99.0:
        return CompletionCheck(False, "progress is below 99%", observed)
    if bool(node.get("paused")):
        return CompletionCheck(False, "printer is paused", observed)
    if z_mm is not None and z_mm <= 0 and progress_value is not None and progress_value < 100.0:
        return CompletionCheck(False, "Z is at zero before completion", observed)
    return CompletionCheck(True, "state, progress and Z agree", observed)


def check_cnc_completion(node: Mapping[str, Any]) -> CompletionCheck:
    state = _lower(node.get("state"))
    progress = node.get("progress_pct")
    try:
        progress_value = float(progress) if progress is not None else None
    except (TypeError, ValueError):
        progress_value = None
    observed = {
        "state": state,
        "progress_pct": progress_value,
        "spindle_rpm": node.get("spindle_rpm"),
        "paused": node.get("paused"),
        "hold_level": node.get("hold_level"),
        "quality_state": node.get("quality_state"),
        "part_defect": node.get("part_defect"),
    }
    if state not in {"idle", "completed"}:
        return CompletionCheck(False, "CNC is not finished", observed)
    if progress_value is not None and progress_value < 99.0:
        return CompletionCheck(False, "progress is below 99%", observed)
    if node.get("paused") or node.get("hold_level"):
        return CompletionCheck(False, "CNC is still held", observed)
    return CompletionCheck(True, "CNC state and progress agree", observed)
