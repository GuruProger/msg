"""Deterministic line-control requests written in the panel chat.

The advisory LLM is never allowed to turn free text into a command.  This
module only recognises a narrow, explicit pattern (on/off + a panel line) and
returns the exact action to confirm.  Anything ambiguous is reported back to
the operator instead of being guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

_LINE_RE = re.compile(
    r"(?:лини[яию]|линия|line|[лl]\.?)\s*[№#]?\s*(\d+)",
    re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(
    r"^(?:как|почему|зачем|что|где|когда|можно|стоит|нужно\s+ли|какая|какой|какие)\b",
    re.IGNORECASE,
)
_ON_WORDS = ("включ", "подай", "запитан", "восстанов", "on")
_OFF_WORDS = ("выключ", "отключ", "обесточ", "сними", "off")
_POWER_WORDS = ("освещ", "свет", "лини", "лайн", "щит", "авари", "line")
_GENERIC_TOKENS = {
    "линия",
    "линии",
    "линию",
    "освещение",
    "свет",
    "щит",
    "щиток",
}
_WORD_RE = re.compile(r"[a-zа-яё0-9]+")
_CONFIRM_WORDS = {"да", "yes", "ок", "ok", "подтверждаю", "выполняй", "подтвердить"}
_CANCEL_WORDS = {"нет", "no", "отмена", "отменить", "не надо", "стоп", "cancel"}


def normalize(value: Any) -> str:
    """Case-fold and strip punctuation so names and requests compare evenly."""
    text = str(value or "").casefold().replace("ё", "е")
    return " ".join(_WORD_RE.findall(text))


def looks_like_line_command(text: str) -> bool:
    """Return True only for an imperative on/off request about a panel line."""
    value = normalize(text)
    if not value or _QUESTION_START_RE.search(value):
        return False
    has_on = any(word in value for word in _ON_WORDS)
    has_off = any(word in value for word in _OFF_WORDS)
    has_power = any(word in value for word in _POWER_WORDS)
    return (has_on or has_off) and has_power


def _bare(text: str) -> str:
    return normalize(text).strip(" .!?;:")


def is_confirmation(text: str) -> bool:
    """Only an explicit, standalone confirmation may execute a proposal."""
    return _bare(text) in _CONFIRM_WORDS


def is_cancellation(text: str) -> bool:
    return _bare(text) in _CANCEL_WORDS


def _token_hit(token: str, words: set[str]) -> bool:
    if token in words:
        return True
    # "аварийку" should still find "аварийное" without a huge synonym table.
    if len(token) >= 5:
        stem = token[:5]
        return any(len(word) >= 5 and word.startswith(stem) for word in words)
    return False


def _line_number(line: Mapping[str, Any]) -> int | None:
    try:
        number = int(line.get("line"))
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= 8 else None


@dataclass(frozen=True, slots=True)
class LineCommand:
    action: str
    line: int | None
    label: str = ""
    reason: str = ""
    candidates: tuple[str, ...] = ()


def resolve_line_command(
    text: str,
    lines: Iterable[Mapping[str, Any]],
) -> LineCommand | None:
    """Resolve a recognised request against the live panel line names.

    ``LineCommand.line`` is ``None`` when the request is ambiguous or the
    named line does not exist; ``reason`` and ``candidates`` explain that to
    the operator.  No action is ever guessed.
    """
    if not looks_like_line_command(text):
        return None
    value = normalize(text)
    action = "line_off" if any(word in value for word in _OFF_WORDS) else "line_on"
    rows: list[tuple[int, Mapping[str, Any]]] = []
    for row in lines:
        if not isinstance(row, Mapping):
            continue
        number = _line_number(row)
        if number is not None:
            rows.append((number, row))
    by_number = {number: row for number, row in rows}

    explicit = _LINE_RE.search(str(text))
    if explicit:
        number = int(explicit.group(1))
        row = by_number.get(number)
        if row is None:
            return LineCommand(
                action,
                None,
                reason=f"Линии {number} нет в живом каталоге щитка.",
                candidates=tuple(str(r.get("name") or f"линия {n}") for n, r in rows),
            )
        return LineCommand(action, number, str(row.get("name") or f"линия {number}"))

    words = set(value.split())
    scored: list[tuple[int, int, str]] = []
    for number, row in rows:
        name = normalize(row.get("name"))
        if not name:
            continue
        if name in value:
            score = 100
        else:
            tokens = [
                token
                for token in name.split()
                if len(token) >= 4 and token not in _GENERIC_TOKENS
            ]
            hits = sum(1 for token in tokens if _token_hit(token, words))
            score = hits * 10
        if score:
            scored.append((score, number, str(row.get("name") or f"линия {number}")))
    if not scored:
        return LineCommand(
            action,
            None,
            reason="Не удалось однозначно определить линию; уточните номер.",
            candidates=tuple(str(r.get("name") or f"линия {n}") for n, r in rows),
        )
    best = max(score for score, _, _ in scored)
    winners = [(number, name) for score, number, name in scored if score == best]
    if len(winners) != 1:
        return LineCommand(
            action,
            None,
            reason="Название подходит к нескольким линиям; уточните номер.",
            candidates=tuple(name for _, name in winners),
        )
    number, name = winners[0]
    return LineCommand(action, number, name)
