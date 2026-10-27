"""Разбор markdown-источников корпуса.

Корпус написан вручную и неоднороден: где-то поля оформлены как
``**Поле.** значение``, где-то ``**Поле:** значение``; таблицы соседствуют
с чек-листами. Модуль приводит это к единым примитивам, чтобы остальные
части конвейера не зависели от стиля конкретного документа.
"""
from __future__ import annotations

import re
from pathlib import Path

FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)
HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
FIELD_RE = re.compile(r"^\*\*([^*\n]{1,80}?)\*\*\s*[:.]?\s*(.*)$")
NUMBERED_RE = re.compile(r"^\s*(\d{1,2})[.)]\s+(.+)$")
BULLET_RE = re.compile(r"^\s*[-*]\s+(.+)$")
CHECKBOX_RE = re.compile(r"^\s*[-*]\s+\[[ xX]\]\s+(.+)$")
PLACEHOLDER_RE = re.compile(r"\[ЗАПОЛНИТЬ[^\]]*\]")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def strip_frontmatter(text: str) -> str:
    return FRONTMATTER_RE.sub("", text, count=1)


def norm(text: str) -> str:
    """Нормализация для сравнений: регистр, ё, пробелы, пунктуация по краям."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_sections(text: str) -> list[dict]:
    """Плоский список разделов с путём заголовков.

    Каждый элемент: ``{level, title, path, line, body}``.
    H1 намеренно не считается разделом: заголовок документа не несёт смысла
    для поиска, а путь строится начиная с H2.
    """
    lines = text.splitlines()
    out: list[dict] = []
    stack: list[tuple[int, str]] = []
    cur: dict | None = None

    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if not m:
            continue
        if cur is not None:
            cur["body"] = "\n".join(lines[cur["start"]:i]).strip()
            out.append(cur)
        level = len(m.group(1))
        title = m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        path = [t for _, t in stack] + [title]
        cur = {
            "level": level,
            "title": title,
            "path": path,
            "line": i + 1,
            "start": i + 1,
            "body": "",
        }
        stack.append((level, title))

    if cur is not None:
        cur["body"] = "\n".join(lines[cur["start"]:]).strip()
        out.append(cur)
    return out


def parse_fields(body: str) -> dict[str, str]:
    """Поля вида ``**Что происходит.** текст`` → словарь.

    Значение продолжается на следующих строках, пока не начнётся новое поле.
    """
    fields: dict[str, str] = {}
    cur: str | None = None
    for raw in body.splitlines():
        line = raw.strip()
        m = FIELD_RE.match(line)
        if m:
            label = m.group(1).strip().rstrip(".:").strip()
            cur = label
            fields[label] = m.group(2).strip()
        elif cur and line and not line.startswith(("-", "*", "|", "#")):
            fields[cur] = (fields[cur] + " " + line).strip()
    return fields


def parse_numbered(body: str) -> list[str]:
    items: list[str] = []
    for raw in body.splitlines():
        m = NUMBERED_RE.match(raw)
        if m:
            items.append(m.group(2).strip())
        elif items and raw.startswith(("   ", "\t")) and raw.strip():
            items[-1] = (items[-1] + " " + raw.strip()).strip()
    return items


def parse_bullets(body: str) -> list[str]:
    out = []
    for raw in body.splitlines():
        m = CHECKBOX_RE.match(raw) or BULLET_RE.match(raw)
        if m:
            out.append(m.group(1).strip())
    return out


def parse_tables(text: str) -> list[dict]:
    """Все markdown-таблицы документа: ``{header, rows, line}``."""
    tables: list[dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        block = []
        start = i
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            block.append(lines[i].strip())
            i += 1
        rows = []
        for row in block:
            cells = [c.strip() for c in row.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            rows.append(cells)
        if len(rows) >= 2:
            tables.append({"header": rows[0], "rows": rows[1:], "line": start + 1})
    return tables


def placeholders(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(text)


def html_comment_gaps(text: str) -> list[str]:
    """Списки «что ещё нужно уточнить», спрятанные в HTML-комментариях."""
    out = []
    for block in HTML_COMMENT_RE.findall(text):
        items = [ln.strip(" -*\t") for ln in block.splitlines() if ln.strip(" -*\t")]
        out.extend(items)
    return out


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def first_sentences(text: str, n: int = 2, limit: int = 320) -> str:
    out = " ".join(sentences(text)[:n])
    # Markdown emphasis may be removed; API identifiers must survive intact.
    out = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", out)
    out = re.sub(r"[#*`|]", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out[:limit]


def slug(text: str, limit: int = 48) -> str:
    text = norm(text)
    text = re.sub(r"[^a-zа-я0-9]+", "-", text).strip("-")
    return text[:limit] or "item"
