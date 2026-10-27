"""Извлечение текста из DOCX без внешних зависимостей.

Задача — не «вытащить весь XML», а сохранить структуру, по которой потом
работает нарезка:

* заголовки (``Heading1..6``) → ``##``/``###``;
* таблицы → markdown-таблицы (значения не должны разъезжаться по ячейкам);
* рисунки → явная пометка, чтобы документ «только из картинок» нельзя было
  считать прочитанным.

Если текста почти нет, а картинки есть, возвращается статус ``image_only``:
такой источник попадает в индекс, но помечается как неполный.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
IMAGE_RE = re.compile(r"\[РИСУНОК: ([^\]]+)\]")
MIN_TEXT_FOR_TEXT_DOC = 300


@dataclass(slots=True)
class Extracted:
    text: str
    images: list[str] = field(default_factory=list)
    tables: int = 0
    paragraphs: int = 0
    status: str = "ok"

    @property
    def has_text(self) -> bool:
        return len(self.text.strip()) >= 40


def _text_of(node: ET.Element) -> str:
    parts: list[str] = []
    for child in node.iter():
        tag = child.tag
        if tag == W + "t":
            parts.append(child.text or "")
        elif tag == W + "tab":
            parts.append(" ")
        elif tag in (W + "br", W + "cr"):
            parts.append("\n")
    text = "".join(parts)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _style(node: ET.Element) -> str:
    for child in node.iter(W + "pStyle"):
        return (child.get(W + "val") or "").strip()
    return ""


def _heading_level(style: str) -> int:
    low = style.casefold()
    if low in {"title", "заголовок"}:
        return 1
    m = re.match(r"(?:heading|заголовок)\s*(\d)", low)
    if m:
        return max(1, min(6, int(m.group(1))))
    return 0


def _cell_text(cell: ET.Element) -> str:
    chunks = [_text_of(p) for p in cell.iter(W + "p")]
    value = " ".join(chunk for chunk in chunks if chunk)
    return value.replace("|", "/").strip()


def _table_markdown(table: ET.Element) -> str:
    rows: list[list[str]] = []
    for row in table.iter(W + "tr"):
        cells = [_cell_text(cell) for cell in row.findall(W + "tc")]
        if any(cell for cell in cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract(path: Path) -> Extracted:
    path = Path(path)
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"не удалось открыть DOCX {path.name}: {exc}") from exc

    with archive:
        names = archive.namelist()
        images = sorted(
            name for name in names if name.startswith("word/media/")
        )
        try:
            xml = archive.read("word/document.xml")
        except KeyError as exc:
            raise ValueError(f"DOCX {path.name} не содержит word/document.xml") from exc

    root = ET.fromstring(xml)
    body = root.find(W + "body")
    if body is None:
        return Extracted(text="", images=images, status="image_only" if images else "ok")

    blocks: list[str] = []
    image_no = 0
    tables = 0
    paragraphs = 0
    for child in list(body):
        if child.tag == W + "p":
            text = _text_of(child)
            drawing = list(child.iter(W + "drawing")) or list(child.iter(W + "pict"))
            if drawing:
                image_no += 1
                name = images[image_no - 1].rsplit("/", 1)[-1] if image_no <= len(images) else f"image{image_no}"
                blocks.append(f"[РИСУНОК: {name}]")
            if not text:
                continue
            paragraphs += 1
            level = _heading_level(_style(child))
            if level:
                blocks.append("#" * level + " " + text)
            else:
                blocks.append(text)
        elif child.tag == W + "tbl":
            table = _table_markdown(child)
            if table:
                tables += 1
                blocks.append(table)

    text = "\n\n".join(block for block in blocks if block.strip())
    status = "ok"
    if len(text.strip()) < MIN_TEXT_FOR_TEXT_DOC and images:
        status = "image_only"
    return Extracted(text=text, images=images, tables=tables, paragraphs=paragraphs, status=status)


def to_markdown(path: Path) -> tuple[str, Extracted]:
    """DOCX → markdown с шапкой-провенансом."""
    path = Path(path)
    extracted = extract(path)
    title = path.stem
    header = [
        f"# {title}",
        "",
        f"> Источник: `{path.name}`. Извлечено автоматически. "
        f"Рисунков: {len(extracted.images)}, таблиц: {extracted.tables}.",
    ]
    if extracted.status == "image_only":
        header.append(
            "> ВНИМАНИЕ: документ почти не содержит текстового слоя — основное "
            "содержание в изображениях. Без OCR это неполный источник."
        )
    header.append("")
    return "\n".join(header) + extracted.text, extracted
