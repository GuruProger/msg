"""Нарезка корпуса на чанки с метаданными.

Правила:
* единица чанка — терминальный раздел (``###`` без потомков) либо ``##`` без ``###``;
* таблицы и нумерованные шаги не разрываются;
* слишком длинные разделы (>2200 символов) делятся по абзацам;
* почти одинаковые чанки (API-документы дублируют knowledge/) помечаются
  ``is_duplicate_of`` и исключаются из выдачи.
"""
from __future__ import annotations

import json
import re

from . import docx, md, vectors
from .paths import CHUNKS, SOURCES, knowledge_docs, rel, sources

FAULT_ID_RE = re.compile(r"^(\d+)\.(\d+)\.\s+(.+)$")
DIAG_TITLE_RE = re.compile(r"^Ситуация\s*(\d+)[.:]?\s*(.*)$")
PART_SPLIT = "# Часть 2. Глоссарий"

DOC_TYPE = {
    "00_assistant.md": "persona",
    "01_legend.md": "narrative",
    "02_station.md": "equipment",
    "03_energy.md": "system",
    "04_telemetry.md": "system",
    "05_production.md": "system",
    "06_microclimate.md": "system",
    "07_rules.md": "rule",
    "08_safety.md": "safety",
    "09_faults.md": "fault",
    "10_procedures.md": "procedure",
    "11_reference.md": "faq",
    "12_history.md": "history",
    "00_Краткая_справка_API.md": "api",
    "01_API.md": "api",
    "dome_sandbox_nodes.md": "registry",
    "02_Задание_и_оценка.md": "task",
    "03_Требования_к_агенту.md": "agent_spec",
    "04_Как_создаётся_агент.md": "overview",
}

# DOCX не имеют стабильных имён-констант, поэтому тип берётся из авторитета.
DOCX_TYPE = {
    "task": "task",
    "theory": "theory",
    "scenario": "scenario",
    "prep": "prep",
    "history": "history",
    "regulation": "rule",
    "reference": "reference",
}

PRIORITY = {
    "00_Краткая_справка_API.md": 5,
    "01_API.md": 5,
    "03_Требования_к_агенту.md": 5,
    "knowledge": 4,
    "dome_sandbox_nodes.md": 3,
}


def _priority(doc_name: str, is_knowledge: bool) -> int:
    if doc_name in PRIORITY:
        return PRIORITY[doc_name]
    return 4 if is_knowledge else 3


def _refine_type(base: str, title: str, body: str) -> str:
    low = (title + " " + body[:200]).lower()
    if title.lower().startswith("ситуация"):
        return "diagnostic"
    if base in ("system", "equipment"):
        if "безопасност" in title.lower():
            return "safety"
        if "ограничени" in title.lower() or "не полностью" in title.lower():
            return "limits"
    if base == "history" and "неисправност" in title.lower():
        return "history_fault"
    return base


class AliasIndex:
    """Сопоставление упоминаний в тексте с узлами и короткими терминами.

    Правила, которые важны для качества:

    * длинное совпадение выигрывает у короткого и «маскирует» его
      (``вытяжная вентиляция`` не должна дополнительно давать
      ``вентиляция → приточная``);
    * короткие алиасы сравниваются по границе слова (``доступ`` не
      срабатывает внутри ``доступные``);
    * падежные формы ловятся по основам, но только для слов/фраз длиной
      от пяти символов — иначе ``сеть`` начинала бы цепляться к ``сетевой``.
    """

    def __init__(self, aliases: dict[str, str], nodes: list[dict]):
        self.map: dict[str, str] = {}
        for alias, node_id in aliases.items():
            key = md.norm(alias)
            if key:
                self.map[key] = node_id
        for n in nodes:
            for alias in [n["node_id"], n.get("code_name")] + list(n.get("aliases") or []):
                if alias:
                    # Явный алиас из curated-файла важнее автоалиаса узла.
                    self.map.setdefault(md.norm(alias), n["node_id"])
        # длинные — первыми: они маскируют вложенные короткие
        self._aliases = sorted(self.map, key=lambda a: (-len(a), a))
        self._exact = {
            alias: re.compile(
                rf"(?<![a-zа-я0-9]){re.escape(alias)}(?![a-zа-я0-9])"
            )
            for alias in self._aliases
        }
        self._stems = {
            alias: sorted({s for s in vectors.stems(alias) if len(s) >= 3})
            for alias in self._aliases
        }

    @staticmethod
    def _overlaps(spans: list[tuple[int, int]], start: int, end: int) -> bool:
        return any(start < right and left < end for left, right in spans)

    def detect(self, text: str) -> list[str]:
        low = md.norm(text)
        if not low:
            return []
        found: list[str] = []
        spans: list[tuple[int, int]] = []
        claimed_stems: list[set[str]] = []
        for alias in self._aliases:
            pattern = self._exact[alias]
            for match in pattern.finditer(low):
                start, end = match.span()
                if self._overlaps(spans, start, end):
                    continue
                spans.append((start, end))
                found.append(self.map[alias])
                alias_stems = set(self._stems[alias])
                if alias_stems:
                    claimed_stems.append(alias_stems)
                break

        # Падежи: «солнечной генерации» → «солнечная генерация».
        q_stems = vectors.stems(low)
        for alias in self._aliases:
            if len(alias) < 5:
                continue
            alias_stems = set(self._stems[alias])
            if not alias_stems or not alias_stems <= q_stems:
                continue
            # Вложенный алиас уже покрыт более длинной фразой:
            # «вытяжная вентиляция» не должна добавлять «приточную».
            if any(alias_stems <= claimed for claimed in claimed_stems):
                continue
            node_id = self.map[alias]
            if node_id not in found:
                found.append(node_id)

        seen: set[str] = set()
        out: list[str] = []
        for nid in found:
            if nid not in seen:
                seen.add(nid)
                out.append(nid)
        return out[:8]


def _split_long(body: str, limit: int = 1600) -> list[str]:
    if len(body) <= limit * 1.4:
        return [body]
    parts, cur = [], ""
    for para in body.split("\n\n"):
        if cur and len(cur) + len(para) > limit:
            parts.append(cur.strip())
            cur = para
        else:
            cur = (cur + "\n\n" + para).strip()
    if cur:
        parts.append(cur.strip())
    return parts or [body]


def _doc_type(source) -> str:
    name = source.path.name
    if name in DOC_TYPE:
        return DOC_TYPE[name]
    if source.kind == "docx":
        return DOCX_TYPE.get(source.authority, "theory")
    return "reference"


def _status_for(ctype: str, authority: str, ph: list[str], source) -> str:
    if source.kind == "docx" and source.status == "image_only":
        return "incomplete_source"
    if ph:
        return "placeholder_partial"
    if ctype == "fault":
        return "catalog"
    if ctype.startswith("history") or authority == "history":
        return "history"
    if authority in {"theory", "scenario", "prep"}:
        return "theory"
    return "normative"


def _sections(text: str, title: str) -> list[dict]:
    """Разделы документа. Для DOCX без заголовков — один условный раздел."""
    sections = md.split_sections(text)
    if sections:
        return sections
    body = text.strip()
    if not body:
        return []
    return [{"level": 2, "title": title, "path": [title], "line": 1, "body": body}]


def build_chunks() -> list[dict]:
    aliases = json.loads((CHUNKS.parent / "data" / "curated" / "aliases.json").read_text(encoding="utf-8"))
    nodes = json.loads((CHUNKS.parent / "data" / "nodes.json").read_text(encoding="utf-8"))
    alias_index = AliasIndex(aliases, nodes)

    chunks: list[dict] = []
    for source in sources():
        path = source.path
        name = path.name
        is_knowledge = bool(SOURCES.knowledge_dir and path.parent == SOURCES.knowledge_dir)
        base_type = _doc_type(source)

        # DOCX → markdown; статус «только картинки» переносится в метаданные,
        # иначе документ молча выпадал бы из индекса.
        if source.kind == "docx":
            raw, extracted = docx.to_markdown(path)
            doc_status = extracted.status
        else:
            raw = md.strip_frontmatter(md.read_text(path))
            doc_status = "ok"

        # 11_reference.md состоит из FAQ и глоссария — разные типы данных
        parts = [(raw, base_type)]
        if PART_SPLIT in raw:
            head, tail = raw.split(PART_SPLIT, 1)
            parts = [(head, "faq"), (PART_SPLIT + tail, "glossary")]
        elif name == "11_reference.md":
            parts = [(raw, "faq")]

        seq = 0
        for text_part, part_type in parts:
            sections = _sections(text_part, path.stem)
            has_child = {tuple(s["path"][:-1]) for s in sections if len(s["path"]) > 1}
            for sec in sections:
                if tuple(sec["path"]) in has_child:
                    continue
                body = sec["body"]
                if len(body) < (20 if source.kind == "docx" else 60) and not FAULT_ID_RE.match(sec["title"]):
                    continue

                ctype = _refine_type(part_type, sec["title"], body)
                canonical = None
                if ctype == "fault":
                    m = FAULT_ID_RE.match(sec["title"])
                    if not m:
                        continue
                    canonical = f"F-{m.group(1)}.{m.group(2)}"
                elif ctype == "diagnostic":
                    m = DIAG_TITLE_RE.match(sec["title"])
                    if m:
                        canonical = f"D-{name.split('_')[0]}-{int(m.group(1)):02d}"
                elif ctype == "procedure" and sec["level"] == 2:
                    canonical = "P-10-" + md.slug(re.sub(r"^\d+\.\s*", "", sec["title"]), 32)
                elif ctype == "faq":
                    seq += 1
                    canonical = f"FAQ-{seq:03d}"
                elif ctype == "glossary" and sec["level"] >= 3:
                    canonical = "G-" + md.slug(sec["title"], 40)

                ph = md.placeholders(body)
                status = _status_for(ctype, source.authority, ph, source)
                pieces = _split_long(body)
                for i, piece in enumerate(pieces):
                    seq_id = f"{rel(path)}#{md.slug('-'.join(sec['path'][-2:]), 60)}"
                    if len(pieces) > 1:
                        seq_id += f"-p{i + 1}"
                    chunks.append(
                        {
                            "chunk_id": seq_id,
                            "collection": "history" if ctype.startswith("history") else "knowledge",
                            "doc": rel(path),
                            "doc_kind": source.kind,
                            "doc_status": doc_status,
                            "authority": source.authority,
                            "heading_path": sec["path"],
                            "title": sec["title"],
                            "type": ctype,
                            "canonical_id": canonical,
                            "entities": alias_index.detect(sec["title"] + " " + body),
                            "verification_status": status,
                            "has_gaps": bool(ph) or status in {"incomplete_source", "placeholder_partial"},
                            "gaps": ph[:5],
                            "source_priority": source.priority,
                            "is_duplicate_of": None,
                            "text": piece,
                            "search_text": " ".join(sec["path"][-2:] + [piece]),
                            "line": sec["line"],
                        }
                    )

    # дедупликация по отпечатку начала текста
    seen: dict[str, dict] = {}
    for ch in chunks:
        fp = md.norm(ch["text"])[:160]
        fp = re.sub(r"\W+", "", fp)
        if not fp:
            continue
        prev = seen.get(fp)
        if prev is None:
            seen[fp] = ch
        elif ch["source_priority"] > prev["source_priority"]:
            prev["is_duplicate_of"] = ch["chunk_id"]
            seen[fp] = ch
        else:
            ch["is_duplicate_of"] = prev["chunk_id"]

    # chunk_id должен быть уникальным: он же первичный ключ в SQLite и в vectors.json
    used: set[str] = set()
    for ch in chunks:
        base = ch["chunk_id"]
        cid, n = base, 1
        while cid in used:
            n += 1
            cid = f"{base}--{n}"
        ch["chunk_id"] = cid
        used.add(cid)

    CHUNKS.mkdir(parents=True, exist_ok=True)
    with (CHUNKS / "knowledge.jsonl").open("w", encoding="utf-8") as fh:
        for ch in chunks:
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")
    return chunks


def cmd_chunk() -> None:
    chunks = build_chunks()
    dup = sum(1 for c in chunks if c["is_duplicate_of"])
    by_type: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for c in chunks:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
        by_kind[c.get("doc_kind", "?")] = by_kind.get(c.get("doc_kind", "?"), 0) + 1
    print(f"chunks      {len(chunks)}")
    print(f"duplicates  {dup}")
    print("источники:  " + ", ".join(f"{k} {v}" for k, v in sorted(by_kind.items())))
    for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14} {v}")
