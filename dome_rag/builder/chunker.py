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

from . import md
from .paths import CHUNKS, knowledge_docs, rel

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
    """Сопоставление упоминаний в тексте с узлами и короткими терминами."""

    def __init__(self, aliases: dict[str, str], nodes: list[dict]):
        self.map = dict(aliases)
        for n in nodes:
            self.map[n["node_id"]] = n["node_id"]
            if n.get("code_name"):
                self.map[n["code_name"]] = n["node_id"]
        # длинные алиасы — подстрокой, короткие — по границе слова
        self.long = sorted((a for a in self.map if len(a) >= 5), key=len, reverse=True)
        self.short = sorted((a for a in self.map if len(a) < 5), key=len, reverse=True)
        self.short_re = {
            a: re.compile(rf"(?<![a-zа-я0-9]){re.escape(a)}(?![a-zа-я0-9])")
            for a in self.short
        }

    def detect(self, text: str) -> list[str]:
        low = md.norm(text)
        found: list[str] = []
        for alias in self.long:
            if alias in low:
                found.append(self.map[alias])
        for alias in self.short:
            if self.short_re[alias].search(low):
                found.append(self.map[alias])
        seen, out = set(), []
        for nid in found:
            if nid not in seen:
                seen.add(nid)
                out.append(nid)
        return out[:6]


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


def build_chunks() -> list[dict]:
    aliases = json.loads((CHUNKS.parent / "data" / "curated" / "aliases.json").read_text(encoding="utf-8"))
    nodes = json.loads((CHUNKS.parent / "data" / "nodes.json").read_text(encoding="utf-8"))
    alias_index = AliasIndex(aliases, nodes)

    chunks: list[dict] = []
    for path in knowledge_docs():
        name = path.name
        is_knowledge = path.parent.name == "knowledge"
        base_type = DOC_TYPE.get(name, "knowledge")
        raw = md.strip_frontmatter(md.read_text(path))

        # 11_reference.md состоит из FAQ и глоссария — разные типы данных
        parts = [(raw, base_type)]
        if PART_SPLIT in raw:
            head, tail = raw.split(PART_SPLIT, 1)
            parts = [(head, "faq"), (PART_SPLIT + tail, "glossary")]
        elif name == "11_reference.md":
            parts = [(raw, "faq")]

        seq = 0
        for text_part, part_type in parts:
            sections = md.split_sections(text_part)
            has_child = {tuple(s["path"][:-1]) for s in sections if len(s["path"]) > 1}
            for sec in sections:
                is_terminal = tuple(sec["path"]) not in has_child
                body = sec["body"]
                if not is_terminal:
                    continue
                if len(body) < 60 and not FAULT_ID_RE.match(sec["title"]):
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
                            "heading_path": sec["path"],
                            "title": sec["title"],
                            "type": ctype,
                            "canonical_id": canonical,
                            "entities": alias_index.detect(sec["title"] + " " + body),
                            "verification_status": (
                                "catalog" if ctype == "fault"
                                else "placeholder_partial" if ph
                                else "history" if ctype.startswith("history")
                                else "normative"
                            ),
                            "has_gaps": bool(ph),
                            "gaps": ph[:5],
                            "source_priority": _priority(name, is_knowledge),
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
    for c in chunks:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
    print(f"chunks      {len(chunks)}")
    print(f"duplicates  {dup}")
    for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14} {v}")
