"""Эпизодическая память: сжатые сводки смен из журналов купола.

Сырые журналы не индексируются: 26 МБ чисел бесполезны для поиска и опасны
для модели. На выходе — один чанк на смену: что видели, что произошло,
какие команды отдавались, что записал жилец.

Важно: запись ``logbook`` — это мнение жильца, а не подтверждённый факт,
поэтому она помечается ``verification_status = logbook_opinion``.
"""
from __future__ import annotations

import json
from collections import Counter

from . import md
from .chunker import AliasIndex
from .paths import CHUNKS, DATA, LOGS


def _load_jsonl(path):
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def build_episodes() -> list[dict]:
    aliases = json.loads((DATA / "curated" / "aliases.json").read_text(encoding="utf-8"))
    nodes = json.loads((DATA / "nodes.json").read_text(encoding="utf-8"))
    alias_index = AliasIndex(aliases, nodes)

    episodes = []
    for day_dir in sorted(p for p in LOGS.glob("*") if p.is_dir()):
        events_path = day_dir / "events.jsonl"
        book_path = day_dir / "logbook.jsonl"
        if not events_path.exists():
            continue
        events = _load_jsonl(events_path)
        books = _load_jsonl(book_path) if book_path.exists() else []

        kinds = Counter(e.get("kind") for e in events)
        dome = next((e.get("dome") for e in events if e.get("dome")), {})

        trips, clears = [], []
        for e in events:
            if e.get("kind") != "automation":
                continue
            ev = e.get("event") or {}
            line = f'{ev.get("interlock")} {ev.get("type")}: {ev.get("message", "")[:120]}'
            (trips if ev.get("type") == "TRIP" else clears).append(line)

        actions = []
        for e in events:
            if e.get("kind") == "action":
                actions.append(
                    f'{e.get("actor")} → {e.get("node_id")}.{e.get("action")} '
                    f'({"ok" if e.get("success") else "fail"})'
                )

        changes = Counter()
        for e in events:
            if e.get("kind") == "change":
                changes[f'{e.get("node_id")}.{e.get("param")}'] += 1

        book_text = " ".join(b.get("text", "") for b in books)

        lines = [
            f"Смена {day_dir.name} (день купола {dome.get('day')}).",
            f"Записей в журнале: {len(events)} "
            f"({', '.join(f'{k} {v}' for k, v in kinds.items())}).",
        ]
        if trips:
            lines.append("Срабатывания интерлоков: " + "; ".join(trips[:6]) + ".")
        if clears:
            lines.append("Снятия интерлоков: " + "; ".join(clears[:6]) + ".")
        if actions:
            lines.append("Команды: " + "; ".join(actions[:10]) + ".")
        if changes:
            top = ", ".join(f"{k} ×{v}" for k, v in changes.most_common(6))
            lines.append(f"Чаще всего менялись: {top}.")
        if book_text:
            lines.append("Бортовой журнал (мнение жильца): " + book_text[:1200])

        text = "\n".join(lines)
        episodes.append(
            {
                "chunk_id": f"episodes/{day_dir.name}",
                "collection": "episodes",
                "doc": f"Logs/{day_dir.name}",
                "heading_path": [f"Смена {day_dir.name}"],
                "title": f"Смена {day_dir.name}",
                "type": "episode",
                "canonical_id": f"EP-{day_dir.name}",
                "entities": alias_index.detect(text),
                "verification_status": "logbook_opinion" if book_text else "observed",
                "has_gaps": False,
                "gaps": [],
                "source_priority": 2,
                "is_duplicate_of": None,
                "text": text,
                "search_text": text,
                "line": 0,
            }
        )
    return episodes


def cmd_episodes() -> None:
    CHUNKS.mkdir(parents=True, exist_ok=True)
    eps = build_episodes()
    with (CHUNKS / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for e in eps:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"episodes    {len(eps)}")
