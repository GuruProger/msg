"""Отчёт о расхождениях корпуса: что нельзя считать подтверждённым.

Отчёт не «украшение», а защита от галлюцинаций: он показывает, где документы
противоречат друг другу, где стоят заглушки и какие узлы описаны неполно.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import md
from .paths import BUILD, CHUNKS, DATA, PROJECT, REPORTS, knowledge_docs, rel

LINE_MAP_RE = re.compile(r"(\d)\s*[.)—–-]\s*([^,\n;]+)")


def _norm_label(label: str) -> str:
    """Приводит подписи линий к сравнимому виду.

    «3D-принтер» и «принтер», «ЧПУ-фрезер» и «фрезер», «обычное освещение» и
    «освещение» — это одна и та же нагрузка, а не расхождение.
    """
    s = (label or "").lower().replace("ё", "е")
    s = re.sub(r"[^a-zа-я0-9]+", " ", s)
    s = re.sub(r"\b(3d|чпу)\b", " ", s)
    s = re.sub(r"\b(обычное|аварийное|аварийный)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _same_label(a: str, b: str) -> bool:
    na, nb = _norm_label(a), _norm_label(b)
    if not na or not nb:
        return na == nb
    return na == nb or na in nb or nb in na


def _load(name: str):
    for base in (DATA, DATA / "curated"):
        p = base / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return []


def extract_line_mapping() -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    sources = {
        "dome_sandbox_nodes.md": PROJECT / "dome_sandbox_nodes.md",
        "00_Краткая_справка_API.md": PROJECT / "00_Краткая_справка_API.md",
    }
    for name, path in sources.items():
        if not path.exists():
            continue
        text = md.read_text(path)
        m = re.search(r"Назначение линий\**\s*:?\s*(.+?)(?:\n\n|\n- |\n#)", text, re.S)
        if not m:
            continue
        block = m.group(1)
        mapping = {}
        for num, label in LINE_MAP_RE.findall(block):
            mapping[int(num)] = re.sub(r"\s+", " ", label).strip(" .")
        if mapping:
            out[name] = mapping
    return out


def _placeholder_report() -> list[tuple[str, int]]:
    rows = []
    for p in knowledge_docs():
        text = md.read_text(p)
        n = len(md.placeholders(text)) + len(md.html_comment_gaps(text))
        if n:
            rows.append((rel(p), n))
    return sorted(rows, key=lambda r: -r[1])


def _chunks() -> list[dict]:
    out = []
    for p in (CHUNKS / "knowledge.jsonl", CHUNKS / "episodes.jsonl"):
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                out += [json.loads(line) for line in fh if line.strip()]
    return out


def build_report() -> str:
    nodes = _load("nodes.json")
    gaps = _load("gaps.json")
    thresholds = _load("thresholds.json")
    chunks = _chunks()
    mapping = extract_line_mapping()

    L: list[str] = []
    L.append("# Отчёт о расхождениях DomeKB")
    L.append("")
    L.append("Файл генерируется автоматически: `py kb.py drift`. Руками не править.")
    L.append("")

    # ------------------------------------------------------------ сводка
    L.append("## 1. Сводка")
    L.append("")
    L.append(f"- узлов: {len(nodes)}")
    L.append(f"- параметров: {sum(len(n.get('params', [])) for n in nodes)}")
    L.append(f"- команд: {sum(len(n.get('controls', [])) for n in nodes)}")
    L.append(f"- чанков: {len(chunks)}")
    L.append(f"- пробелов в реестре: {len(gaps)}")
    L.append(f"- порогов с пометкой «гипотеза»: {sum(1 for t in thresholds if t.get('status') != 'confirmed')}")
    L.append("")

    # --------------------------------------------------- конфликт линий
    L.append("## 2. Конфликт распределения линий щита (GAP-001)")
    L.append("")
    if len(mapping) >= 2:
        names = list(mapping)
        all_nums = sorted({n for m in mapping.values() for n in m})
        L.append("| Линия | " + " | ".join(f"`{n}`" for n in names) + " |")
        L.append("|---|" + "---|" * len(names))
        for num in all_nums:
            cells = [mapping[n].get(num, "—") for n in names]
            present = [c for c in cells if c != "—"]
            conflict = any(
                not _same_label(a, b)
                for i, a in enumerate(present) for b in present[i + 1:]
            )
            mark = " ⚠️" if conflict else ""
            L.append(f"| {num}{mark} | " + " | ".join(cells) + " |")
        L.append("")
        diffs = []
        for num in all_nums:
            present = [mapping[n].get(num) for n in names if mapping[n].get(num)]
            if any(not _same_label(a, b) for i, a in enumerate(present) for b in present[i + 1:]):
                diffs.append(num)
        L.append(f"Расходятся линии: {', '.join(map(str, diffs))}." if diffs else "Расхождений не найдено.")
        L.append("")
    else:
        L.append("Не удалось извлечь обе таблицы назначения линий.")
        L.append("")
    L.append(
        "Третий источник — `knowledge/03_energy.md` §5.6 — прямо утверждает, что подтверждённой "
        "таблицы нет. **Правило: не утверждать, что линия питает конкретное оборудование.** "
        "Ориентир — поле `lines[].name` в живой телеметрии и сообщение Оператора."
    )
    L.append("")

    # ----------------------------------------------------- заглушки
    L.append("## 3. Незаполненные места в корпусе")
    L.append("")
    ph = _placeholder_report()
    if ph:
        L.append("| Документ | Пометок |")
        L.append("|---|---|")
        for name, n in ph:
            L.append(f"| `{name}` | {n} |")
    else:
        L.append("Пометок `[ЗАПОЛНИТЬ]` и скрытых списков в комментариях нет.")
    L.append("")
    gap_chunks = [c for c in chunks if c.get("has_gaps")]
    L.append(f"Чанков с флагом `has_gaps`: {len(gap_chunks)}. Они не могут быть единственной опорой ответа.")
    L.append("")

    # ------------------------------------------- каталог и история
    L.append("## 4. Каталог отказов ≠ история станции")
    L.append("")
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for c in chunks:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
        by_status[c["verification_status"]] = by_status.get(c["verification_status"], 0) + 1
    L.append("| Тип чанка | Штук |")
    L.append("|---|---|")
    for t, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        L.append(f"| `{t}` | {n} |")
    L.append("")
    L.append("| Статус проверки | Штук |")
    L.append("|---|---|")
    for s, n in sorted(by_status.items(), key=lambda kv: -kv[1]):
        L.append(f"| `{s}` | {n} |")
    L.append("")
    L.append(
        "Записи `catalog` описывают возможные отказы. Утверждать «этот отказ происходил» "
        "можно только по `observed`/журналу. `logbook_opinion` — мнение жильца, не факт."
    )
    L.append("")

    # ------------------------------------- реестр против телеметрии
    L.append("## 5. Реестр и телеметрия")
    L.append("")
    no_controls = [n["node_id"] for n in nodes if not n.get("controls")]
    no_unit = [
        f"{n['node_id']}.{p['key']}"
        for n in nodes for p in n.get("params", [])
        if p.get("kind") == "measurement" and not p.get("unit")
    ]
    fallback_titles = [
        n["node_id"] for n in nodes
        if n.get("title") in (None, "", n["node_id"])
    ]
    L.append(f"- узлы без команд (только чтение): {', '.join(f'`{x}`' for x in no_controls) or 'нет'}")
    L.append(f"- измерения без единицы: {len(no_unit)}" + (f" — например {', '.join(no_unit[:6])}" if no_unit else ""))
    L.append(f"- узлы без названия из реестра: {', '.join(f'`{x}`' for x in fallback_titles) or 'нет'}")
    L.append("")
    L.append(
        "Часть узлов отсутствует в `dome_sandbox_nodes.md` (дополнительные датчики климата, "
        "резервный канал, автоматика). Для них название и система восстановлены по телеметрии — "
        "это помечено в отчёте, но не в самих данных."
    )
    L.append("")

    # ------------------------------------------------------ дубли
    L.append("## 6. Дубли и почти-дубли")
    L.append("")
    dups = [c for c in chunks if c.get("is_duplicate_of")]
    L.append(f"Помечено дублями: {len(dups)}.")
    for c in dups[:15]:
        L.append(f"- `{c['chunk_id']}` → дубль `{c['is_duplicate_of']}`")
    if len(dups) > 15:
        L.append(f"- … ещё {len(dups) - 15}")
    L.append("")

    # ------------------------------------------------------ пробелы
    L.append("## 7. Открытые пробелы")
    L.append("")
    L.append("| ID | Область | Что неизвестно | Кто закрывает |")
    L.append("|---|---|---|---|")
    for g in gaps:
        L.append(
            f"| {g['id']} | {g.get('area')} | {g.get('description')} | "
            f"{g.get('resolution') or '—'} |"
        )
    L.append("")

    # ------------------------------------------------------ источники
    L.append("## 8. Источники и контрольные суммы")
    L.append("")
    L.append("| Документ | Байт | SHA-256 (16) |")
    L.append("|---|---|---|")
    for s in _load("sources.json"):
        L.append(f"| `{s['path']}` | {s['bytes']} | `{s['checksum']}` |")
    L.append("")
    return "\n".join(L)


def cmd_report() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    text = build_report()
    path = REPORTS / "drift.md"
    path.write_text(text, encoding="utf-8")
    print(f"report      {path}")
    print(f"bytes       {len(text.encode('utf-8'))}")
