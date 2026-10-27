"""Извлечение структурированного ядра из корпуса.

Что генерируется автоматически:

* ``nodes.json``      — узлы, параметры, единицы, тип, команды;
* ``faults.json``     — 67 отказов с машинными признаками-заготовками;
* ``diagnostics.json``— 27 сценариев с шагами;
* ``procedures.json`` — процедуры из 10_procedures.md;
* ``faq.json``        — пары вопрос/ответ;
* ``glossary.json``   — термины и синонимы;
* ``equipment.json``  — таблицы оборудования из 02_station.md;
* ``sources.json``    — реестр источников с хешами.

Что поддерживается руками (см. ``data/curated``): пороги, интерлоки,
политики, пробелы, связи, карта команд и синонимы.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import md
from .paths import DATA, KNOWLEDGE, LOGS, PROJECT, ROOT_DOCS, sources

# ---------------------------------------------------------------- телеметрия

TELEMETRY_FALLBACKS = (
    "tests/fixtures/telemetry.jsonl",
    "Dome_RAG-main/tests/fixtures/telemetry.jsonl",
)


def _telemetry_paths() -> list[Path]:
    """Все известные снимки телеметрии, от новых к старым.

    Раньше здесь было ``sorted(LOGS.glob("*/telemetry.jsonl"))[-1]``: если
    каталога журналов нет (свежий клон, другой компьютер), сборка падала с
    ``IndexError``. Теперь отсутствие журналов — не ошибка: ниже есть
    резервный снимок из тестовой фикстуры.
    """
    found: list[Path] = []
    if LOGS.is_dir():
        found += sorted(LOGS.glob("*/telemetry.jsonl"), reverse=True)
        found += sorted(LOGS.glob("telemetry*.jsonl"), reverse=True)
    for name in TELEMETRY_FALLBACKS:
        for base in (PROJECT, PROJECT.parent):
            candidate = base / name
            if candidate.exists():
                found.append(candidate)
    return [p for p in found if p.is_file()]


def load_telemetry_sample() -> dict | None:
    """Первый снимок телеметрии: реальная схема живого API.

    ``None`` означает «снимка нет»; вызывающий код обязан это пережить и
    собрать базу из реестра и корпуса. Пустая база лучше падения, но база без
    документов — уже ошибка (см. ``paths.require_corpus``).
    """
    for path in _telemetry_paths():
        try:
            with path.open(encoding="utf-8") as fh:
                line = fh.readline()
            if line.strip():
                return json.loads(line)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _corpus_doc(name: str) -> Path | None:
    """Найти документ корпуса по имени, не полагаясь на конкретный каталог."""
    for source in sources():
        if source.path.name == name:
            return source.path
    for path in (KNOWLEDGE / name, PROJECT / name):
        if path.exists():
            return path
    for path in ROOT_DOCS:
        if path.name == name:
            return path
    return None


def _require_doc(name: str) -> Path:
    path = _corpus_doc(name)
    if path is None:
        raise SystemExit(f"В корпусе нет обязательного документа {name!r}")
    return path


def _read_doc(name: str) -> str:
    return md.strip_frontmatter(md.read_text(_require_doc(name)))


def _optional_doc(name: str) -> str:
    path = _corpus_doc(name)
    return md.strip_frontmatter(md.read_text(path)) if path else ""


UNIT_SUFFIXES = sorted(
    [
        ("_usv_h", "µSv/h"), ("_m3_h", "m³/h"), ("_l_min", "L/min"),
        ("_l_h", "L/h"), ("_kwh", "kWh"), ("_kbps", "kbps"), ("_mhz", "MHz"),
        ("_cpm", "cpm"), ("_ppm", "ppm"), ("_lux", "lx"), ("_rpm", "rpm"),
        ("_hpa", "hPa"), ("_ntu", "NTU"), ("_dbm", "dBm"), ("_deg", "°"),
        ("_ah", "Ah"), ("_ms", "m/s"), ("_km", "km"), ("_mm", "mm"),
        ("_pa", "Pa"), ("_bar", "bar"), ("_db", "dB"), ("_ma", "mA"),
        ("_mv", "mV"), ("_pct", "%"), ("_hz", "Hz"), ("_v", "V"), ("_a", "A"),
        ("_w", "W"), ("_c", "°C"), ("_l", "L"), ("_g", "g"), ("_h", "h"),
        ("_s", "s"), ("_min", "min"),
    ],
    key=lambda kv: -len(kv[0]),
)

STATE_KEYS = {
    "state", "status", "online", "mode", "alarm", "alarm_state", "sensor_state",
    "hold_level", "hold_reason", "protection_state", "lockout_reason", "fault_code",
    "error_code", "gcode_state", "webhook_state", "real_state", "fuel_quality",
    "filter_state", "system_state", "active_path", "link_state", "reserve_link_state",
    "panel_link_state", "ac_source", "grid_state", "fan_state", "fan_mode",
    "preheat_state", "balancing_state", "gateway_zone", "event_type", "substance",
    "quality_state", "part_defect", "bleeding", "reception", "protocol",
    "active_channel", "traffic_priority", "available", "data_available",
    "quake_detected", "smoke_detected", "leak_detected", "false_alarm_suspected",
    "sealed_zone", "active_zone", "lockdown", "reception",
}

COMPUTED_HINTS = (
    "dew_point", "power_deficit", "autonomy_forecast", "progress_pct",
    "print_time_left", "job_time_left", "hours_to_replacement", "uptime_h",
    "estimated_capacity", "wear_pct", "obscuration", "snr_db", "clock_offset",
    "reading_delay", "drift_", "soiling_loss", "cell_voltage_delta",
)

META_KEYS = {"updated_at", "status", "online", "confidence", "sensor_state"}


def infer_unit(key: str) -> str | None:
    for suffix, unit in UNIT_SUFFIXES:
        if key.endswith(suffix):
            return unit
    return None


def infer_kind(key: str, value) -> str:
    if key in ("updated_at", "status", "online"):
        return "meta"
    if isinstance(value, bool):
        return "state"
    if isinstance(value, str):
        return "state"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, dict):
        return "struct"
    if isinstance(value, (int, float)):
        if any(h in key for h in COMPUTED_HINTS):
            return "computed"
        return "measurement"
    return "unknown"


def infer_trust(key: str, kind: str) -> str:
    if key in META_KEYS:
        return "meta"
    if kind == "computed":
        return "computed"
    if kind in ("state", "measurement"):
        return "sensor"
    return "unknown"


# ------------------------------------------------------------------ реестр


REG_BULLET = re.compile(r"^-\s+\*\*(.+?):\*\*\s*(.+)$")


def parse_registry() -> dict[str, dict]:
    """Реестр узлов из dome_sandbox_nodes.md: id → карточка.

    В документе сбита нумерация разделов (два раза «## 4», «## 5»), поэтому
    карточки собираются по факту наличия ``Уникальный ID``, а не по номерам.
    """
    path = _require_doc("dome_sandbox_nodes.md")
    text = md.strip_frontmatter(md.read_text(path))
    out: dict[str, dict] = {}
    for sec in md.split_sections(text):
        card: dict = {"telemetry_doc": [], "controls_doc": []}
        for raw in sec["body"].splitlines():
            line = raw.strip()
            m = REG_BULLET.match(line)
            if m:
                key = m.group(1).strip().lower()
                val = m.group(2).strip().replace("`", "")
                if key == "название":
                    card["title"] = val
                elif key == "code name":
                    card["code_name"] = val
                elif "id" in key:
                    card["node_id"] = val
                elif key == "система":
                    card["system"] = val
                continue
            if line.startswith("**Телеметрия"):
                card["_b"] = "telemetry_doc"
                continue
            if line.startswith("**Управляющие"):
                card["_b"] = "controls_doc"
                continue
            b = card.get("_b")
            if b and line.startswith("-"):
                card[b].append(line.lstrip("- ").strip().rstrip(";"))
            elif b and line and not line.startswith(("**", "---")) and card[b]:
                card[b][-1] = (card[b][-1] + " " + line).strip()
        if card.get("node_id"):
            card.pop("_b", None)
            for one in re.split(r"\s*/\s*", card["node_id"]):
                one = one.strip()
                if one:
                    out.setdefault(one, {**card, "node_id": one})
    return out


SYSTEM_MAP = {
    "Энергетика": "energy",
    "Электроснабжение": "power",
    "Производство": "production",
    "Климат": "climate",
    "Вентиляция": "ventilation",
    "Метеорология": "weather",
    "Внешние системы": "external",
    "Водоснабжение": "water",
    "Безопасность": "safety",
    "Пожарная безопасность": "fire",
    "Вычисления": "compute",
    "Автоматика": "automation",
    "Связь и вычисления": "compute",
    "МС-ТЮК": "ms_tuk",
}

SYSTEM_FALLBACK = {
    "climate_sensor_02": "Климат",
    "climate_sensor_03": "Климат",
    "smoke_detector_02": "Пожарная безопасность",
    "network_link_01": "Связь и вычисления",
    "dome_automation_01": "Автоматика",
}

FALLBACK_TITLES = {
    "network_link_01": "Сетевой канал связи",
    "dome_automation_01": "Автоматика купола (интерлоки)",
    "smoke_detector_02": "Датчик дыма №2",
    "climate_sensor_03": "Датчик климата №3",
}


def build_nodes(telemetry: dict | None) -> list[dict]:
    """Узлы из снимка телеметрии и/или реестра.

    Снимок телеметрии — не обязательный вход. Если журналов нет, узлы всё
    равно описываются по ``dome_sandbox_nodes.md`` и ``controls_api.json``;
    иначе один отсутствующий каталог ломал всю сборку.
    """
    registry = parse_registry()
    controls = json.loads((DATA / "curated" / "controls_api.json").read_text(encoding="utf-8"))
    aliases = json.loads((DATA / "curated" / "aliases.json").read_text(encoding="utf-8"))
    alias_by_node: dict[str, list[str]] = {}
    for alias, node_id in aliases.items():
        alias_by_node.setdefault(node_id, []).append(alias)

    payloads: dict[str, dict] = dict((telemetry or {}).get("nodes") or {})
    schema_path = DATA / "curated/node_schema.json"
    schema = {n["node_id"]: n for n in json.loads(schema_path.read_text(encoding="utf-8"))} if schema_path.is_file() else {}
    node_ids = sorted(set(payloads) | set(registry) | set(schema))

    nodes = []
    for node_id in node_ids:
        payload = payloads.get(node_id) or {}
        reg = registry.get(node_id, {})
        params = []
        for key, value in payload.items():
            kind = infer_kind(key, value)
            unit = infer_unit(key)
            entry = {
                "key": key,
                "unit": unit,
                "kind": kind,
                "trust": infer_trust(key, kind),
                "sample": value if not isinstance(value, (list, dict)) else None,
            }
            if isinstance(value, str) and len(value) < 40:
                entry["enum_hint"] = value
            if isinstance(value, list) and value and isinstance(value[0], dict):
                entry["item_fields"] = list(value[0].keys())
            if isinstance(value, dict):
                entry["item_fields"] = list(value.keys())
            params.append(entry)

        observed_keys = {p["key"] for p in params}
        params.extend(dict(p) for p in schema.get(node_id, {}).get("params", []) if p["key"] not in observed_keys)
        system_ru = reg.get("system") or SYSTEM_FALLBACK.get(node_id)
        nodes.append(
            {
                "node_id": node_id,
                "code_name": reg.get("code_name") or node_id.rsplit("_", 1)[0],
                "title": reg.get("title") or FALLBACK_TITLES.get(node_id, node_id),
                "system": SYSTEM_MAP.get(system_ru or "", "unknown"),
                "system_ru": system_ru,
                "params": params,
                "controls": controls.get(node_id, []),
                "aliases": sorted(set(alias_by_node.get(node_id, []))),
                "telemetry_doc": reg.get("telemetry_doc", []),
                "controls_doc": reg.get("controls_doc", []),
                "data_source": "telemetry" if payload else "registry",
                "source": (
                    "dome_sandbox_nodes.md + telemetry.jsonl"
                    if payload
                    else "dome_sandbox_nodes.md (снимок телеметрии недоступен)"
                ),
            }
        )
    nodes.sort(key=lambda n: n["node_id"])
    return nodes


# ------------------------------------------------------------------ отказы


FAULT_FIELD_MAP = {
    "что происходит": "what_happens",
    "как заметить": "how_to_notice",
    "наблюдаемость": "observability",
    "что может сделать человек": "human_action",
    "что может сделать агент": "agent_action",
    "чего нельзя допустить": "must_not",
}


def build_faults() -> list[dict]:
    text = _read_doc("09_faults.md")
    faults = []
    for sec in md.split_sections(text):
        m = re.match(r"^(\d+)\.(\d+)\.\s+(.+)$", sec["title"])
        if not m:
            continue
        fields = md.parse_fields(sec["body"])
        rec = {
            "id": f"F-{m.group(1)}.{m.group(2)}",
            "category_no": int(m.group(1)),
            "category": sec["path"][-2] if len(sec["path"]) > 1 else "",
            "title": m.group(3).strip(),
            "source": "09_faults.md",
            "status": "catalog",
            "signals": [],
        }
        for label, value in fields.items():
            key = FAULT_FIELD_MAP.get(label.lower().strip())
            if key:
                rec[key] = value
        rec["category"] = re.sub(r"^\d+\.\s*", "", rec.get("category", ""))
        obs = rec.get("observability", "").lower()
        if "скрыт" in obs:
            rec["observability"] = "hidden"
        elif "прям" in obs:
            rec["observability"] = "direct"
        else:
            rec["observability"] = "unknown"
        faults.append(rec)
    return faults


# ------------------------------------------------------------- диагностика

DIAG_FILES = ["03_energy.md", "04_telemetry.md", "05_production.md", "06_microclimate.md"]

DIAG_LABELS = {
    "наблюдение": "observation",
    "что видно": "observation",
    "что проверить": "checks",
    "возможные объяснения": "hypotheses",
    "что может быть": "hypotheses",
    "дальнейшие действия": "next_steps",
    "что делать": "next_steps",
}


def build_diagnostics() -> list[dict]:
    out = []
    for fname in DIAG_FILES:
        text = _read_doc(fname)
        n = 0
        for sec in md.split_sections(text):
            if not sec["title"].lower().startswith("ситуация"):
                continue
            n += 1
            fields = md.parse_fields(sec["body"])
            rec = {
                "id": f"D-{fname.split('_')[0]}-{n:02d}",
                "title": re.sub(r"^Ситуация\s*\d+[.:]?\s*", "", sec["title"]).strip(),
                "system": {
                    "03_energy.md": "energy",
                    "04_telemetry.md": "telemetry",
                    "05_production.md": "production",
                    "06_microclimate.md": "climate",
                }[fname],
                "source": f"{fname}#{sec['title']}",
                "status": "hypothesis_catalog",
                "checks": md.parse_numbered(sec["body"]),
                "hypotheses": [],
                "observation": "",
                "next_steps": "",
            }
            for label, value in fields.items():
                key = DIAG_LABELS.get(label.lower().strip())
                if not key:
                    continue
                if key in ("checks", "hypotheses"):
                    continue
                rec[key] = value
            hyps = []
            grab = False
            for raw in sec["body"].splitlines():
                low = raw.strip().lower()
                if low.startswith("**возможные объяснения") or low.startswith("**что может быть"):
                    grab = True
                    continue
                if grab:
                    if raw.strip().startswith("-"):
                        hyps.append(raw.strip().lstrip("- ").strip())
                    elif raw.strip().startswith("**"):
                        grab = False
            rec["hypotheses"] = hyps
            out.append(rec)
    return out


# ---------------------------------------------------------------- процедуры


FORBIDDEN_HINTS = ("нельзя", "запрещ", "не следует", "не пытайтесь", "не допускается")
STEP_PREFIX = re.compile(r"^\d+\.\s*")


def build_procedures() -> list[dict]:
    text = _read_doc("10_procedures.md")
    sections = md.split_sections(text)
    out = []
    for sec in sections:
        if sec["level"] != 2:
            continue
        body = sec["body"]
        steps = md.parse_numbered(body)
        forbidden = [b for b in md.parse_bullets(body) if any(w in b.lower() for w in FORBIDDEN_HINTS)]

        if not steps:
            # часть процедур разбита на дочерние секции «Шаг N. ...»
            children = [
                s for s in sections
                if s["level"] == 3 and len(s["path"]) >= 2 and s["path"][-2] == sec["title"]
            ]
            step_children = [c for c in children if c["title"].lower().startswith("шаг")]
            if step_children:
                steps = [
                    f'{c["title"]}. {md.first_sentences(c["body"], 1, 200)}'.strip()
                    for c in step_children
                ]
                for c in step_children:
                    forbidden += [
                        b for b in md.parse_bullets(c["body"])
                        if any(w in b.lower() for w in FORBIDDEN_HINTS)
                    ]
            else:
                steps = md.parse_bullets(body)[:15]

        if not steps:
            continue
        title = STEP_PREFIX.sub("", sec["title"]).strip()
        out.append(
            {
                "id": "P-10-" + md.slug(title, 32),
                "title": title,
                "system": "operations",
                "source": f"10_procedures.md#{sec['title']}",
                "status": "normative",
                "steps": steps,
                "forbidden": forbidden,
            }
        )
    return out


# ------------------------------------------------------------------ справка


def build_faq_and_glossary() -> tuple[list[dict], list[dict]]:
    text = _read_doc("11_reference.md")
    cut = text.find("# Часть 2. Глоссарий")
    part1, part2 = (text[:cut], text[cut:]) if cut > 0 else (text, "")

    faq, glossary = [], []
    n = 0
    for sec in md.split_sections(part1):
        if sec["level"] != 3 or not sec["title"].strip():
            continue
        n += 1
        answer = re.sub(r"\n{2,}", "\n", sec["body"]).strip()
        faq.append(
            {
                "id": f"FAQ-{n:03d}",
                "question": sec["title"],
                "answer": answer,
                "section": sec["path"][-2] if len(sec["path"]) > 1 else "",
                "source": "11_reference.md",
            }
        )

    term_re = re.compile(r"^\*\*(.+?)\*\*\s*[—–-]\s*(.+)$")
    for raw in part2.splitlines():
        m = term_re.match(raw.strip())
        if not m:
            continue
        glossary.append(
            {
                "term": m.group(1).strip(),
                "definition": m.group(2).strip(),
                "source": "11_reference.md",
            }
        )
    return faq, glossary


# ---------------------------------------------------------------- оборудование


def build_equipment() -> list[dict]:
    text = _read_doc("02_station.md")
    sections = md.split_sections(text)
    out = []
    for sec in sections:
        for table in md.parse_tables(sec["body"]):
            header = table["header"]
            for row in table["rows"]:
                if len(row) < 2:
                    continue
                out.append(
                    {
                        "section": sec["path"][-1],
                        "path": sec["path"],
                        "columns": header,
                        "row": row,
                        "source": "02_station.md",
                    }
                )
    return out


# ------------------------------------------------------------------ источники


def build_sources() -> list[dict]:
    """Реестр источников с хешами, авторитетом и статусом извлечения.

    DOCX попадают сюда наравне с markdown: если документ «только из
    картинок», это видно в ``status``, а не теряется молча.
    """
    from . import manifest

    scanned = manifest.scan()
    return [
        {
            "id": item.rel_path,
            "path": item.rel_path,
            "kind": item.kind,
            "priority": item.priority,
            "authority": item.authority,
            "status": item.status,
            "checksum": item.sha256[:16],
            "sha256": item.sha256,
            "bytes": item.bytes,
            "chars": item.chars,
            "note": item.note,
        }
        for item in scanned
    ]


# ------------------------------------------------------------------- запись


def cmd_ingest() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    telemetry = load_telemetry_sample()
    if telemetry is None:
        print("telemetry   снимок не найден — узлы собираются из реестра")
    nodes = build_nodes(telemetry)
    faults = build_faults()
    diagnostics = build_diagnostics()
    procedures = build_procedures()
    faq, glossary = build_faq_and_glossary()
    equipment = build_equipment()
    sources = build_sources()

    payloads = {
        "nodes.json": nodes,
        "faults.json": faults,
        "diagnostics.json": diagnostics,
        "procedures.json": procedures,
        "faq.json": faq,
        "glossary.json": glossary,
        "equipment.json": equipment,
        "sources.json": sources,
    }
    for name, data in payloads.items():
        (DATA / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    print(f"nodes       {len(nodes):>4}")
    print(f"  params    {sum(len(n['params']) for n in nodes):>4}")
    print(f"  controls  {sum(len(n['controls']) for n in nodes):>4}")
    print(f"faults      {len(faults):>4}")
    print(f"diagnostics {len(diagnostics):>4}")
    print(f"procedures  {len(procedures):>4}")
    print(f"faq         {len(faq):>4}")
    print(f"glossary    {len(glossary):>4}")
    print(f"equipment   {len(equipment):>4}")
    print(f"sources     {len(sources):>4}")
    for item in sources:
        if item["status"] != "ok":
            print(f"  ! {item['path']}: {item['status']} {item['note']}".rstrip())
