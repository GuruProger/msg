"""Карточки знаний — единица поиска вместо «сырого абзаца».

Почему не абзац: абзац отвечает «где-то здесь были похожие слова», а карточка
отвечает «вот субъект, вот атрибут, вот значение и вот источник». Это
позволяет проверить, что найденное действительно отвечает на вопрос, и честно
отказаться, если ответа нет.

Схема карточки::

    {
      "card_id": "KC-...",          # стабильный id
      "subject": "printer_3d_01",   # о ком/чём
      "subject_label": "3D-принтер",
      "attribute": "fault_action",  # что именно утверждается
      "value": "stop_job; проверить bed_clear",  # краткое значение
      "text": "...",                # формулировка для человека
      "entities": ["printer_3d_01"],
      "codes": ["F-3.2"],
      "source": {"doc": "...", "title": "...", "line": 12},
      "authority": 5,               # 5 — нормативный API, 2 — история
      "status": "normative",        # normative|catalog|theory|history|incomplete|conflict
      "canonical_id": "F-3.2",
      "chunk_id": "...",            # обратная ссылка на фрагмент
      "search_text": "..."
    }
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import md, taxonomy
from .chunker import AliasIndex
from .paths import AUTHORITY, CARDS_PATH, CHUNKS, DATA

CODE_RE = re.compile(
    r"\b(?:IL_[A-Z0-9_]+|TH_[A-Z0-9_]+|GAP-\d+|F-\d+\.\d+|D-\d{2}-\d{2}|"
    r"P-10-[A-Za-zА-Яа-я0-9_-]+|FAQ-\d+|POL-\d+|E\d{2})\b",
    re.IGNORECASE,
)

STATUS_MAP = {
    "normative": "normative",
    "catalog": "catalog",
    "theory": "theory",
    "history": "history",
    "history_fault": "history",
    "placeholder_partial": "incomplete",
    "incomplete_source": "incomplete",
    "logbook_opinion": "opinion",
    "hypothesis": "hypothesis",
    "gap": "gap",
    "reference": "reference",
}

# Тип чанка → атрибут, который он может подтвердить. Это грубая, но
# объяснимая привязка: журнал отвечает на «что было», диагностика — на
# «что происходит», правила — на «можно ли».
CHUNK_ATTRIBUTE = {
    "history": "history_event",
    "history_fault": "history_event",
    "episode": "history_event",
    "diagnostic": "fault_symptom",
    "procedure": "procedure_steps",
    "rule": "policy_rule",
    "safety": "policy_rule",
    "limits": "policy_rule",
    "fault": "fault_symptom",
    "equipment": "equipment_spec",
    "registry": "equipment_spec",
    "api": "control_capability",
    "glossary": "definition",
    "faq": "definition",
}


def _clip(value, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


# Признаки, по которым абзац действительно что-то утверждает. Раньше атрибут
# ставился по типу документа: любой API-абзац становился «возможностью
# управления», даже если это был пример JSON. Теперь нужен содержательный
# сигнал, иначе абзац остаётся section_text.
RULE_CUE = re.compile(
    r"\b(?:запрещ\w*|нельзя|не\s+допуска\w*|обязан\w*|разреш\w*|только|"
    r"правил\w*|политик\w*|должен\w*)\b",
    re.IGNORECASE,
)
STEP_CUE = re.compile(
    r"(?:^|\n)\s*\d+[\).]\s|\bшаг\s*\d|\bпорядок\s+действий|\bпошагов",
    re.IGNORECASE,
)
SPEC_CUE = re.compile(
    r"\d+\s*(?:мм|см|кг|г|вт|квт|в|а|гц|°|%|л/мин|об/мин|м/с)",
    re.IGNORECASE,
)
# Заголовки «Параметры наблюдения», «Что наблюдать», «Что проверять»
# описывают именно метрики, а не просто раздел. Без этого они оставались
# section_text и не могли ответить на «какой параметр показывает заряд».
MEASURE_TITLE = re.compile(
    r"(?:параметр\w*|наблюда\w*|что\s+наблюдать|что\s+проверять|телеметри\w*)",
    re.IGNORECASE,
)
COMMAND_CUE = re.compile(
    r"\b(?:action|команд\w*|запрос\w*|endpoint|list_files|start_job|"
    r"stop_job|resume_job|probe_z|bed_clear|control|snapshot|say)\b",
    re.IGNORECASE,
)


def _chunk_attribute(chunk: dict) -> str:
    ctype = chunk.get("type") or ""
    title = chunk.get("title") or ""
    body = chunk.get("text") or ""
    if ctype in {"history", "history_fault"}:
        return "history_event"
    if ctype in {"faq", "glossary"}:
        return "definition"
    if re.match(r"^\s*что\s+(?:такое|это|значит|означает)\b", title, re.IGNORECASE):
        return "definition"
    if ctype in {"fault", "diagnostic"}:
        return "fault_symptom"
    if ctype in {"rule", "safety", "limits"}:
        return "policy_rule" if RULE_CUE.search(body) else "section_text"
    if ctype == "procedure":
        return "procedure_steps" if STEP_CUE.search(body) else "section_text"
    if ctype == "api":
        # API-документ отвечает на «как это сделать», только если в нём
        # действительно названа команда. Пример JSON без команды — контекст.
        return "control_capability" if COMMAND_CUE.search(body) else "section_text"
    if ctype in {"equipment", "registry"}:
        return "equipment_spec" if SPEC_CUE.search(body) else "section_text"
    if ctype == "system" and MEASURE_TITLE.search(title):
        return "measurement_spec"
    return "section_text"



def _codes(*values: str) -> list[str]:
    found: list[str] = []
    for value in values:
        for match in CODE_RE.findall(str(value or "")):
            code = match.upper() if match[:3].upper() in {"IL_", "TH_"} else match
            if code not in found:
                found.append(code)
    return found


ACTION_WORDS = {
    "remove_part": "снять или забрать деталь со стола",
    "clean_filter": "очистить фильтр",
    "inspect_cable": "осмотреть кабель",
    "open_box_door": "открыть дверь бокса",
    "replace_fan": "заменить вентилятор",
    "set_door": "управлять дверью",
    "switch_patchcord": "переключить патчкорд",
    "workplace_mode": "режим рабочего места",
    "hardware_interlock": "аппаратная блокировка",
    "emergency_stop": "аварийная остановка",
    "fire_suppression": "пожаротушение",
    "dome_sealing": "герметизация купола",
}

# Бытовые формулировки команд. Без них «как написать Оператору» не находило
# post_message: карточка говорила «отправить сообщение в чат», а вопрос —
# «написать». Это не синонимы атрибута, а именно алиасы конкретной команды,
# поэтому они и живут рядом с ACTION_WORDS.
ACTION_ALIASES = {
    "list_files": "список файлов, какие файлы, узнать файлы, посмотреть файлы",
    "upload_file": "загрузить файл, закачать файл",
    "delete_file": "удалить файл, стереть файл",
    "start_job": "запустить печать, начать печать, запустить фрезеровку",
    "resume": "снять с паузы, продолжить печать, возобновить",
    "home": "найти нулевое положение, вернуться в ноль",
    "set_speed": "изменить скорость, скорость печати",
    "move": "перемещение по осям, подвинуть",
    "post_message": "написать оператору, отправить сообщение, написать сообщение",
    "read_messages": "прочитать сообщения, прочитать чат, история чата",
    "reset_error": "сбросить ошибку",
    "recalibrate": "перекалибровать датчик",
    "bed_clear": "стол свободен, деталь на столе",
    "probe_z": "измерить ноль по оси z",
    "snapshot": "снимок телеметрии",
}


def _action_words(actions) -> str:
    return ", ".join(
        f"{action} ({ACTION_WORDS[action]})" for action in (actions or []) if action in ACTION_WORDS
    )


METRIC_WORDS = {
    "co2_ppm": "CO2",
    "humidity_pct": "влажность",
    "dew_point_delta_c": "точка росы",
    "inverter_temp_c": "температура инвертора",
    "soc_pct": "SOC аккумулятора",
    "soc": "уровень заряда",
    "battery": "АКБ",
    "switches": "переключения линии",
    "speed_pct": "скорость печати",
    "requests_per_second": "запросы в секунду",
    "wind_speed_ms": "скорость ветра",
    "voltage_v": "напряжение линии",
    "current_a": "ток линии",
    "resume_attempts": "попытки resume",
    "error_code": "код ошибки",
    "power_w": "мощность генерации",
    "feedrate": "подача",
    "move": "ход осей",
    "messages": "сообщения",
}


# Метрика порога почти всегда записана латиницей (`climate.humidity_pct`),
# поэтому алиасы её не узнают. Явное соответствие возвращает карточке порога
# тот узел, о котором спрашивают («какая влажность опасна для станков»).
METRIC_ENTITIES = {
    "climate.": "climate_sensor_01",
    "humidity": "climate_sensor_01",
    "dew_point": "climate_sensor_01",
    "co2": "climate_sensor_01",
    "smoke": "smoke_detector_01",
    "wind": "weather_station_01",
    "radiation": "radiation_sensor_01",
    "battery": "battery_01",
    "soc_": "battery_01",
    "smart_panel": "smart_panel_01",
    "inverter": "solar_inverter_01",
    "solar": "solar_panels_01",
    "printer": "printer_3d_01",
    "cnc": "cnc_01",
}


def _metric_entities(metric: str) -> list[str]:
    low = str(metric or "").lower()
    return list(dict.fromkeys(
        node for key, node in METRIC_ENTITIES.items() if key in low
    ))


def _metric_label(metric: str) -> str:
    """Человеческое имя метрики: `humidity_pct` → «влажность»."""
    low = str(metric or "").lower()
    labels = [text for key, text in METRIC_WORDS.items() if key in low]
    return ", ".join(dict.fromkeys(labels))


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={_fmt(item)}" for key, item in value.items())
    return _clip(value, 120)


def _authority(value: str | int | None, default: int = 3) -> int:
    if isinstance(value, int):
        return value
    if value in AUTHORITY:
        return AUTHORITY[value]
    return default


class CardBuilder:
    def __init__(self) -> None:
        self.aliases = json.loads((DATA / "curated" / "aliases.json").read_text(encoding="utf-8"))
        self.nodes = self._load("nodes.json")
        self.alias_index = AliasIndex(self.aliases, self.nodes)
        self.cards: list[dict] = []
        self._seen: set[str] = set()
        self._by_key: dict[tuple[str, str, str], dict] = {}
        # (canonical_id, attribute), уже описанные структурной карточкой.
        # Нужны, чтобы абзац не дублировал точный факт из thresholds/faults.
        self._structured_pairs: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ утилиты

    def _load(self, name: str) -> list[dict] | dict:
        for path in (DATA / name, DATA / "curated" / name):
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        return []

    def _entities(self, *values: str) -> list[str]:
        text = " ".join(str(v or "") for v in values)
        found = self.alias_index.detect(text)
        for code in _codes(text):
            if code.startswith("IL_"):
                found.append("dome_automation_01")
        return list(dict.fromkeys(found))[:8]

    def add(
        self,
        *,
        subject: str,
        subject_label: str,
        attribute: str,
        value: str,
        text: str = "",
        entities: list[str] | None = None,
        codes: list[str] | None = None,
        source_doc: str,
        source_title: str = "",
        source_line: int = 0,
        authority: int = 3,
        status: str = "reference",
        canonical_id: str | None = None,
        chunk_id: str | None = None,
        has_gaps: bool = False,
        origin: str = "structured",
    ) -> dict:
        value = _clip(value, 600)
        text = _clip(text or value, 1100)
        subject = _clip(subject, 160) or source_doc
        subject_label = _clip(subject_label, 160) or subject
        entities = list(dict.fromkeys(entities or []))[:8]
        codes = list(dict.fromkeys(codes or []))[:8]
        key = (
            subject.casefold(),
            attribute,
            (chunk_id or canonical_id or f"{source_doc}|{source_title}|{value[:40]}").casefold(),
        )
        existing = self._by_key.get(key)
        if existing is not None:
            # Та же тройка субъект-атрибут-источник: оставляем более полный текст.
            if len(text) > len(existing["text"]):
                existing.update({"value": value, "text": text})
            return existing

        card_id = "KC-" + hashlib.sha1(
            f"{key[0]}|{key[1]}|{key[2]}|{value[:80]}".encode("utf-8")
        ).hexdigest()[:12]
        while card_id in self._seen:
            card_id += "x"
        self._seen.add(card_id)

        synonyms = " ".join(taxonomy.attributes().get(attribute, {}).get("synonyms", [])[:8])
        search_text = " ".join(
            part
            for part in (
                subject_label, subject, canonical_id or "", " ".join(codes),
                " ".join(entities), taxonomy.label(attribute), synonyms, value, text,
                source_title,
            )
            if part
        )
        card = {
            "card_id": card_id,
            "subject": subject,
            "subject_label": subject_label,
            "attribute": attribute,
            "attribute_label": taxonomy.label(attribute),
            "value": value,
            "text": text,
            "entities": entities,
            "codes": codes,
            "source": {"doc": source_doc, "title": source_title, "line": source_line},
            "authority": authority,
            "status": status,
            "canonical_id": canonical_id,
            "chunk_id": chunk_id,
            "has_gaps": bool(has_gaps),
            "origin": origin,
            "search_text": _clip(search_text, 2400),
        }
        self.cards.append(card)
        self._by_key[key] = card
        if origin != "chunk" and canonical_id:
            self._structured_pairs.add((str(canonical_id), attribute))
        return card

    # ------------------------------------------------------------- структурные

    def from_thresholds(self) -> None:
        for item in self._load("thresholds.json"):
            label = _metric_label(item.get("metric") or "")
            aliases = str(item.get("aliases") or "").strip()
            alias_text = f" Другими словами: {aliases}." if aliases else ""
            value = (
                f"{label + ': ' if label else ''}"
                f"{item.get('metric')} {item.get('op')} {_fmt(item.get('value'))}"
                + (f" в течение {item['for_s']} с" if item.get("for_s") else "")
                + (f" (окно {item['window_s']} с)" if item.get("window_s") else "")
            )
            alternatives = item.get("or") or []
            # В названии порога должны быть все метрики, а не только первая.
            # Иначе вопрос «какая влажность опасна» не находил TH_CONDENSATE:
            # его заголовок говорил только про точку росы.
            label_parts = [label] if label else []
            for alt in alternatives:
                alt_label = _metric_label(alt.get("metric") or "")
                if alt_label and alt_label not in label_parts:
                    label_parts.append(alt_label)
            metrics = [
                str(metric)
                for metric in [item.get("metric")] + [alt.get("metric") for alt in alternatives]
                if metric
            ]
            subject_label = " / ".join(label_parts) or (item.get("metric") or item["id"])
            if metrics:
                subject_label += f" ({', '.join(dict.fromkeys(metrics))})"
            if alternatives:
                value += " или " + " или ".join(
                    f"{_metric_label(alt.get('metric') or '') + ': ' if _metric_label(alt.get('metric') or '') else ''}"
                    f"{alt.get('metric')} {alt.get('op')} {_fmt(alt.get('value'))}"
                    for alt in alternatives
                )
            text = (
                f"{item['id']}: {value}. "
                + (f"Эффект: {item.get('effect')}. " if item.get("effect") else "")
                + (f"Источник: {item.get('source')}." if item.get("source") else "")
                + alias_text
            )
            entities = self._entities(item.get("metric"), item.get("effect"), item["id"])
            entities = list(dict.fromkeys(entities + _metric_entities(item.get("metric"))))[:8]
            if item.get("area") == "climate" and "climate_sensor_01" not in entities:
                entities.append("climate_sensor_01")
            authority = 5 if item.get("status") == "confirmed" else 3
            status = "normative" if item.get("status") == "confirmed" else "hypothesis"
            self.add(
                subject=(
                    f"{item.get('metric') or item['id']} {aliases}".strip()
                ),
                subject_label=subject_label,
                attribute="threshold_value",
                value=value,
                text=text,
                entities=entities,
                codes=[item["id"]],
                source_doc=item.get("source") or "data/curated/thresholds.json",
                source_title=item["id"],
                authority=authority,
                status=status,
                canonical_id=item["id"],
            )
            if item.get("effect"):
                # Порог часто отвечает и на вопрос «что произойдёт»: у него
                # есть последствие. Без отдельной карточки вопрос «что будет
                # после трёх resume» не находил порог, хотя ответ записан
                # именно в его эффекте.
                self.add(
                    subject=(f"{item['id']} {aliases}".strip()),
                    subject_label=f"{item['id']} — последствие",
                    attribute="interlock_effect",
                    value=item["effect"],
                    text=(
                        f"{item['id']} ({value}): последствие — {item['effect']}."
                        + alias_text
                    ),
                    entities=entities,
                    codes=[item["id"]],
                    source_doc=item.get("source") or "data/curated/thresholds.json",
                    source_title=item["id"],
                    authority=authority,
                    status=status,
                    canonical_id=item["id"],
                )

    def from_interlocks(self) -> None:
        for item in self._load("interlocks.json"):
            affected = item.get("affected", [])
            title = f"{item.get('title') or item['id']}"
            entities = self._entities(item["id"], title, item.get("condition"), item.get("effect"))
            entities = list(dict.fromkeys(entities + [a for a in affected if "_" in a]))[:8]
            source = item.get("source") or "data/curated/interlocks.json"
            # Код отказа (E07) живёт в условии, а не в id интерлока. Без этого
            # вопрос «что означает ошибка E07» не находил ответ: точный код
            # вопроса не совпадал с кодами карточки и жёстко отбраковывался.
            codes = _codes(item["id"], title, item.get("condition"), item.get("effect"))
            aliases = str(item.get("aliases") or "").strip()
            alias_text = f" Другими словами: {aliases}." if aliases else ""
            self.add(
                subject=item["id"], subject_label=item["id"], attribute="definition",
                value=f"{item['id']} — {title}",
                text=f"{item['id']} «{title}»." + alias_text,
                entities=entities, codes=codes, source_doc=source,
                source_title=item["id"], authority=5, status="normative", canonical_id=item["id"],
            )
            self.add(
                subject=item["id"], subject_label=item["id"], attribute="interlock_trigger",
                value=f"{item.get('condition') or title}",
                text=f"{item['id']} «{title}» срабатывает при условии: {item.get('condition') or title}."
                + alias_text,
                entities=entities, codes=codes, source_doc=source,
                source_title=item["id"], authority=5, status="normative", canonical_id=item["id"],
            )
            self.add(
                subject=item["id"], subject_label=item["id"], attribute="interlock_effect",
                value=f"{item.get('effect') or ''}; затрагивает: {_fmt(affected)}",
                text=(
                    f"{item['id']} «{title}»: эффект — {item.get('effect') or 'не описан'}; "
                    f"затрагивает: {_fmt(affected)}." + alias_text
                ),
                entities=entities, codes=codes, source_doc=source,
                source_title=item["id"], authority=5, status="normative", canonical_id=item["id"],
            )
            self.add(
                subject=item["id"], subject_label=item["id"], attribute="interlock_clear",
                value=f"{item.get('clear') or 'не описано'}",
                text=f"{item['id']} снимается так: {item.get('clear') or 'не описано'}.",
                entities=entities, codes=codes, source_doc=source,
                source_title=item["id"], authority=5, status="normative", canonical_id=item["id"],
            )

    def from_holds(self) -> None:
        """Пауза станка: hold_level / hold_reason.

        Вопрос «почему принтер встал на паузу» раньше находил только статьи
        каталога отказов. Причина паузы в API — это поле ``hold_reason``;
        без отдельной карточки ответ выглядел уверенным, но не по делу.
        """
        for item in self._load("holds.json"):
            node = item.get("node")
            entities = self._entities(item.get("title"), node, " ".join(item.get("reasons", [])))
            if node and node not in entities:
                entities.append(node)
            entities = list(dict.fromkeys(entities))[:8]
            source = item.get("source") or "data/curated/holds.json"
            reasons = ", ".join(item.get("reasons", []))
            self.add(
                subject=item["id"], subject_label=item.get("title") or item["id"],
                attribute="fault_cause",
                value=(
                    f"Причина паузы — hold_reason: {reasons}. "
                    f"Уровень удержания: {item.get('fields')}"
                ),
                text=(
                    f"{item.get('title')}: причину паузы показывает hold_reason "
                    f"({reasons}); поля: {item.get('fields')}."
                ),
                entities=entities, codes=[item["id"]], source_doc=source,
                source_title=item["id"], authority=5, status="normative",
                canonical_id=item["id"],
            )
            rules = " ".join(item.get("rules", []))
            self.add(
                subject=item["id"], subject_label=item.get("title") or item["id"],
                attribute="fault_action",
                value=rules,
                text=f"{item.get('title')}: {rules}",
                entities=entities, codes=[item["id"]], source_doc=source,
                source_title=item["id"], authority=5, status="normative",
                canonical_id=item["id"],
            )

    def from_faults(self) -> None:
        for item in self._load("faults.json"):
            codes = [item["id"]]
            title = item.get("title") or item["id"]
            entities = self._entities(title, item.get("what_happens"), item.get("how_to_notice"),
                                      item.get("human_action"), item.get("agent_action"))
            source = item.get("source") or "knowledge/09_faults.md"
            status = "catalog"
            self.add(
                subject=item["id"], subject_label=title, attribute="definition",
                value=f"{item['id']} — {title}",
                text=f"{item['id']} «{title}». Категория: {item.get('category')}. "
                     f"Наблюдаемость: {item.get('observability')}.",
                entities=entities, codes=codes, source_doc=source, source_title=item["id"],
                authority=3, status=status, canonical_id=item["id"],
            )
            symptom = " ".join(
                part for part in (
                    f"Что происходит: {item['what_happens']}." if item.get("what_happens") else "",
                    f"Как заметить: {item['how_to_notice']}." if item.get("how_to_notice") else "",
                ) if part
            )
            if symptom:
                self.add(
                    subject=item["id"], subject_label=title, attribute="fault_symptom",
                    value=symptom, text=f"{item['id']} «{title}». {symptom}",
                    entities=entities, codes=codes, source_doc=source, source_title=item["id"],
                    authority=3, status=status, canonical_id=item["id"],
                )
            if item.get("what_happens"):
                # На вопрос «почему» каталог отвечает механизмом отказа.
                # Это версия из каталога, а не доказанный факт, поэтому
                # статус остаётся catalog.
                self.add(
                    subject=item["id"], subject_label=title, attribute="fault_cause",
                    value=item["what_happens"],
                    text=f"{item['id']} «{title}». Вероятный механизм: {item['what_happens']}",
                    entities=entities, codes=codes, source_doc=source, source_title=item["id"],
                    authority=3, status=status, canonical_id=item["id"],
                )
            action = " ".join(
                part for part in (
                    f"Что может сделать человек: {item['human_action']}." if item.get("human_action") else "",
                    f"Что может сделать агент: {item['agent_action']}." if item.get("agent_action") else "",
                ) if part
            )
            if action:
                self.add(
                    subject=item["id"], subject_label=title, attribute="fault_action",
                    value=action, text=f"{item['id']} «{title}». {action}",
                    entities=entities, codes=codes, source_doc=source, source_title=item["id"],
                    authority=3, status=status, canonical_id=item["id"],
                )
            if item.get("must_not"):
                self.add(
                    subject=item["id"], subject_label=title, attribute="fault_forbidden",
                    value=item["must_not"],
                    text=f"{item['id']} «{title}» — чего нельзя допустить: {item['must_not']}",
                    entities=entities, codes=codes, source_doc=source, source_title=item["id"],
                    authority=3, status=status, canonical_id=item["id"],
                )

    def from_diagnostics(self) -> None:
        for item in self._load("diagnostics.json"):
            title = item.get("title") or item["id"]
            entities = self._entities(title, item.get("observation"), item.get("next_steps"),
                                      " ".join(item.get("hypotheses", [])))
            source = item.get("source") or "diagnostics"
            codes = [item["id"]]
            if item.get("observation"):
                self.add(
                    subject=item["id"], subject_label=title, attribute="fault_symptom",
                    value=item["observation"],
                    text=f"Сценарий {item['id']} «{title}». Что видно: {item['observation']}",
                    entities=entities, codes=codes, source_doc=source, source_title=item["id"],
                    authority=4, status="theory", canonical_id=item["id"],
                )
            if item.get("hypotheses"):
                hypotheses = "; ".join(item["hypotheses"][:6])
                self.add(
                    subject=item["id"], subject_label=title, attribute="fault_cause",
                    value=hypotheses,
                    text=f"Сценарий {item['id']} «{title}». Возможные объяснения: {hypotheses}",
                    entities=entities, codes=codes, source_doc=source, source_title=item["id"],
                    authority=4, status="theory", canonical_id=item["id"],
                )
            if item.get("checks"):
                steps = "; ".join(f"{n}) {step}" for n, step in enumerate(item["checks"][:10], 1))
                self.add(
                    subject=item["id"], subject_label=title, attribute="procedure_steps",
                    value=steps,
                    text=f"Сценарий {item['id']} «{title}». Что проверить: {steps}",
                    entities=entities, codes=codes, source_doc=source, source_title=item["id"],
                    authority=4, status="theory", canonical_id=item["id"],
                )

    def from_procedures(self) -> None:
        for item in self._load("procedures.json") + self._load("procedures_curated.json"):
            title = item.get("title") or item["id"]
            entities = self._entities(title, " ".join(str(s) for s in item.get("steps", [])))
            source = item.get("source") or "knowledge/10_procedures.md"
            codes = [item["id"]]
            if item.get("steps"):
                steps = "; ".join(f"{n}) {step}" for n, step in enumerate(item["steps"][:12], 1))
                self.add(
                    subject=item["id"], subject_label=title, attribute="procedure_steps",
                    value=steps, text=f"Процедура «{title}»: {steps}",
                    entities=entities, codes=codes, source_doc=source, source_title=item["id"],
                    authority=5, status="normative", canonical_id=item["id"],
                )
            if item.get("preconditions"):
                pre = "; ".join(_fmt(p) for p in item["preconditions"][:8])
                self.add(
                    subject=item["id"], subject_label=title, attribute="procedure_precondition",
                    value=pre, text=f"Процедура «{title}». Перед началом: {pre}",
                    entities=entities, codes=codes, source_doc=source, source_title=item["id"],
                    authority=5, status="normative", canonical_id=item["id"],
                )
            if item.get("forbidden"):
                forbidden = "; ".join(item["forbidden"][:8])
                self.add(
                    subject=item["id"], subject_label=title, attribute="fault_forbidden",
                    value=forbidden, text=f"Процедура «{title}» — запрещено: {forbidden}",
                    entities=entities, codes=codes, source_doc=source, source_title=item["id"],
                    authority=5, status="normative", canonical_id=item["id"],
                )

    def from_policies(self) -> None:
        for item in self._load("policies.json"):
            rule = item.get("rule") or item["id"]
            applies = _action_words(item.get("applies_to"))
            aliases = str(item.get("aliases") or "").strip()
            entities = self._entities(rule, item.get("action"), " ".join(item.get("applies_to", [])))
            if item.get("applies_to"):
                entities = list(dict.fromkeys(entities + ["production_tools"]))[:8]
            self.add(
                subject=item["id"], subject_label=rule[:80], attribute="policy_rule",
                value=(
                    rule + (f" → {item['action']}" if item.get("action") else "")
                    + (f"; касается: {applies}" if applies else "")
                ),
                text=(
                    f"{item['id']} ({item.get('scope')}): {rule}."
                    + (f" Действие: {item['action']}." if item.get("action") else "")
                    + (f" Касается: {applies}." if applies else "")
                    + (f" Статус: {item.get('verdict')}." if item.get("verdict") else "")
                    # Бытовые формулировки правила. Без них «стол занят
                    # деталью» не находило POL-011 «start_job только при
                    # bed_clear = true»: общих слов в тексте не было.
                    + (f" Другими словами: {aliases}." if aliases else "")
                ),
                entities=entities, codes=[item["id"]],
                source_doc=item.get("source") or "data/curated/policies.json",
                source_title=item["id"], authority=5, status="normative", canonical_id=item["id"],
            )

    def from_gaps(self) -> None:
        for item in self._load("gaps.json"):
            text = (
                f"{item['id']}: {item.get('description')}. "
                + (f"Конфликт: {item['conflict']}. " if item.get("conflict") else "")
                + (f"Как закрыть: {item['resolution']}." if item.get("resolution") else "")
            )
            self.add(
                subject=item["id"], subject_label=item.get("area") or item["id"],
                attribute="gap", value=text,
                text=text,
                entities=self._entities(item.get("description"), item.get("conflict"), " ".join(item.get("blocks", []))),
                codes=[item["id"]],
                source_doc=item.get("source") or "data/curated/gaps.json",
                source_title=item["id"], authority=4, status="gap", canonical_id=item["id"],
                has_gaps=True,
            )

    def from_faq(self) -> None:
        for item in self._load("faq.json"):
            self.add(
                subject=item.get("question") or item["id"],
                subject_label=_clip(item.get("question"), 120),
                attribute="definition",
                value=item.get("answer") or "",
                text=f"Вопрос: {item.get('question')}. Ответ: {item.get('answer')}",
                entities=self._entities(item.get("question"), item.get("answer")),
                codes=[item["id"]],
                source_doc=item.get("source") or "knowledge/11_reference.md",
                source_title=_clip(item.get("question"), 120),
                authority=4, status="reference", canonical_id=item["id"],
            )

    def from_glossary(self) -> None:
        for item in self._load("glossary.json"):
            self.add(
                subject=item["term"], subject_label=item["term"], attribute="definition",
                value=item["definition"],
                text=f"«{item['term']}»: {item['definition']}",
                entities=self._entities(item["term"], item["definition"]),
                source_doc=item.get("source") or "knowledge/11_reference.md",
                source_title=item["term"], authority=4, status="reference",
                canonical_id="G-" + md.slug(item["term"], 40),
            )

    def from_nodes(self) -> None:
        for node in self.nodes:
            node_id = node["node_id"]
            label = node.get("title") or node_id
            entities = [node_id] + [node.get("code_name") or ""] + list(node.get("aliases", []))
            entities = [e for e in dict.fromkeys(entities) if e]
            self.add(
                subject=node_id, subject_label=label, attribute="definition",
                value=f"{label} ({node_id}), система: {node.get('system_ru') or node.get('system')}",
                text=f"{label} — узел `{node_id}`, система {node.get('system_ru') or node.get('system')}.",
                entities=entities, codes=[node_id], source_doc="data/nodes.json",
                source_title=node_id, authority=4, status="reference", canonical_id=node_id,
            )
            for param in node.get("params", []):
                unit = f" {param['unit']}" if param.get("unit") else ""
                value = f"{param['key']}{unit} — {param.get('kind')}"
                self.add(
                    subject=node_id, subject_label=label, attribute="measurement_spec",
                    value=value,
                    text=f"{label}: параметр телеметрии `{param['key']}`{unit} "
                         f"(тип {param.get('kind')}, доверие {param.get('trust')}).",
                    entities=entities[:6],
                    codes=[param["key"]] if re.search(r"[_0-9]", param["key"]) else [],
                    source_doc="data/nodes.json", source_title=node_id,
                    authority=4, status="reference", canonical_id=node_id,
                )
            for control in node.get("controls", []):
                who = "только Оператор" if control.get("allowed_for") == "operator" else "агент"
                confirmed = "подтверждено в API" if control.get("api_confirmed") else "не подтверждено в API"
                value = f"{control['action']} ({control.get('label')}; {who}; {confirmed})"
                aliases = ACTION_ALIASES.get(control["action"], "")
                self.add(
                    subject=node_id, subject_label=label, attribute="control_capability",
                    value=value,
                    text=f"{label}: команда `{control['action']}` — {control.get('label')}; "
                         f"доступна: {who}; {confirmed}."
                         + (f" Другими словами: {aliases}." if aliases else ""),
                    entities=entities[:6], codes=[control["action"]],
                    source_doc="data/nodes.json", source_title=node_id,
                    authority=4, status="reference", canonical_id=node_id,
                )

    def from_equipment(self) -> None:
        for item in self._load("equipment.json"):
            row = item.get("row") or []
            columns = item.get("columns") or []
            value = "; ".join(
                f"{col}: {cell}" for col, cell in zip(columns, row) if str(cell).strip()
            ) or " | ".join(str(cell) for cell in row)
            self.add(
                subject=item.get("section") or "оборудование",
                subject_label=item.get("section") or "оборудование",
                attribute="equipment_spec", value=value,
                text=f"{item.get('section')}: {value}",
                entities=self._entities(value, item.get("section")),
                source_doc=item.get("source") or "knowledge/02_station.md",
                source_title=item.get("section") or "", authority=4, status="reference",
            )

    def from_conflicts(self) -> None:
        for item in self._load("conflicts.json"):
            values = "; ".join(
                f"{entry['value']} [{entry['source']}]" for entry in item.get("values", [])
            )
            self.add(
                subject=item["id"], subject_label=item.get("description", "")[:80],
                attribute=item.get("attribute") or "gap",
                value=item.get("description") or values,
                text=(
                    f"{item['id']}: {item.get('description')} "
                    f"Расхождение: {values}. "
                    f"Решение: {item.get('resolution')}"
                ),
                entities=item.get("subject_entities", []),
                codes=[item["id"]],
                source_doc=item.get("source") or "data/curated/conflicts.json",
                source_title=item["id"], authority=5, status="conflict",
                canonical_id=item["id"], has_gaps=True,
            )

    def from_chunks(self) -> None:
        for filename in ("knowledge.jsonl", "episodes.jsonl"):
            path = CHUNKS / filename
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("is_duplicate_of"):
                        continue
                    title = chunk.get("title") or ""
                    body = chunk.get("text") or ""
                    entities = chunk.get("entities") or []
                    codes = _codes(chunk.get("canonical_id") or "", title, body[:400])
                    # Дата смены — точный идентификатор карточки истории.
                    for date in re.findall(
                        r"\b20\d{2}-\d{2}-\d{2}\b",
                        f"{chunk.get('doc') or ''} {chunk.get('canonical_id') or ''} {title}",
                    ):
                        if date not in codes:
                            codes.append(date)
                    authority = _authority(chunk.get("authority"), chunk.get("source_priority", 3))

                    if chunk.get("type") == "episode":
                        # Смена содержит две разные вещи: факты автоматики
                        # (срабатывания, команды) и мнение жильца. Смешивать
                        # их в одну карточку нельзя: журнал — факт, запись
                        # жильца — мнение.
                        facts, marker, opinion = body.partition("Бортовой журнал")
                        facts = facts.strip() or body
                        self.add(
                            subject=chunk.get("canonical_id") or chunk["doc"],
                            subject_label=title or chunk["doc"],
                            attribute="history_event",
                            value=md.first_sentences(facts, 5, 700),
                            text=facts,
                            entities=entities,
                            codes=codes,
                            source_doc=chunk["doc"],
                            source_title=title,
                            source_line=chunk.get("line", 0),
                            authority=authority,
                            status="history",
                            canonical_id=chunk.get("canonical_id"),
                            chunk_id=chunk["chunk_id"],
                            origin="chunk",
                        )
                        if marker and opinion.strip():
                            self.add(
                                subject=chunk.get("canonical_id") or chunk["doc"],
                                subject_label=f"{title} — бортовой журнал",
                                attribute="history_event",
                                value=md.first_sentences(opinion.strip(), 3, 600),
                                text=opinion.strip(),
                                entities=entities,
                                codes=codes,
                                source_doc=chunk["doc"],
                                source_title=f"{title} — бортовой журнал",
                                source_line=chunk.get("line", 0),
                                authority=2,
                                status="opinion",
                                canonical_id=chunk.get("canonical_id"),
                                chunk_id=chunk["chunk_id"] + "-opinion",
                                origin="chunk",
                            )
                        continue

                    # Абзац не знает, на какой атрибут он отвечает. Атрибут
                    # ставится только по содержательному сигналу; если его
                    # нет — section_text (он не может дать ANSWERABLE).
                    attribute = _chunk_attribute(chunk)
                    canonical = chunk.get("canonical_id")
                    # Точный структурный факт уже есть: абзац — его копия.
                    if canonical and (str(canonical), attribute) in self._structured_pairs:
                        continue
                    status = STATUS_MAP.get(chunk.get("verification_status", ""), "reference")
                    self.add(
                        subject=(entities[0] if entities else (canonical or chunk["doc"])),
                        subject_label=title or chunk["doc"],
                        attribute=attribute,
                        value=md.first_sentences(body, 2, 520),
                        text=body,
                        entities=entities,
                        codes=codes,
                        source_doc=chunk["doc"],
                        source_title=title,
                        source_line=chunk.get("line", 0),
                        authority=authority,
                        status=status,
                        canonical_id=canonical,
                        chunk_id=chunk["chunk_id"],
                        has_gaps=bool(chunk.get("has_gaps")),
                        origin="chunk",
                    )

    # ------------------------------------------------------------------- сборка

    def build(self) -> list[dict]:
        self.from_thresholds()
        self.from_interlocks()
        self.from_holds()
        self.from_faults()
        self.from_diagnostics()
        self.from_procedures()
        self.from_policies()
        self.from_gaps()
        self.from_faq()
        self.from_glossary()
        self.from_nodes()
        self.from_equipment()
        self.from_conflicts()
        self.from_chunks()
        self.cards.sort(key=lambda c: (c["attribute"], c["subject"], c["card_id"]))
        return self.cards


def build_cards() -> list[dict]:
    cards = CardBuilder().build()
    CARDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CARDS_PATH.write_text(json.dumps(cards, ensure_ascii=False, indent=1), encoding="utf-8")
    return cards


def load_cards(path: Path | None = None) -> list[dict]:
    path = Path(path or CARDS_PATH)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_cards() -> None:
    cards = build_cards()
    by_attr: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for card in cards:
        by_attr[card["attribute"]] = by_attr.get(card["attribute"], 0) + 1
        by_status[card["status"]] = by_status.get(card["status"], 0) + 1
    print(f"cards       {len(cards)}")
    print("атрибуты:")
    for attribute, count in sorted(by_attr.items(), key=lambda kv: -kv[1]):
        print(f"  {attribute:<22} {count}")
    print("статусы:")
    for status, count in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"  {status:<22} {count}")
