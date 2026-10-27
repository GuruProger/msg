"""Проверка покрытия: отвечает ли найденное на заданный вопрос.

Идея простая и не требует нейросетей:

1. разобрать вопрос на **слоты** — о каком объекте спрашивают и какой атрибут
   хотят узнать;
2. для каждой карточки проверить, что она привязана к этому объекту и этому
   атрибуту;
3. если привязки нет — честно сказать «в корпусе нет ответа», а не выдавать
   похожий абзац за факт.

Вердикты:

* ``ANSWERABLE``   — есть карточка(и) с нужным субъектом и атрибутом;
* ``PARTIAL``      — ответ есть частично (нет части слотов, источник неполный);
* ``NO_RELEVANT_EVIDENCE`` — релевантных свидетельств нет;
* ``CONFLICT``     — источники противоречат друг другу.

Уверенность выводится из вердикта, а не из «похожести текста». Поэтому
``PARTIAL`` не может получить 0.86, а ``NO_RELEVANT_EVIDENCE`` — вообще
ненулевую уверенность.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from . import taxonomy, vectors
from .retrieval import overlap, stems

CODE_RE = re.compile(
    r"\b(?:IL_[A-Z0-9_]+|TH_[A-Z0-9_]+|GAP-\d+|F-\d+\.\d+|D-\d{2}-\d{2}|"
    r"P-10-[A-Za-zА-Яа-я0-9_-]+|FAQ-\d+|POL-\d+|E\d{2})\b",
    re.IGNORECASE,
)
# Даты смен — тоже точный идентификатор: «что было 2026-10-15» не должно
# находить запись любой другой смены только потому, что слова похожи.
DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")

ANSWERABLE = "ANSWERABLE"
PARTIAL = "PARTIAL"
NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"
CONFLICT = "CONFLICT"

FALLBACK_ATTRIBUTES = ("policy_rule", "fault_forbidden", "interlock_effect", "interlock_trigger")
# Атрибуты-«подпорки»: они уместны рядом с ответом, но не заменяют прямой
# ответ на вопрос («что делать» → процедура, а не только запрет).
GENERIC_SIBLINGS = {"policy_rule", "fault_forbidden", "procedure_precondition"}
# Если основная трактовка вопроса даёт слабое свидетельство, пробуем
# соседние по смыслу атрибуты. Это не «угадывание»: карточка всё равно должна
# быть привязана к субъекту и заметно перекрывать лучшую карточку основной
# трактовки.
SECONDARY_ATTRIBUTES = (
    "policy_rule",
    "fault_action",
    "procedure_steps",
    "threshold_value",
    "control_capability",
    "fault_forbidden",
)
USABLE_FOR_ANSWER = {"normative", "reference", "theory", "catalog"}
WEAK_STATUS = {"hypothesis", "opinion", "incomplete"}


@dataclass(slots=True)
class Slots:
    question: str
    attributes: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    off_domain: bool = False
    generic: bool = False

    @property
    def primary(self) -> str:
        return self.attributes[0] if self.attributes else "section_text"


@dataclass(slots=True)
class Assessment:
    verdict: str
    coverage: float
    slots: Slots
    cards: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    support: int = 0
    reason: str = ""
    fallback: bool = False

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "coverage": round(self.coverage, 3),
            "attributes": self.slots.attributes,
            "entities": self.slots.entities,
            "codes": self.slots.codes,
            "missing": self.missing,
            "support": self.support,
            "conflicts": self.conflicts,
            "off_domain": self.slots.off_domain,
            "fallback": self.fallback,
            "reason": self.reason,
        }


def _normalize_codes(code: str) -> str:
    return code.upper() if code[:3].upper() in {"IL_", "TH_"} else code


_FAQ_PREFIX = re.compile(
    r"^(?:что\s+(?:такое|это|за|значит|означает)|кто\s+(?:такой|такая|такое)|"
    r"определение|расшифровка)\s+",
    re.IGNORECASE,
)


@lru_cache(maxsize=16384)
def _subject_variants_cached(label: str, subject: str) -> tuple[str, ...]:
    out: list[str] = []
    for value in (label, subject):
        text = vectors.normalize_text(value)
        text = _FAQ_PREFIX.sub("", text).strip()
        base = re.split(r"\s*[(:—]\s*", text)[0].strip()
        candidates = [text, base, *re.split(r"\s*[/,_]\s*", base)]
        # Метрика порога почти всегда записана в скобках латиницей:
        # «попытки resume (printer_3d_01.resume_attempts)». Без разбора
        # скобок вопрос «сколько раз можно нажать resume» не привязывался к
        # порогу, хотя нужное слово буквально стоит в названии.
        for group in re.findall(r"\(([^)]*)\)", text):
            candidates.extend(re.split(r"\s*[.,/]\s*", group))
        for part in candidates:
            part = part.strip(" ?.!:—-/")
            if len(part) >= 3 and part not in out:
                out.append(part)
    return tuple(out)


def _subject_variants(card: dict) -> tuple[str, ...]:
    """Нормализованные варианты названия субъекта.

    У FAQ название — целый вопрос («Что такое резервный сценарий?»), у порога —
    «влажность / точка росы (climate.humidity_pct, …)». Без нормализации
    точное совпадение субъекта не срабатывало ни там, ни там.
    """
    return _subject_variants_cached(
        str(card.get("subject_label") or ""),
        str(card.get("subject") or ""),
    )


def _same_stem(left: str, right: str) -> bool:
    """Одна и та же лексема при грубом стемминге.

    «попыток» → «попыт», «попытки» → «попытк»: основа режет окончания
    по-разному. Одна правка (вставка/замена) — это ещё та же лексема; две и
    больше — уже разные слова («перегрев»/«перегруз»).
    """
    if left == right:
        return True
    # Префикс — это та же основа, даже если стеммер обрезал её по-разному:
    # «систем» (из «системы») и «сист» (из «систем»). Раньше проверка длины
    # отсекала такое совпадение, и FAQ «Из каких систем состоит Купол?» не
    # привязывался к вопросу «какие системы входят в Купол?».
    if left.startswith(right) or right.startswith(left):
        return min(len(left), len(right)) >= 4
    if len(left) < 5 or len(right) < 5 or left[:3] != right[:3]:
        return False
    if abs(len(left) - len(right)) > 1:
        return False
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (a != b),
            ))
        previous = current
    return previous[-1] <= 1


# Вопросительные слова не описывают субъект. После стемминга они схлопываются
# в несколько основ («какие/каких/какой» → «как»), и без фильтра название
# «Какие зоны доступа есть в Куполе?» совпадало с вопросом «Какие системы
# входят в купол?» ровно так же, как правильный FAQ «Из каких систем состоит
# Купол?». Значимая часть субъекта должна весить больше служебной.
_QUESTION_STEMS = {"как", "кто", "где", "когд", "скольк", "поч", "куд", "зач"}


@lru_cache(maxsize=32768)
def _subject_variant_score(question: str, variant: str) -> float:
    q = vectors.normalize_text(question)
    q_stems = [stem for stem in stems(q) if stem not in _QUESTION_STEMS]
    variant_stems = [
        stem for stem in stems(variant) if stem not in _QUESTION_STEMS
    ]
    if not variant_stems:
        return 0.0
    if len(variant) >= 4 and variant in q:
        score = 1.0
    else:
        common = [
            stem for stem in variant_stems
            if any(_same_stem(stem, other) for other in q_stems)
        ]
        if not common:
            return 0.0
        if len(variant_stems) >= 2 and len(common) < 2:
            return 0.0
        score = len(common) / len(variant_stems)
    # Однословное название («Купол») — полный балл только если вопрос не
    # добавляет других содержательных слов: иначе «высота купола» получала бы
    # 1.0 от FAQ «Что за купол».
    if len(variant_stems) == 1 and (set(q_stems) - set(variant_stems)):
        score = min(score, 0.6)
    return score


def subject_match(question: str, card: dict) -> float:
    """Насколько название субъекта карточки совпадает с темой вопроса.

    Это отдельная ось, потому что «Что такое Сбой?» и «Что такое зависание
    скрипта?» лексически похожи, но отвечают на разные вопросы. Раньше
    глоссарная статья про сбой проигрывала каталогу отказов.
    """
    return max(
        (_subject_variant_score(question, variant) for variant in _subject_variants(card)),
        default=0.0,
    )


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Длина наибольшей общей подпоследовательности основ."""
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for token in left:
        current = [0] * (len(right) + 1)
        for index, other in enumerate(right, 1):
            if token == other:
                current[index] = prev[index - 1] + 1
            else:
                current[index] = max(prev[index], current[index - 1])
        prev = current
    return prev[-1]


@lru_cache(maxsize=8192)
def phrase_overlap(question: str, text: str) -> float:
    """Доля формулировки вопроса, найденная в тексте в правильном порядке.

    Нужна для правил и процедур: «снять деталь со стола» — это не набор
    отдельных слов, а фраза. Соседние пары слов были слишком строгими:
    «резкое падение генерации» не совпадало с «внезапное падение солнечной
    генерации», хотя это один и тот же факт. LCS сохраняет порядок слов, но
    допускает вставки.
    """
    q = [vectors.stem(t) for t in vectors.tokenize(question)]
    d = [vectors.stem(t) for t in vectors.tokenize(text)][:160]
    if len(q) < 2 or len(d) < 2:
        return 0.0
    return _lcs_length(q, d) / len(q)


CODE_CLASS_ATTRIBUTES = (
    ("GAP-", "gap"),
    ("POL-", "policy_rule"),
    ("F-", "definition"),
    ("D-", "procedure_steps"),
    ("TH_", "threshold_value"),
    ("IL_", "definition"),
    ("FAQ-", "definition"),
    ("P-", "procedure_steps"),
    ("E", "definition"),
)


def extract_slots(question: str, entities: list[str] | None = None) -> Slots:
    attributes = taxonomy.detect_attributes(question)
    if not attributes:
        attributes = ["section_text"]
    off_domain = any(taxonomy.is_off_domain(a) for a in attributes)
    normalized = vectors.normalize_text(question)
    for term in taxonomy.payload().get("off_domain_terms", []):
        if vectors.normalize_text(term) in normalized:
            off_domain = True
            if "off_topic" not in attributes:
                attributes.insert(0, "off_topic")
            break
    codes = list(
        dict.fromkeys(
            _normalize_codes(c) for c in CODE_RE.findall(question)
        )
    )
    codes += [date for date in DATE_RE.findall(question) if date not in codes]
    # Класс точного идентификатора подсказывает, что именно спрашивают:
    # GAP-007 — пробел, POL-001 — правило, TH_* — порог. Формулировка
    # вопроса при этом важнее: «чем отличается» остаётся сравнением.
    if "comparison" not in attributes:
        for code in codes:
            for prefix, attribute in CODE_CLASS_ATTRIBUTES:
                if code.upper().startswith(prefix) and attribute not in attributes:
                    attributes.append(attribute)

    low = normalized
    # Точная дата + «смена/журнал/история» — это вопрос к записи истории,
    # даже если в нём есть слово «интерлоки». Без этого карточка смены
    # отбрасывалась по атрибуту, и ответ «какие интерлоки срабатывали
    # 2026-10-16» оказывался пустым.
    if codes and any(cue in low for cue in ("смен", "журнал", "истори", "logbook")):
        if "history_event" not in attributes:
            attributes.insert(0, "history_event")
    # «Когда срабатывает» относится к интерлоку, только если в вопросе
    # действительно назван интерлок/блокировка. «Куда сообщить при
    # срабатывании аварийной остановки» — это процедура, а не триггер.
    if "interlock_trigger" in attributes:
        has_interlock = (
            any(code.upper().startswith("IL_") for code in codes)
            or "интерлок" in low
            or "блокиров" in low
        )
        if not has_interlock:
            attributes.remove("interlock_trigger")
    # «Куда/кому сообщить» — шаг процедуры, а не характеристика узла.
    if any(
        cue in low
        for cue in ("куда сообщить", "кому сообщить", "куда доложить", "кому доложить")
    ) and "procedure_steps" not in attributes:
        attributes.append("procedure_steps")
    # Сравнение истории и каталога отвечает карточка истории/правила,
    # поэтому она должна попасть в рассмотрение рядом со сравнением.
    if "comparison" in attributes and ("истори" in low or "каталог" in low):
        if "history_event" not in attributes:
            attributes.append("history_event")
    # Сравнение двух отказов («чем отличается завышение от занижения»)
    # отвечают не заголовки каталога, а карточки симптома/причины: именно в них
    # записано «что происходит» и «как заметить».
    if "comparison" in attributes and any(
        cue in low for cue in ("завыш", "заниж", "отказ", "неисправн", "ложн", "дефект")
    ):
        for attribute in ("fault_symptom", "fault_cause"):
            if attribute not in attributes:
                attributes.append(attribute)
    # Экипаж — документированная сущность (FAQ-003/004/005), хотя атрибут
    # «численность» помечен off-domain: в корпусе нет штатного расписания.
    # «Сколько человек в экипаже» — вопрос к FAQ, а «сколько сотрудников»
    # остаётся вопросом о данных, которых нет.
    if "экипаж" in low and "капитан" not in low and any(
        cue in low for cue in ("сколько", "рол", "состав", "какие", "кто входит")
    ):
        attributes = [a for a in attributes if a != "headcount"]
        if "definition" not in attributes:
            attributes.insert(0, "definition")
        off_domain = any(taxonomy.is_off_domain(a) for a in attributes)
        attributes = [a for a in attributes if a != "off_topic"]
    # «Датчик завышает/занижает показания» — это симптом из каталога отказов,
    # а не запрос живого значения. Без этого «как понять, что датчик врёт»
    # уходило в live_state и не находило F-1.1.
    if any(cue in low for cue in ("завыша", "занижа", "врёт", "врет", "физически невозмож")) and (
        "датчик" in low or "показан" in low or "значени" in low
    ):
        attributes = [a for a in attributes if a != "live_state"]
        if "fault_symptom" not in attributes:
            attributes.insert(0, "fault_symptom")
    if not attributes:
        attributes = ["section_text"]
    return Slots(
        question=question,
        attributes=attributes,
        entities=list(dict.fromkeys(entities or []))[:6],
        codes=codes,
        keywords=taxonomy.keyword_terms(question),
        off_domain=off_domain,
        generic=attributes == ["section_text"],
    )


def _attribute_match(requested: str, card_attribute: str) -> float:
    if requested == card_attribute:
        return 1.0
    if requested == "section_text":
        return 0.65
    if requested == "fault_action" and card_attribute == "procedure_steps":
        # «Что делать» — это и действие из каталога отказов, и шаг процедуры.
        # Процедура не хуже каталога: иначе общий FAQ-ответ вытеснял точные
        # шаги, а вердикт оставался PARTIAL при готовом ответе.
        return 1.0
    if card_attribute in taxonomy.family_members(requested):
        return 0.65
    return 0.0


def score_card(slots: Slots, card: dict) -> dict:
    """Оценка карточки по шести осям. Без «магии» и без нейросетей.

    * ``attribute`` — отвечает ли карточка на запрошенный аспект;
    * ``entity``    — тот ли это объект;
    * ``code``      — точный идентификатор (``IL_CO2``, ``F-1.7``, дата смены);
    * ``lexical``   — перекрытие значимых слов;
    * ``subject``   — совпадение названия субъекта (глоссарий, код, узел);
    * ``phrase``    — совпадение формулировки, а не отдельных слов.
    """
    # Вопрос может просить несколько вещей сразу («что такое X и когда он
    # срабатывает»). Карточка проверяется по каждому запрошенному атрибуту,
    # и берётся лучший.
    attribute = 0.0
    for requested in slots.attributes or [slots.primary]:
        attribute = max(attribute, _attribute_match(requested, card.get("attribute", "")))
    if attribute <= 0:
        return {"total": 0.0, "attribute": 0.0, "entity": 0.0, "code": 0.0,
                "lexical": 0.0, "subject": 0.0, "phrase": 0.0, "rejected": "attribute"}

    is_chunk = card.get("origin") == "chunk"
    incidental_mention = is_chunk and card.get("attribute") != "history_event"
    card_entities = set(card.get("entities") or [])
    q_entities = set(slots.entities)
    if q_entities and card_entities:
        if q_entities & card_entities:
            # У абзаца сущность часто упомянута попутно, у структурной
            # карточки — это её адрес. Поэтому вес разный.
            entity = 0.55 if incidental_mention else 1.0
        elif card.get("attribute") in {"policy_rule", "fault_forbidden", "procedure_precondition"}:
            # Общее правило безопасности может не называть конкретный узел
            # («пайка только при работающей вытяжке»). Это не повод его
            # выбрасывать, но и не полная привязка к объекту.
            entity = 0.35
        else:
            # Вопрос про конкретный объект, а карточка про другой — это не
            # «слабое совпадение», это другой объект.
            return {"total": 0.0, "attribute": attribute, "entity": 0.0,
                    "code": 0.5, "lexical": 0.0, "subject": 0.0, "phrase": 0.0,
                    "rejected": "entity"}
    elif q_entities:
        entity = 0.20 if incidental_mention else 0.30
    else:
        entity = 0.5

    card_codes = {str(c) for c in (card.get("codes") or [])}
    q_codes = set(slots.codes)
    if q_codes and card_codes:
        if q_codes & card_codes:
            code = 1.0
        else:
            return {"total": 0.0, "attribute": attribute, "entity": entity,
                    "code": 0.0, "lexical": 0.0, "subject": 0.0, "phrase": 0.0,
                    "rejected": "code"}
    elif q_codes:
        # Вопрос называет точный код, а карточка его не содержит: это уже не
        # доказательство, даже если слова похожи.
        return {"total": 0.0, "attribute": attribute, "entity": entity,
                "code": 0.0, "lexical": 0.0, "subject": 0.0, "phrase": 0.0,
                "rejected": "code"}
    else:
        code = 0.5

    body = " ".join(
        str(part or "")
        for part in (
            card.get("subject_label"),
            card.get("value"),
            card.get("text"),
            " ".join(card.get("codes") or []),
        )
    )
    lexical = overlap(stems(slots.question), body)
    subject = subject_match(slots.question, card)
    phrase = phrase_overlap(slots.question, body)

    total = (
        0.36 * attribute
        + 0.18 * entity
        + 0.12 * code
        + 0.10 * lexical
        + 0.12 * subject
        + 0.12 * phrase
    )
    # Точная формулировка и точное название субъекта — сильные сигналы,
    # которые нельзя растворять в среднем: они подтверждают, что найден
    # именно этот факт, а не просто похожий абзац.
    if phrase >= 0.45:
        total += 0.06 * phrase
    if subject >= 0.99:
        total += 0.04

    direct_focus = subject >= 0.99 or phrase >= 0.45
    if lexical < 0.08 and not (q_codes & card_codes) and not direct_focus:
        total *= 0.45
    elif (
        lexical < 0.22
        and not (q_codes & card_codes)
        and not direct_focus
        and card.get("attribute") not in {"policy_rule", "fault_forbidden", "procedure_precondition"}
    ):
        total *= 0.70
    return {
        "total": round(total, 4),
        "attribute": round(attribute, 3),
        "entity": round(entity, 3),
        "code": round(code, 3),
        "lexical": round(lexical, 3),
        "subject": round(subject, 3),
        "phrase": round(phrase, 3),
        "rejected": "",
    }



def _annotate(slots: Slots, cards: list[dict]) -> list[dict]:
    out = []
    for card in cards:
        score = score_card(slots, card)
        if score["total"] <= 0:
            continue
        item = dict(card)
        item["_coverage"] = score
        item["_match"] = score["total"]
        out.append(item)
    out.sort(key=lambda c: (-c["_match"], -int(c.get("authority") or 3), c["card_id"]))
    return out


def _conflicts(cards: list[dict]) -> list[dict]:
    found = [c for c in cards if c.get("status") == "conflict"]
    unique: dict[str, dict] = {}
    for card in found:
        unique.setdefault(card.get("canonical_id") or card["card_id"], card)
    return [
        {
            "id": card.get("canonical_id") or card["card_id"],
            "subject": card.get("subject"),
            "description": card.get("value"),
            "source": card.get("source", {}).get("doc"),
        }
        for card in unique.values()
    ]


def assess(
    question: str,
    cards: list[dict],
    *,
    slots: Slots | None = None,
    entities: list[str] | None = None,
    all_cards: list[dict] | None = None,
) -> Assessment:
    slots = slots or extract_slots(question, entities=entities)
    ordered = _annotate(slots, cards)
    # Точный код/дата из вопроса обязан присутствовать в рассмотрении, даже
    # если канал поиска не поднял карточку в top-k. «Что было в смену
    # 2026-10-15» не должно зависеть от того, обогнала ли заметка жильца
    # фактический отчёт смены.
    if all_cards and slots.codes:
        known = {c["card_id"] for c in ordered}
        wanted = {code.upper() for code in slots.codes}
        extra = [
            c for c in all_cards
            if c["card_id"] not in known
            and wanted & {str(x).upper() for x in (c.get("codes") or [])}
        ][:20]
        if extra:
            ordered = _annotate(slots, list(cards) + extra)

    if slots.off_domain:
        # «Вне предметной области» — это сигнал, а не приговор. В корпусе
        # есть статья «Коллизия IP-адресов», хотя слово «IP-адрес» помечено
        # как учётные данные. Отказываемся только тогда, когда сильного
        # свидетельства действительно нет.
        requested = [
            a for a in slots.attributes
            if not taxonomy.is_off_domain(a) and a != "section_text"
        ]
        probe = next(
            (
                c for c in ordered
                if c["_coverage"].get("attribute", 0) >= 0.65
                and c["_coverage"].get("lexical", 0) >= 0.30
                and (
                    c["_coverage"].get("subject", 0) >= 0.90
                    # Парафраз («потеря Wi-Fi» → «Потеря Wi-Fi») может не
                    # дать точного названия субъекта, но совпадает фразой.
                    or c["_coverage"].get("phrase", 0) >= 0.45
                )
            ),
            None,
        )
        if not probe or not requested:
            return Assessment(
                verdict=NO_RELEVANT_EVIDENCE,
                coverage=0.0,
                slots=slots,
                missing=list(slots.attributes),
                reason="вопрос относится к данным, которых в корпусе нет "
                       f"({', '.join(slots.attributes)})",
            )
        # В корпусе нашлось прямое свидетельство: отвечаем на содержательный
        # атрибут, а не на «вне домена». Оставляем только слоты, которые
        # действительно подтверждает найденная карточка: «какие действия при
        # потере Wi-Fi» не должно тянуть за собой случайный control_capability.
        probe_attribute = probe.get("attribute")
        slots.attributes = [
            a for a in requested
            if a == probe_attribute
            or _attribute_match(a, probe_attribute) >= 0.65
            or _attribute_match(probe_attribute, a) >= 0.65
        ] or [probe_attribute]
        slots.off_domain = False

    attributes = slots.attributes or [slots.primary]
    # «Мнение жильца» — слабый статус для обычного факта, но для вопроса
    # «что записал жилец в журнал» это и есть запрошенное содержание.
    # Поэтому слабые карточки допускаются к слоту истории только тогда, когда
    # вопрос действительно про мнение/запись жильца. Для «что было в смену»
    # нужен фактический отчёт смены, а не заметка жильца.
    question_low = slots.question.casefold()
    opinion_requested = any(
        cue in question_low
        for cue in ("мнение", "жилец", "записал", "писал", "бортовой журнал")
    )
    allow_weak_history = "history_event" in attributes and opinion_requested
    usable = [
        c for c in ordered
        if c.get("status") not in WEAK_STATUS
        or (allow_weak_history and c.get("attribute") == "history_event")
    ]
    top_pool = usable or ordered
    if not top_pool:
        # Резерв: вопрос просит эффект, а в корпусе есть только правило/запрет.
        # Это честный PARTIAL: «прямого описания последствий нет, но есть норма».
        fallback_slots = Slots(
            question=slots.question,
            attributes=[a for a in FALLBACK_ATTRIBUTES if a != slots.primary],
            entities=slots.entities,
            codes=slots.codes,
            keywords=slots.keywords,
        )
        fallback = _annotate(fallback_slots, all_cards or cards)
        forced = [c for c in fallback if c["_match"] >= 0.45]
        if forced:
            return Assessment(
                verdict=PARTIAL,
                coverage=max(c["_match"] for c in forced),
                slots=slots,
                cards=forced[:8],
                missing=list(attributes),
                support=len(forced),
                reason="прямого ответа на запрошенный атрибут нет; "
                       "найдены связанные правила/условия",
                fallback=True,
            )
        return Assessment(
            verdict=NO_RELEVANT_EVIDENCE,
            coverage=0.0,
            slots=slots,
            missing=list(attributes),
            reason="ни одна карточка не привязана к субъекту и атрибуту вопроса",
        )

    # Слоты вопроса проверяются по отдельности. Один «громкий» посторонний
    # атрибут не должен вытеснять прямой ответ на другой слот.
    covered_by_slot: dict[str, dict] = {}
    for attribute in attributes:
        candidates = [
            c for c in top_pool
            if _attribute_match(attribute, c["attribute"]) >= 0.65
        ]
        if candidates:
            # Слот «что делать» может закрываться и запретом, и процедурой.
            # Процедура — прямой ответ; общее правило/запрет — поддержка.
            # Из двух равнозначных по атрибуту карточек выигрывает более
            # релевантная, а не формально точная: иначе карточка каталога
            # (0.27) вытесняла точную процедуру (0.49).
            candidates.sort(
                key=lambda c: (
                    0 if c["attribute"] == attribute
                    or _attribute_match(attribute, c["attribute"]) >= 1.0
                    else 1 if c["attribute"] not in GENERIC_SIBLINGS
                    else 2,
                    -c["_coverage"].get("subject", 0),
                    -c["_match"],
                )
            )
            covered_by_slot[attribute] = candidates[0]

    if not covered_by_slot:
        return Assessment(
            verdict=NO_RELEVANT_EVIDENCE,
            coverage=0.0,
            slots=slots,
            missing=list(attributes),
            reason="лучшая карточка не покрывает вопрос",
        )

    primary_cards = list(covered_by_slot.values())

    def direct_flag(card: dict) -> bool:
        """Отвечает ли карточка на вопрос напрямую, а не «похожа по словам»."""
        cov = card.get("_coverage") or {}
        attribute = card.get("attribute")
        lexical = cov.get("lexical", 0)
        is_phrase = cov.get("phrase", 0) >= 0.45
        is_subject_exact = cov.get("subject", 0) >= 0.99
        # Общее правило безопасности может не называть конкретный узел
        # (entity 0.35), но всё равно прямо отвечает на «можно ли…».
        # Раньше такие карточки жёстко отбрасывались, и POL-001 про снятие
        # детали оказывался лишь «частичным» свидетельством.
        is_entity_ok = bool(
            not slots.entities
            or cov.get("entity", 0) >= 0.55
            or (
                attribute in GENERIC_SIBLINGS
                and cov.get("entity", 0) >= 0.30
                and (is_phrase or lexical >= 0.50)
            )
        )
        card_codes = {str(x) for x in (card.get("codes") or [])}
        is_code_exact = bool(set(slots.codes) & card_codes)
        # Точный код в вопросе может указывать на карточку любого атрибута:
        # «что означает GAP-001» отвечает карточка GAP-001, а не словарная
        # статья «определение». Раньше такие вопросы оставались PARTIAL.
        canonical = str(card.get("canonical_id") or "")
        is_code_anchor = bool(
            canonical and canonical.upper() in {code.upper() for code in slots.codes}
        )
        if not is_entity_ok:
            return False
        # 1. Прямое определение по точному коду или точному названию субъекта.
        #    Каталожная статья «F-3.3 Утечка памяти» — это ответ на «что такое
        #    утечка памяти», даже если статус источника catalog.
        if (
            attribute == "definition"
            and any(_attribute_match(a, "definition") >= 1.0 for a in attributes)
            and lexical >= 0.10
            and (is_code_exact or is_subject_exact)
        ):
            return True
        # 1b. Перефразированное определение: «какие системы входят в Купол» ↔
        #     FAQ «Из каких систем состоит Купол?». Точного совпадения основ
        #     нет, но совпадает фраза и половина субъекта. Без этого правило
        #     «при ухудшении связи» считалось более прямым ответом только
        #     потому, что в нём встретилось слово «компоненты».
        if (
            attribute == "definition"
            and is_phrase
            and lexical >= 0.70
            and cov.get("subject", 0) >= 0.40
        ):
            return True
        # 2. Точный идентификатор в вопросе: F-4.1, IL_CO2, E07, TH_*, GAP-001.
        if is_code_exact and (
            attribute in {
                "definition", "interlock_trigger", "interlock_effect",
                "interlock_clear", "threshold_value", "history_event", "gap",
            }
            or is_code_anchor
        ):
            return True
        # 3. Точная формулировка правила/процедуры: «снять деталь со стола».
        if (
            is_phrase and lexical >= 0.15
            and attribute in {
                "policy_rule", "fault_forbidden", "fault_action",
                "procedure_steps", "procedure_precondition",
            }
        ):
            return True
        # 4. Запись истории с точной датой смены — первичный источник.
        if card.get("status") == "history" and is_code_exact and lexical >= 0.15:
            return True
        # 5. Действие/процедура, привязанная к названному субъекту. Название
        #    «Потеря телеметрии узла» и вопрос «Пропала телеметрия узла»
        #    совпадают по двум основам из трёх — это тот же инцидент.
        #    FAQ-ответ («Что делать, если…») хранится как definition, но
        #    фактически является готовой процедурой.
        if (
            attribute in {"fault_action", "procedure_steps"}
            and (
                (cov.get("subject", 0) >= 0.50 and lexical >= 0.30)
                or (
                    attribute == "definition"
                    and lexical >= 0.35
                    and cov.get("subject", 0) >= 0.35
                )
                # Вопрос без названного узла, но с точной формулировкой
                # симптома: «показания датчика физически невозможны».
                or (not slots.entities and is_phrase and lexical >= 0.55)
            )
        ):
            return True
        # 6. Порог/последствие интерлока, у которых в названии есть
        #    запрошенная метрика: «влажность» → TH_CONDENSATE,
        #    «сколько раз resume» → TH_RESUME_LIMIT.
        if (
            attribute in {
                "threshold_value", "interlock_effect",
                "interlock_trigger", "interlock_clear",
            }
            and cov.get("subject", 0) >= 0.50
            and (lexical >= 0.15 or cov.get("subject", 0) >= 0.90)
        ):
            return True
        # 7. Возможность управления с точной командой или формулировкой:
        #    «список файлов» → list_files, «написать Оператору» → post_message.
        if (
            attribute == "control_capability"
            and lexical >= 0.40
            and (is_phrase or is_code_exact)
        ):
            return True
        # 8. Нормативное описание причины, привязанное к названному узлу.
        #    «Из-за чего станок на паузе?» → «Причина паузы — hold_reason».
        if (
            card.get("status") == "normative"
            and attribute in {"fault_cause", "fault_symptom"}
            and lexical >= 0.30
            and (cov.get("entity", 0) >= 0.55 or cov.get("subject", 0) >= 0.50)
        ):
            return True
        # 9. Параметр/количество: «какой параметр показывает заряд»,
        #    «сколько времени занимает подогрев». Ответом может быть
        #    описание метрики, порог или карточка возможности.
        if (
            any(a in {"quantity", "measurement_spec"} for a in attributes)
            and attribute in {
                "measurement_spec", "threshold_value",
                "equipment_spec", "control_capability",
            }
            and cov.get("subject", 0) >= 0.45
            and lexical >= 0.20
        ):
            return True
        # 10. Сравнение истории и каталога: карточка истории/правила прямо
        #     объясняет различие, даже если вопрос сформулирован как
        #     «в чём разница».
        if (
            "comparison" in attributes
            and attribute in {"history_event", "policy_rule", "definition"}
            and lexical >= 0.15
            and (cov.get("subject", 0) >= 0.50 or is_phrase)
        ):
            return True
        return False

    best = max(primary_cards, key=lambda c: c["_match"])
    best_score = best["_match"]
    floor = best_score - 0.16
    primary_ids = {c["card_id"] for c in primary_cards}
    covered_attributes = set(covered_by_slot)
    top_pool = [
        c for c in top_pool
        if c["card_id"] in primary_ids
        or (
            c["_match"] >= floor
            and (
                c["attribute"] in covered_attributes
                or _attribute_match(best["attribute"], c["attribute"]) >= 0.65
            )
        )
        # Нормативная карточка того же атрибута может быть ниже по общему
        # баллу (у неё другая формулировка), но именно она содержит ответ:
        # «Удержание фрезера» ниже каталожной статьи об экстренной паузе.
        or (
            c.get("status") == "normative"
            and c["attribute"] in covered_attributes
            and c["_match"] >= 0.55
        )
    ]
    direct_ids = {c["card_id"] for c in top_pool if direct_flag(c)}
    top_pool.sort(
        key=lambda c: (
            0 if c["card_id"] in direct_ids else 1,
            # Релевантность важнее формального главного слота: вопрос может
            # распознаться с лишним вторичным атрибутом. Главный слот решает
            # только при близких баллах.
            -c["_match"],
            0 if c["attribute"] == slots.primary else 1,
            -int(c.get("authority") or 3),
            c["card_id"],
        )
    )
    best = top_pool[0]
    best_score = best["_match"]
    best_cov = best.get("_coverage", {})

    # Поддержка — это независимые канонические факты, а не любые карточки
    # того же семейства. Иначе один порог «подтверждал» себя тремя копиями.
    support_keys = {
        c.get("canonical_id") or c["card_id"]
        for c in top_pool
        if c["_match"] >= max(0.55, best_score - 0.12)
        and (
            c["card_id"] in primary_ids
            or c.get("canonical_id") == best.get("canonical_id")
            or taxonomy.family(c["attribute"]) == taxonomy.family(best["attribute"])
        )
    }
    support = max(1, len(support_keys))

    missing = [attribute for attribute in attributes if attribute not in covered_by_slot]

    conflicts = _conflicts(top_pool[:12])
    if conflicts:
        # Конфликт объявляется только тогда, когда расхождение не перекрыто
        # более сильным нормативным ответом. Иначе любой связанный спорный
        # документ превращал бы прямой ответ в «конфликт».
        normative = [
            c for c in top_pool
            if c.get("origin") == "structured"
            and c.get("status") == "normative"
            and c.get("attribute") != "gap"
            and c["_match"] >= best_score - 0.03
        ]
        if normative:
            conflicts = []
    if conflicts:
        return Assessment(
            verdict=CONFLICT,
            coverage=best_score,
            slots=slots,
            cards=top_pool[:8],
            missing=missing,
            conflicts=conflicts,
            support=support,
            reason="источники расходятся; нужен Оператор или живая телеметрия",
        )

    direct = direct_flag(best)
    # 8. Сравнение: обе стороны должны быть представлены.
    if "comparison" in attributes:
        side_codes = {
            str(code)
            for code in slots.codes
            if any(
                str(code) in {str(x) for x in (c.get("codes") or [])}
                for c in top_pool
            )
        }
        side_entities = {
            entity
            for entity in slots.entities
            if any(entity in (c.get("entities") or []) for c in top_pool)
        }
        if len(side_codes) >= 2 or len(side_entities) >= 2:
            direct = True

    strong = bool(
        best_score >= 0.72
        and (support >= 2 or int(best.get("authority") or 3) >= 4)
    )
    if not strong and (not missing or direct):
        # Вторичный слот может остаться незакрытым, но прямой ответ уже есть:
        # «какие интерлоки срабатывали в смену 2026-10-16» отвечает запись
        # смены, даже если отдельной карточки interlock_trigger нет.
        # Незакрытый главный слот отсекается ниже.
        strong = direct
    if getattr(slots, "generic", False) and not direct:
        # Вопрос без распознанного атрибута не может считаться отвеченным
        # уверенно: раздел корпуса — это не ответ на конкретный вопрос.
        # Но если карточка дословно повторяет вопрос (success:true …),
        # это уже не «раздел», а прямой ответ.
        strong = False
    if [a for a in missing if a == slots.primary]:
        strong = False
    # Слабая карточка (мнение/гипотеза/неполная) не даёт уверенного ответа,
    # даже если формально совпала по формулировке.
    if best.get("status") in WEAK_STATUS and not (
        allow_weak_history and best.get("attribute") == "history_event"
    ):
        strong = False
    # Вторичный слот может остаться незакрытым, если прямой ответ уже есть:
    # «показания датчика невозможны» отвечает каталог отказов, а
    # сопутствующий live_state не должен превращать ответ в PARTIAL.
    # Незакрытый главный слот отсекается выше.
    if missing and not direct:
        strong = False
    if strong:
        reason = (
            "прямое определение по точному субъекту/коду"
            if direct and not (support >= 2 or int(best.get("authority") or 3) >= 4)
            else f"привязка субъект+атрибут подтверждена ({support} карточек)"
        )
        return Assessment(
            verdict=ANSWERABLE,
            coverage=best_score,
            slots=slots,
            cards=top_pool[:8],
            missing=missing,
            support=support,
            reason=reason,
        )
    if best_score >= 0.45:
        return Assessment(
            verdict=PARTIAL,
            coverage=best_score,
            slots=slots,
            cards=top_pool[:8],
            missing=missing,
            support=support,
            reason=(
                "есть частичная привязка; часть слотов вопроса не закрыта"
                if missing
                else "свидетельство привязано к вопросу, но не подтверждено достаточно надёжно"
            ),
        )
    return Assessment(
        verdict=NO_RELEVANT_EVIDENCE,
        coverage=best_score,
        slots=slots,
        missing=list(attributes),
        reason="лучшая карточка не покрывает вопрос",
    )


def confidence_for(assessment: Assessment) -> dict:
    """Калибровка уверенности по вердикту.

    Это ключевое отличие от старой формулы ``0.15 + 0.45*top_score``: она
    могла дать 0.86 ответу, который не отвечает на вопрос. Здесь потолок
    определяется вердиктом.
    """
    coverage = max(0.0, min(1.0, assessment.coverage))
    verdict = assessment.verdict
    if verdict == ANSWERABLE:
        score = 0.50 + 0.40 * coverage + 0.04 * max(0, assessment.support - 1)
        if assessment.missing:
            score -= 0.05
        score = min(0.92, score)
    elif verdict == PARTIAL:
        score = min(0.55, 0.15 + 0.50 * coverage)
    elif verdict == CONFLICT:
        score = min(0.40, 0.22 + 0.20 * coverage)
    else:
        score = 0.0
    score = round(score, 2)
    level = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
    return {
        "score": score,
        "level": level,
        "verdict": verdict,
        "coverage": round(coverage, 3),
        "support": assessment.support,
        "reason": assessment.reason,
    }


def verdict_is_abstention(verdict: str) -> bool:
    return verdict == NO_RELEVANT_EVIDENCE


def verdict_is_cautious(verdict: str) -> bool:
    return verdict in {PARTIAL, CONFLICT, NO_RELEVANT_EVIDENCE}
