"""Сборка ответа: карточки → проверка покрытия → калиброванная уверенность.

Ответ всегда собирается в конверт:

``verdict`` / ``facts`` / ``hypotheses`` / ``unknown`` / ``actions`` /
``citations`` / ``confidence`` / ``cards``

Главное отличие от v1: уверенность больше не выводится из «похожести
фрагмента». Сначала проверяется, что найденная карточка привязана к субъекту и
атрибуту вопроса (``coverage``), и только потом считается число. Поэтому
частичный ответ не может выглядеть как уверенный, а вопрос вне корпуса
получает ровно 0.0 и явный отказ.

Режим по умолчанию детерминированный. ``--llm`` добавляет пересказ через
DeepSeek, но факты, цитаты, вердикт и уверенность всё равно считаются здесь.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from . import coverage, taxonomy
from .retrieval import KB

# «Показания датчика» и «что показывает параметр» — это вопрос о смысле
# величины, а не запрос живого значения. Раньше такие вопросы помечались как
# LIVE, и в факты добавлялась строка про снимок телеметрии, хотя человек
# спрашивал совсем о другом. Живой запрос — это явное «сейчас/текущее».
LIVE_MARKERS = (
    "сейчас", "текущ", "в реальном времени", "прямо сейчас", "онлайн",
    "в данный момент", "live snapshot", "снимок телеметри",
)
DIAG_MARKERS = (
    "не работает", "не включа", "не запуска", "ошибк", "отказ", "почему",
    "сработал", "упал", "пропал", "нет связи", "авари", "пауза", "hold",
    "alarm", "диагност", "причин", "дефект", "брак", "перегрев", "утечк",
)
HISTORY_MARKERS = (
    "было", "произошл", "в прошл", "истори", "журнал", "logbook", "раньше",
    "смена", "вчера", "сегодн", "хроник",
)
POLICY_MARKERS = (
    "можно ли", "разрешено", "разрешён", "запрещ", "правил", "политик",
    "обязан", "допустимо", "безопасно ли", "имеет ли право", "нужно ли",
)


def classify(question: str) -> str:
    q = question.lower()
    if any(m in q for m in LIVE_MARKERS):
        return "LIVE"
    if any(m in q for m in DIAG_MARKERS):
        return "DIAGNOSTIC"
    if any(m in q for m in LIVE_MARKERS):
        return "LIVE"
    if any(m in q for m in HISTORY_MARKERS):
        return "HISTORY"
    if any(m in q for m in POLICY_MARKERS):
        return "POLICY"
    return "REFERENCE"


def _fmt_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return "…".join(_fmt_value(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt_value(v)}" for k, v in value.items())
    if isinstance(value, bool):
        return "да" if value else "нет"
    return str(value)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clip(text: str, limit: int) -> str:
    text = _clean(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _split_steps(value: str) -> list[str]:
    parts = re.split(r";\s*(?=\d+\))", value)
    if len(parts) <= 1:
        parts = re.split(r";\s*", value)
    return [_clean(part) for part in parts if _clean(part)][:12]


ACTION_ATTRIBUTES = {"procedure_steps", "fault_action"}
HYPOTHESIS_STATUS = {"catalog", "theory", "hypothesis", "opinion"}
UNSAFE_RESET = re.compile(
    r"(?:блокир\w*\s+тревог|отмен\w*\s+отключен|"
    r"(?:отключ|сброс|обход|снят)\w*[^.;]{0,35}(?:защит|блокиров|интерлок)|"
    r"block_interlock|block_automation|set_thresholds)", re.I,
)


def _incident_decision(question: str) -> bool:
    """A supplied observation cannot prove a false alarm or grant a reset."""
    q = question.casefold()
    observation = bool(re.search(r"\d+\s*(?:ppm|ппм|%)|датчик\w*\s+показыва", q))
    decision = bool(re.search(r"ложн|фейк|сброс|снят|отключ.*защит|достаточно|недостовер", q))
    return observation and decision
MAX_FACTS = 6
MAX_HYPOTHESES = 4
MAX_ACTIONS = 8


def _order_cards(cards: list[dict], slots) -> list[dict]:
    """Прямой ответ — первым, затем главный слот, затем остальные.

    Порядок внутри ``assessment.cards`` уже ставит карточку с прямым
    доказательством первой. Раньше ``_order_cards`` сортировал только по
    атрибуту, и карточка «Залипание на последнем значении» (тот же атрибут
    ``fault_action``) обгоняла точную процедуру «Потеря телеметрии узла».
    """
    wanted = set(slots.attributes or [])
    primary = slots.primary
    first_id = cards[0].get("card_id") if cards else None

    def key(card: dict) -> tuple:
        attribute = card.get("attribute")
        rank = (
            0 if card.get("card_id") == first_id
            else 1 if attribute == primary
            else 2 if attribute in wanted
            else 3
        )
        return (rank, -float(card.get("score") or 0.0), card.get("card_id") or "")

    return sorted(cards, key=key)


def _keep_card(card: dict, best: dict) -> bool:
    """Оставить карточку, которая действительно относится к вопросу.

    ``assessment.cards`` — это пул покрытия, а не готовый список фактов: в нём
    могут оказаться соседние карточки того же семейства (например, все
    параметры датчика на вопрос «что делать при потере телеметрии»). Фильтр
    опирается на оси покрытия, а не на порядок.
    """
    if card is best:
        return True
    if card.get("canonical_id") and card.get("canonical_id") == best.get("canonical_id"):
        return True
    # Тот же порог/интерлок/код: карточка-последствие и карточка-порог
    # описывают один и тот же факт с разных сторон.
    if set(card.get("codes") or []) & set(best.get("codes") or []):
        return True
    cov = card.get("_coverage") or {}
    subject = float(cov.get("subject") or 0)
    entity = float(cov.get("entity") or 0)
    code = float(cov.get("code") or 0)
    phrase = float(cov.get("phrase") or 0)
    lexical = float(cov.get("lexical") or 0)
    # Два разных определения не заменяют друг друга. Вопрос «какие системы
    # входят в Купол» — это FAQ о составе, а не «какие зоны доступа есть в
    # Куполе». Раньше соседний FAQ попадал в факты только потому, что в его
    # тексте тоже встречалось слово «системы».
    if card.get("attribute") == "definition" and best.get("attribute") == "definition":
        if subject < 0.45:
            return False
    # Определение — самостоятельный ответ. Правило или процедура того же
    # текста уместны только при совпадении субъекта, иначе «какие системы
    # входят в Купол» подкреплялось правилом «при ухудшении связи» лишь
    # потому, что там встретилось слово «компоненты».
    if (
        best.get("attribute") == "definition"
        and card.get("attribute") in {"policy_rule", "fault_forbidden", "procedure_steps"}
        and subject < 0.45
    ):
        return False
    return bool(
        code >= 0.99
        or phrase >= 0.35
        or (subject >= 0.75 and lexical >= 0.10)
        or entity >= 0.85
        or (entity >= 0.55 and lexical >= 0.30)
    )



def _status_note(card: dict) -> str:
    status = card.get("status")
    if status == "catalog":
        return "каталог отказов описывает возможный отказ, а не доказанный факт"
    if status == "theory":
        return "теоретический/справочный материал"
    if status == "hypothesis":
        return "гипотеза, требующая проверки"
    if status == "opinion":
        return "мнение из журнала, не подтверждённый факт"
    if status == "history":
        return "запись истории"
    if status == "incomplete":
        return "источник неполный (например, DOCX без текстового слоя)"
    if status == "conflict":
        return "источники противоречат друг другу"
    return ""


def _card_citation(card: dict) -> dict:
    source = card.get("source") or {}
    return {
        "kind": "card",
        "card_id": card["card_id"],
        "source": source.get("doc") or "—",
        "title": source.get("title") or card.get("subject_label"),
        "line": source.get("line"),
        "canonical_id": card.get("canonical_id"),
        "attribute": card.get("attribute"),
        "status": card.get("status"),
        "score": card.get("score"),
        "text": _clip(card.get("value"), 300),
    }


def _build_text(
    question: str,
    result: dict,
    confidence: dict,
    assessment,
) -> str:
    verdict = result["verdict"]
    lines = [f"Вопрос: {question}", f"Вердикт: {verdict}. "
             f"Уверенность: {confidence['level']} ({confidence['score']:.2f})."]
    if assessment.slots.attributes:
        lines.append(
            "Разбор вопроса: объект — "
            + (", ".join(assessment.slots.entities + assessment.slots.codes) or "не указан конкретный")
            + "; атрибут — "
            + ", ".join(taxonomy.label(a) for a in assessment.slots.attributes)
        )
    lines.append("")

    if verdict == coverage.NO_RELEVANT_EVIDENCE:
        lines.append("В корпусе нет достаточных сведений, чтобы ответить на этот вопрос.")
        if assessment.slots.off_domain:
            lines.append(
                "Запрошенный тип данных ("
                + ", ".join(taxonomy.label(a) for a in assessment.slots.attributes)
                + ") в корпусе не описан."
            )
        if result["unknown"]:
            lines.append("")
            lines.append("Чего не хватает:")
            for item in result["unknown"]:
                lines.append(f"- {item}")
        lines.append("")
        lines.append("Что можно сделать:")
        for i, action in enumerate(result["actions"] or [
            "Уточнить вопрос или назвать узел/идентификатор (`IL_CO2`, `F-1.7`, `E07`).",
            "Запросить текущее состояние через живой API (`py kb.py live snapshot`).",
            "Спросить Оператора через `meeting_01`.",
        ], 1):
            lines.append(f"{i}. {action}")
        return "\n".join(lines).strip()

    if result.get("reference_facts"):
        lines.append("Справочные сведения (не текущие измерения):")
        lines.extend(f"- {item}" for item in result["reference_facts"])
        lines.append("")
    if result["facts"]:
        lines.append("Подтверждённые сведения:")
        for item in result["facts"]:
            lines.append(f"- {item}")
        lines.append("")
    if result["hypotheses"]:
        lines.append("Гипотезы и каталожные описания (не доказательство факта):")
        for item in result["hypotheses"]:
            lines.append(f"- {item}")
        lines.append("")
    if result["unknown"]:
        lines.append("Ограничения и неизвестное:")
        for item in result["unknown"]:
            lines.append(f"- {item}")
        lines.append("")
    if result["actions"]:
        lines.append("Что делать:")
        for i, item in enumerate(result["actions"], 1):
            lines.append(f"{i}. {item}")
        lines.append("")
    if result["citations"]:
        lines.append("Источники:")
        for citation in result["citations"][:8]:
            suffix = f" — {citation['title']}" if citation.get("title") else ""
            attr = f" [{citation.get('attribute')}]" if citation.get("attribute") else ""
            lines.append(f"- {citation['source']}{suffix}{attr}")
        lines.append("")
    lines.append("Проверка покрытия: " + assessment.reason + ".")
    return "\n".join(lines).strip()


def _short_answer(result: dict) -> str:
    """Одна фраза для быстрого показа агенту и человеку."""
    if result["verdict"] == coverage.NO_RELEVANT_EVIDENCE:
        return "В корпусе нет достаточных сведений, чтобы ответить на этот вопрос."
    if result["actions"]:
        return _clip("; ".join(result["actions"][:2]), 360)
    if result["facts"]:
        return _clip(result["facts"][0], 360)
    if result["hypotheses"]:
        return _clip(result["hypotheses"][0], 360)
    return ""


def _call_deepseek(question: str, context: str, facts: list[str], verdict: str) -> str | None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None
    model = os.environ.get("DOMEKB_LLM_MODEL", "deepseek-chat")
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    system = (
        "Ты — ассистент базы знаний купола выживальщика. Отвечай по-русски, кратко и по делу. "
        "Используй ТОЛЬКО предоставленный контекст. Если данных нет — прямо скажи, чего не хватает. "
        "Различай каталог отказов (описание возможного) и подтверждённый факт (запись истории/журнала). "
        "Не выдумывай значения телеметрии: живые данные в контекст не входят. "
        "Физические действия выполняет только Оператор. success:true означает приём команды, а не её выполнение. "
        f"Вердикт проверки покрытия: {verdict}. Не повышай его: если PARTIAL или CONFLICT — скажи об этом."
    )
    user = (
        f"Вопрос: {question}\n\n"
        f"Структурные факты:\n" + "\n".join(f"- {f}" for f in facts[:40]) +
        f"\n\nКонтекст корпуса:\n{context[:12000]}"
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2,
            "max_tokens": 900,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, IndexError, TimeoutError, json.JSONDecodeError):
        return None


def answer(question: str, kb: KB, use_llm: bool = False, k: int = 8) -> dict:
    question = (question or "").strip()
    intent = classify(question)
    entities = kb.detect_entities(question)
    slots = coverage.extract_slots(question, entities=entities)
    # Пул кандидатов не должен зависеть от того, сколько карточек просили
    # показать: иначе один и тот же вопрос получает ANSWERABLE при k=8 и
    # PARTIAL при k=5. Верхняя граница вывода (k) — это оформление, а не
    # доказательная база.
    retrieved = kb.search_cards(question, k=max(24, k * 2), slots=slots)
    assessment = coverage.assess(
        question, retrieved, slots=slots, entities=entities, all_cards=kb.cards
    )
    confidence = coverage.confidence_for(assessment)

    cards = assessment.cards[:k] if assessment.verdict != coverage.NO_RELEVANT_EVIDENCE else []
    cards = _order_cards(cards, assessment.slots)
    if cards:
        cards = [c for c in cards if _keep_card(c, cards[0])] or cards
    # Абзац без привязки к атрибуту — контекст, а не ответ. Показываем его
    # только если ничего лучше не нашлось.
    answer_cards = [
        c for c in cards
        if c.get("attribute") != "section_text"
        or (cards and c.get("card_id") == cards[0].get("card_id"))
    ] or cards

    facts: list[str] = []
    hypotheses: list[str] = []
    unknown: list[str] = []
    actions: list[str] = []
    citations: list[dict] = []
    seen: set[str] = set()
    # Вопрос-сравнение («чем отличается A от B») просит сами описания A и B,
    # а не гипотезу о них. Поэтому каталожные карточки обеих сторон попадают
    # в факты (с честной пометкой о статусе), иначе одна сторона оказывалась
    # в фактах, а вторая — только в гипотезах.
    comparison = "comparison" in (assessment.slots.attributes or [])
    action_requested = assessment.slots.primary in ACTION_ATTRIBUTES or bool(
        answer_cards and answer_cards[0].get("attribute") in ACTION_ATTRIBUTES
    )
    # У одной стороны сравнения может быть и заголовок, и подробное описание.
    # Подробное описание полезнее: заголовок «F-1.2 — Ложное занижение» лишь
    # повторяет название, если рядом уже есть симптом/причина того же отказа.
    rich_sides = {
        str(card.get("canonical_id"))
        for card in answer_cards
        if card.get("attribute") in {
            "fault_symptom", "fault_cause", "threshold_value",
            "interlock_trigger", "interlock_effect",
        }
    }

    def add(bucket: list[str], text: str) -> bool:
        text = _clean(text)
        if text and text not in seen:
            seen.add(text)
            bucket.append(text)
            return True
        return False

    for card in answer_cards:
        value = _clean(card.get("value"))
        if not value:
            continue
        status = card.get("status")
        attribute = card.get("attribute")
        if (
            comparison
            and attribute == "definition"
            and str(card.get("canonical_id") or "") in rich_sides
        ):
            continue
        label = card.get("subject_label") or card.get("subject") or ""
        prefix = f"{label}: " if label else ""
        note = _status_note(card)
        line = prefix + value
        used = False

        if attribute in ACTION_ATTRIBUTES:
            if not action_requested:
                continue
            # Catalog entries describe possible incidents. They are not an
            # authorized response to the incident currently being discussed.
            if status in {"normative", "reference"} and len(actions) < MAX_ACTIONS:
                for step in _split_steps(value):
                    if len(actions) >= MAX_ACTIONS:
                        break
                    if UNSAFE_RESET.search(step):
                        used = add(unknown, "Источник упоминает изменение защит. Это не разрешение на действие; нужна проверка политики и условий.") or used
                    else:
                        used = add(actions, step) or used
            if status in HYPOTHESIS_STATUS and len(hypotheses) < MAX_HYPOTHESES:
                used = add(hypotheses, line + (f" ({note})" if note else "")) or used
        elif attribute in {"policy_rule", "fault_forbidden", "procedure_precondition"}:
            if status in HYPOTHESIS_STATUS and len(hypotheses) < MAX_HYPOTHESES:
                used = add(hypotheses, line + (f" ({note})" if note else ""))
            elif len(facts) < MAX_FACTS:
                used = add(facts, line)
        elif status in HYPOTHESIS_STATUS and not (
            comparison
            and attribute in {"definition", "fault_symptom", "fault_cause"}
        ) and len(hypotheses) < MAX_HYPOTHESES:
            used = add(hypotheses, line + (f" ({note})" if note else ""))
        elif status in {"incomplete", "conflict"}:
            used = add(unknown, line + (f" ({note})" if note else ""))
        elif len(facts) < MAX_FACTS:
            used = add(facts, line + (f" ({note})" if note else ""))

        # Цитируем только те карточки, чьё утверждение реально попало в ответ.
        if used:
            citations.append(_card_citation(card))

    # Ограничения, которые важны для правильного чтения ответа.
    if assessment.missing:
        unknown.append(
            "не найден прямой ответ по атрибуту: "
            + ", ".join(taxonomy.label(a) for a in assessment.missing)
        )
    if assessment.fallback:
        unknown.append(
            "прямого описания запрошенного эффекта нет; показаны связанные правила и условия."
        )
    for conflict in assessment.conflicts:
        unknown.append(
            f"конфликт {conflict['id']}: {_clip(conflict.get('description') or '', 240)}"
        )
    if any(c.get("has_gaps") or c.get("status") == "incomplete" for c in cards):
        unknown.append(
            "часть источников неполная (пометки [ЗАПОЛНИТЬ] или DOCX без OCR) — "
            "нельзя считать ответ исчерпывающим."
        )
    if assessment.verdict == coverage.CONFLICT:
        actions.append("Сверить конфликтующие источники с Оператором и живой телеметрией.")
    if assessment.verdict == coverage.PARTIAL and not actions:
        if assessment.missing:
            actions.append("Уточнить вопрос или проверить недостающие данные у Оператора.")
        else:
            actions.append(
                "Сверить запись с журналом или живой телеметрией: "
                "второго подтверждающего источника нет."
            )

    if assessment.verdict == coverage.NO_RELEVANT_EVIDENCE:
        facts, hypotheses, actions, citations = [], [], [], []
        if assessment.slots.off_domain:
            unknown.append(
                "в корпусе нет сведений по теме: "
                + ", ".join(taxonomy.label(a) for a in assessment.slots.attributes)
            )
        else:
            unknown.append(assessment.reason + ".")
    elif not cards:
        unknown.append("карточки не найдены")

    reference_facts = []
    requires_verification = _incident_decision(question)
    if intent == "LIVE":
        reference_facts, facts = facts, []
        unknown.insert(0, "Текущее значение не установлено: в RAG нет свежего снимка телеметрии. Справочные пороги не являются показаниями датчика.")
        actions = ["Получить свежую телеметрию через API и проверить время, узел и качество измерения."]
        if assessment.verdict != coverage.NO_RELEVANT_EVIDENCE:
            assessment.verdict = coverage.PARTIAL
        assessment.reason = "Для ответа нужно текущее измерение; справочная база его не содержит"
        assessment.missing = list(dict.fromkeys(assessment.missing + ["live_telemetry"]))
        confidence = {**confidence, "score": 0.0, "level": "low"}
    if requires_verification:
        unknown.insert(0, "Расхождение показаний само по себе не доказывает ложную тревогу. Нужны свежесть, зона и независимые подтверждения; сброс защит не разрешён этим ответом.")
        actions = ["Сопоставить время, зону и независимые датчики; сохранить защитное состояние до выяснения причины."]
        if assessment.verdict == coverage.ANSWERABLE:
            assessment.verdict = coverage.PARTIAL
        assessment.reason = "Каталог отказов не подтверждает причину текущего инцидента"
        confidence = {**confidence, "score": min(float(confidence.get("score", 0)), 0.35), "level": "low"}

    confidence.update(verdict=assessment.verdict, reason=assessment.reason)
    result = {
        "question": question,
        "intent": intent,
        "verdict": assessment.verdict,
        "abstained": assessment.verdict == coverage.NO_RELEVANT_EVIDENCE,
        "requires_live": intent == "LIVE",
        "requires_verification": requires_verification,
        "reference_facts": reference_facts,
        "facts": list(dict.fromkeys(facts)),
        "hypotheses": list(dict.fromkeys(hypotheses)),
        "unknown": list(dict.fromkeys(unknown)),
        "actions": list(dict.fromkeys(actions)),
        "citations": citations,
        "confidence": confidence,
        "coverage": assessment.as_dict(),
        "cards": cards,
        "chunks": [
            {
                "chunk_id": card.get("chunk_id"),
                "doc": (card.get("source") or {}).get("doc"),
                "title": (card.get("source") or {}).get("title"),
                "text": card.get("text"),
                "score": card.get("score"),
                "canonical_id": card.get("canonical_id"),
                "verification_status": card.get("status"),
                "has_gaps": card.get("has_gaps"),
            }
            for card in cards
        ],
        "mode": "deterministic",
    }
    result["short_answer"] = _short_answer(result)
    result["answer"] = _build_text(question, result, confidence, assessment)

    if use_llm and not (result["requires_live"] or requires_verification or result["abstained"]):
        context = "\n\n".join(
            f"[{card['source'].get('doc')} :: {card['source'].get('title')}]\n{card.get('text')}"
            for card in cards[:8]
        )
        text = _call_deepseek(question, context, result["facts"], assessment.verdict)
        if text:
            result["answer"] = text
            result["mode"] = "llm+deterministic"
        else:
            result["mode"] = "deterministic (LLM недоступна)"
            result["unknown"].append(
                "LLM-режим запрошен, но ключ/сеть недоступны — показан детерминированный ответ."
            )

    return result


def render(result: dict) -> str:
    return result["answer"]
