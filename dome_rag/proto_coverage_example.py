"""Прототип «проверки покрытия» (coverage gate) поверх текущего answer().

Идея: наличие похожего текста != наличие ответа.
Ответ считается допустимым, только если найденный материал закрывает
«слот» вопроса (значение, процедура, политика, история, определение...).
Иначе возвращается честный отказ.
"""
from __future__ import annotations

import re
import sys

from builder.retrieval import KB
from builder.answer import answer, classify
from builder import vectors as vec

VALUE_RE = re.compile(r"(при как|как(ая|ой|ое|ие|ов)|сколько|чему|какой уровень|до скольк|порог)", re.I)
PROC_RE = re.compile(r"(как |что делать|что нужно|порядок|инструкц|шаг|запуст|выполн|проверить|починить)", re.I)
POLICY_RE = re.compile(r"(можно ли|разреш|запрещ|допустим|обязан|имеет ли право|нужно ли|безопасно ли)", re.I)
HIST_RE = re.compile(r"(что было|что происход|в прошл|истори|журнал|смена|вчера|сегодня|записал|раньше)", re.I)
DEF_RE = re.compile(r"(что такое|что означает|что значит|расшифруй|определение|чем отличается)", re.I)
DIAG_RE = re.compile(r"(почему|не работает|не включа|не запуска|встал|останов|ошибк|отказ|авари|пауза|hold|причин)", re.I)

NUMBER_RE = re.compile(r"\d")


def frame(question: str) -> str:
    q = question.lower()
    if DIAG_RE.search(q):
        return "DIAGNOSTIC"
    if DEF_RE.search(q):
        return "DEFINITION"
    if POLICY_RE.search(q):
        return "POLICY"
    if VALUE_RE.search(q):
        return "VALUE"
    if PROC_RE.search(q):
        return "PROCEDURE"
    if HIST_RE.search(q):
        return "HISTORY"
    return "REFERENCE"


def _has_number(text: str) -> bool:
    return bool(NUMBER_RE.search(text or ""))


def coverage(question: str, result: dict, kb: KB) -> tuple[bool, str]:
    """True, если найденное действительно закрывает запрошенный слот."""
    f = frame(question)
    facts = result.get("facts") or []
    actions = result.get("actions") or []
    chunks = result.get("chunks") or []
    if result.get("abstained"):
        return True, "already abstained"

    if f == "VALUE":
        # нужен числовой факт с единицей/порогом
        for x in facts:
            if _has_number(x) and re.search(r"(>|<|≥|≤|=|порог|ppm|°C|%|В|А|Вт|ч|с\b)", x):
                return True, "numeric fact"
        # либо числовой ответ прямо в верхнем чанке и он про сущность вопроса
        if chunks and _has_number(chunks[0]["text"]) and result.get("confidence", {}).get("score", 0) >= 0.7:
            return True, "numeric chunk"
        return False, "нет числового факта"

    if f == "PROCEDURE":
        if actions:
            return True, "есть шаги"
        if any(c["type"] in ("procedure", "diagnostic", "safety", "limits") for c in chunks):
            return True, "процедурный чанк"
        return False, "нет процедуры"

    if f == "POLICY":
        for x in facts:
            if re.search(r"(запрещ|разреш|нельзя|обязан|только Оператор|политик|POL-)", x):
                return True, "политика"
        if any(c["type"] in ("rule", "safety", "agent_spec") for c in chunks):
            return True, "нормативный чанк"
        return False, "нет политики"

    if f == "HISTORY":
        if any(c["collection"] in ("history", "episodes") for c in chunks):
            return True, "история"
        return False, "нет истории"

    if f == "DEFINITION":
        if facts and any("«" in x and "»" in x for x in facts):
            return True, "глоссарий"
        if chunks and chunks[0]["type"] in ("glossary", "faq", "reference", "persona"):
            return True, "справочный чанк"
        return False, "нет определения"

    if f == "DIAGNOSTIC":
        if any(c["type"] in ("fault", "diagnostic", "history_fault", "system") for c in chunks):
            return True, "диагностический материал"
        return False, "нет диагностики"

    # REFERENCE / прочее: требуем содержательного пересечения запроса и топа
    q_stems = {t[:5] for t in vec.tokenize(question) if len(t) >= 3}
    if not chunks:
        return False, "нет чанков"
    body = " ".join(vec.tokenize(chunks[0]["text"]))
    hit = sum(1 for s in q_stems if s in body)
    if q_stems and hit / len(q_stems) >= 0.4:
        return True, f"покрытие {hit}/{len(q_stems)}"
    return False, "слабое покрытие запроса"


if __name__ == "__main__":
    kb = KB()
    qs = sys.argv[1:] or ["какая погода на улице"]
    for q in qs:
        r = answer(q, kb)
        ok, why = coverage(q, r, kb)
        print(f"{q}\n  frame={frame(q)} conf={r['confidence']['score']} covered={ok} ({why})")
    kb.close()
