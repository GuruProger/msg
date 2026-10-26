"""Сборка ответа: маршрутизация вопроса, структурные факты, цитаты, уверенность.

Ответ всегда собирается в конверт:

``facts`` / ``hypotheses`` / ``unknown`` / ``actions`` / ``citations`` / ``confidence``

Это важнее «красивого текста»: агент должен видеть, что подтверждено, что
является гипотезой, чего в базе нет и на что он опирается. Текст — лишь
представление конверта.

По умолчанию режим детерминированный. ``--llm`` добавляет пересказ через
DeepSeek, но факты, цитаты и уверенность всё равно считаются здесь.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from . import retrieval as ret
from .retrieval import KB, _jload

LIVE_MARKERS = (
    "сейчас", "текущ", "в реальном времени", "прямо сейчас", "онлайн",
    "показыва", "показани", "статус", "состояни",
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

NO_DATA_MARKERS = (
    "пароль", "серийный номер", "серийник", "wi-fi", "wifi", "телефон",
    "почтовый адрес", "домашний адрес", "стоимость", "цена", "сколько стоит",
    "сколько человек", "состав экипажа", "прогноз на завтра", "погода на марсе",
    "госномер", "инвентарный номер",
)


def classify(question: str) -> str:
    q = question.lower()
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


def _node_facts(node: dict) -> tuple[list[str], list[str]]:
    facts, unknown = [], []
    facts.append(f"{node['title']} (`{node['node_id']}`), система: {node.get('system_ru') or node.get('system')}")
    measurements = [p for p in node["params"] if p["kind"] in ("measurement", "computed")]
    states = [p for p in node["params"] if p["kind"] == "state"]
    for p in measurements[:6]:
        unit = f" {p['unit']}" if p.get("unit") else ""
        facts.append(f"параметр `{p['key']}`{unit}")
    if states:
        facts.append("состояния: " + ", ".join(f"`{p['key']}`" for p in states[:8]))
    if node["controls"]:
        parts = []
        for c in node["controls"]:
            who = "только Оператор" if c.get("allowed_for") == "operator" else "агент"
            confirmed = "" if c.get("api_confirmed") else " — не подтверждено в API"
            parts.append(f"`{c['action']}` ({c['label']}; {who}{confirmed})")
        facts.append("команды: " + "; ".join(parts[:8]))
    else:
        unknown.append(f"у узла `{node['node_id']}` нет управляющих команд в API — только чтение.")
    return facts, unknown


def _gap_facts(gaps: list[dict]) -> tuple[list[str], list[str]]:
    facts, unknown = [], []
    for g in gaps:
        unknown.append(f"{g['id']}: {g['description']}")
        if g.get("conflict"):
            facts.append(f"Конфликт {g['id']}: {g['conflict']}")
        if g.get("resolution"):
            facts.append(f"Как закрыть {g['id']}: {g['resolution']}")
    return facts, unknown


def _build_text(question: str, intent: str, facts: list[str], hypotheses: list[str],
                unknown: list[str], actions: list[str], citations: list[dict],
                chunks: list[dict], confidence: dict) -> str:
    lines = []
    if not chunks and not facts:
        lines.append("В базе знаний нет достаточных данных, чтобы ответить на этот вопрос.")
        if unknown:
            lines.append("")
            lines.append("Чего не хватает:")
            for u in unknown:
                lines.append(f"- {u}")
        lines.append("")
        lines.append("Что можно сделать:")
        lines.append("1. Уточнить вопрос или назвать узел/идентификатор (`IL_CO2`, `F-1.7`, `E07`).")
        lines.append("2. Запросить текущее состояние через живой API (`py kb.py live snapshot`).")
        lines.append("3. Спросить Оператора через `meeting_01`.")
        return "\n".join(lines)

    lines.append(f"Вопрос: {question}")
    lines.append(f"Тип: {intent}. Уверенность: {confidence['level']} ({confidence['score']:.2f}).")
    lines.append("")

    if facts:
        lines.append("Подтверждённые сведения:")
        for f in facts:
            lines.append(f"- {f}")
        lines.append("")
    if hypotheses:
        lines.append("Гипотезы (не подтверждено):")
        for h in hypotheses:
            lines.append(f"- {h}")
        lines.append("")
    if unknown:
        lines.append("Чего не хватает / что неизвестно:")
        for u in unknown:
            lines.append(f"- {u}")
        lines.append("")
    if actions:
        lines.append("Что делать:")
        for i, a in enumerate(actions, 1):
            lines.append(f"{i}. {a}")
        lines.append("")
    if chunks:
        lines.append("Ближайшие фрагменты корпуса:")
        for c in chunks[:4]:
            canon = f" [{c['canonical_id']}]" if c.get("canonical_id") else ""
            lines.append(f"- {c['doc']} → {c['title']}{canon} (score {c['score']:.2f})")
        lines.append("")
    if citations:
        lines.append("Источники:")
        for c in citations[:6]:
            lines.append(f"- {c['source']}" + (f" — {c['title']}" if c.get("title") else ""))
    return "\n".join(lines).strip()


def _call_deepseek(question: str, context: str, facts: list[str]) -> str | None:
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
        "Физические действия выполняет только Оператор. success:true означает приём команды, а не её выполнение."
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
    no_data = [m for m in NO_DATA_MARKERS if m in question.lower()]
    qlow = question.lower()
    line_focus = any(w in qlow for w in ("линия", "линии", "щит", "smart_panel"))
    chunks = kb.search(question, k=k)

    facts: list[str] = []
    hypotheses: list[str] = []
    unknown: list[str] = []
    actions: list[str] = []
    citations: list[dict] = []
    structured_hits = 0

    # ---------------------------------------------------------------- чанки
    for c in chunks[:5]:
        citations.append(
            {
                "kind": "chunk",
                "source": c["doc"],
                "title": c["title"],
                "chunk_id": c["chunk_id"],
                "canonical_id": c.get("canonical_id"),
                "score": c["score"],
                "verification_status": c["verification_status"],
            }
        )
        if c["verification_status"] == "logbook_opinion":
            hypotheses.append(f"Запись журнала (мнение жильца, не факт): {c['text'][:220]}")
        elif c["has_gaps"]:
            unknown.append(f"«{c['title']}» содержит незаполненные места — нельзя опираться как на полный источник.")

    # ---------------------------------------------------------------- узлы
    nodes = kb.find_nodes(question)
    if re.search(r"\bIL_[A-Z0-9_]+\b", question.upper()):
        preferred = [n for n in nodes if n["node_id"] == "dome_automation_01"]
        if preferred:
            nodes = preferred
    has_controls = False
    for node in nodes:
        f, u = _node_facts(node)
        facts += f
        unknown += u
        structured_hits += 1
        citations.append(
            {
                "kind": "structured",
                "source": f"data/nodes.json#{node['node_id']}",
                "title": node["title"],
            }
        )
        if intent == "LIVE":
            facts.append(
                "Фактическое значение в базе не хранится: телеметрия запрашивается у живого API. "
                f"Запрос: `{{\"type\":\"query\",\"node_id\":\"{node['node_id']}\",\"query\":\"...\"}}` "
                "или снимок `py kb.py live snapshot`."
            )
        if node["controls"]:
            has_controls = True
        if any(c.get("allowed_for") == "operator" for c in node["controls"]):
            actions.append("Часть действий доступна только Оператору — запросить через `meeting_01`.")
    if has_controls:
        actions.append(
            "Перед каждой командой читать состояние, после — проверять `verify_by`: "
            "`success:true` означает приём команды, а не её исполнение."
        )
    if intent == "LIVE" and not nodes:
        facts.append(
            "Живое значение в базе не хранится: температура, влажность и другие "
            "показания запрашиваются у телеметрии — `py kb.py live snapshot`."
        )
    node_ids = {n["node_id"] for n in nodes}
    generic_machine = any(
        w in qlow for w in ("станк", "оборудован", "производств", "техник")
    )

    def _mentions_nodes(text: str, ids: set[str]) -> bool:
        return bool(ids & set(kb.alias_index.detect(text)))

    # ---------------------------------------------------------- интерлоки
    for il in kb.interlocks_for(question):
        if il.get("_score", 0) < 0.2 and il["id"].upper() not in question.upper():
            continue
        affected = set(il.get("affected", []))
        if (
            node_ids and affected and not (affected & node_ids)
            and "production" not in affected and not generic_machine
        ):
            continue
        structured_hits += 1
        facts.append(
            f"{il['id']} «{il['title']}»: условие — {il.get('condition')}; "
            f"эффект — {il.get('effect')}; снятие — {il.get('clear_rule')}"
        )
        citations.append({"kind": "structured", "source": il.get("source") or "data/curated/interlocks.json",
                          "title": il["id"]})

    # ------------------------------------------------------------- пороги
    for t in kb.thresholds_for(question):
        if t.get("_score", 0) < 0.3:
            continue
        if line_focus and t.get("area") not in ("power", "energy"):
            continue
        if node_ids and not generic_machine:
            mentioned = set(kb.alias_index.detect(f"{t.get('metric')} {t.get('effect')}"))
            if mentioned and not (mentioned & node_ids):
                continue
        structured_hits += 1
        facts.append(
            f"{t['id']}: `{t['metric']}` {t['op']} {_fmt_value(t['value'])}"
            + (f" в течение {t['for_s']} с" if t.get("for_s") else "")
            + (f" (окно {t['window_s']} с)" if t.get("window_s") else "")
            + f" → {t.get('effect') or 'без описания эффекта'}"
            + ("" if t.get("status") == "confirmed" else f" [статус: {t.get('status')}]")
        )
        citations.append({"kind": "structured", "source": t.get("source") or "data/curated/thresholds.json",
                          "title": t["id"]})

    # ------------------------------------------------------------- отказы
    explicit_fault = re.search(r"\bF-?\d+\.\d+\b", question.upper())
    want_faults = intent == "DIAGNOSTIC" or explicit_fault or any(
        w in question.lower() for w in ("отказ", "ошибк", "не работает", "не включа")
    )
    failure_words = ("не работает", "останов", "прекрат", "ошиб", "отказ",
                     "не включа", "не запуска", "пауза", "авари", "сбой")
    faults = kb.faults_for(question) if want_faults else []
    if explicit_fault:
        faults = faults[:1]
    else:
        kept = []
        for f in faults:
            body = " ".join(str(f.get(x) or "") for x in
                            ("title", "what_happens", "how_to_notice", "must_not"))
            if f.get("_score", 0) < 0.45:
                continue
            if node_ids and not _mentions_nodes(body, node_ids):
                continue
            if any(w in qlow for w in ("не работает", "не включа", "не запуска")) and not any(
                w in body.lower() for w in failure_words
            ):
                continue
            kept.append(f)
        faults = kept[:2]
    for f in faults:
        structured_hits += 1
        facts.append(
            f"Каталог {f['id']} «{f['title']}» (категория: {f.get('category')}; наблюдаемость: {f.get('observability')}). "
            + (f"Что происходит: {f['what_happens']} " if f.get("what_happens") else "")
            + (f"Как заметить: {f['how_to_notice']} " if f.get("how_to_notice") else "")
        )
        if f.get("agent_action"):
            facts.append(f"Что может агент ({f['id']}): {f['agent_action']}")
        if f.get("human_action"):
            actions.append(f"{f['id']}: {f['human_action']}")
        if f.get("must_not"):
            actions.append(f"{f['id']} — чего нельзя допустить: {f['must_not']}")
        citations.append({"kind": "structured", "source": f.get("source") or "09_faults.md", "title": f["id"]})
    if faults:
        unknown.append("Запись каталога описывает возможный отказ и не доказывает, что он происходил на станции.")

    # --------------------------------------------------------- диагностика
    diags = kb.diagnostics_for(question) if intent == "DIAGNOSTIC" else []
    if node_ids and diags:
        diags = [
            d for d in diags
            if _mentions_nodes(
                " ".join(str(d.get(x) or "") for x in ("title", "observation", "next_steps")),
                node_ids,
            )
        ]
    for d in diags:
        structured_hits += 1
        facts.append(f"Сценарий {d['id']} «{d['title']}» (система: {d.get('system')}).")
        if d.get("observation"):
            facts.append(f"Что видно: {d['observation']}")
        for i, ch in enumerate(d.get("checks", [])[:8], 1):
            actions.append(f"{d['id']} проверка {i}: {ch}")
        for h in d.get("hypotheses", [])[:6]:
            hypotheses.append(f"{d['id']}: {h}")
        citations.append({"kind": "structured", "source": d.get("source") or "diagnostics", "title": d["id"]})

    # ---------------------------------------------------------- процедуры
    if any(w in question.lower() for w in ("как ", "запуст", "выполн", "порядок", "инструкц", "шаг")):
        for p in kb.procedures_for(question):
            structured_hits += 1
            facts.append(f"Процедура «{p['title']}» (`{p['id']}`).")
            for pre in p.get("preconditions", [])[:6]:
                if isinstance(pre, str):
                    try:
                        pre = json.loads(pre)
                    except json.JSONDecodeError:
                        pass
                if isinstance(pre, dict):
                    facts.append(f"Предусловие: {pre.get('check')}" + (f"; иначе: {pre['else']}" if pre.get("else") else ""))
                else:
                    facts.append(f"Предусловие: {pre}")
            for i, s in enumerate(p.get("steps", [])[:12], 1):
                actions.append(f"{p['id']} шаг {i}: {s}")
            for f in p.get("forbidden", [])[:6]:
                actions.append(f"Запрещено: {f}")
            citations.append({"kind": "structured", "source": p.get("source") or "procedures", "title": p["id"]})

    # ------------------------------------------------------------ политики
    if intent in ("POLICY", "LIVE") or (nodes and intent == "DIAGNOSTIC"):
        kept_policies = 0
        for p in kb.policies_for(question):
            if p.get("match", 0) < 0.5:
                continue
            kept_policies += 1
            if kept_policies > 2:
                break
            structured_hits += 1
            facts.append(f"{p['id']} ({p['scope']}): {p['rule']}" + (f" → {p['action']}" if p.get("action") else ""))
            citations.append({"kind": "structured", "source": p.get("source") or "data/curated/policies.json",
                              "title": p["id"]})

    # --------------------------------------------------------------- пробелы
    gaps = [g for g in kb.gaps_for(question, k=3) if g.get("match", 0) >= 0.5][:2]
    if node_ids and not line_focus:
        systems = {n["system"] for n in nodes}
        gaps = [
            g for g in gaps
            if g.get("area") in systems or g.get("area") == "terminology" or g["id"] == "GAP-001"
        ]
    if line_focus or any(n["node_id"] == "smart_panel_01" for n in nodes):
        g1 = next((g for g in kb.all_gaps() if g["id"] == "GAP-001"), None)
        if g1 and all(g["id"] != "GAP-001" for g in gaps):
            gaps.insert(0, g1)
    if gaps:
        gf, gu = _gap_facts(gaps)
        structured_hits += 1
        for g in gaps:
            citations.append({"kind": "structured", "source": g.get("source") or "data/curated/gaps.json",
                              "title": g["id"]})
        if any(g["id"] == "GAP-001" for g in gaps):
            # прямой ответ важнее описания узлов: убираем шум параметров и чужие пороги
            facts = [
                f for f in facts
                if not f.startswith(("параметр ", "состояния:", "команды:"))
                and not f.startswith(("TH_CO2", "TH_CONDENSATE", "TH_RESUME", "TH_CHAT"))
            ]
            facts = gf + facts
            unknown = gu + unknown
        else:
            facts += gf
            unknown += gu

    # ------------------------------------------------------------- глоссарий
    if not nodes and not faults and intent == "REFERENCE" and not no_data:
        for g in kb.glossary_for(question):
            if g.get("_score", 0) >= 0.2:
                facts.append(f"«{g['term']}»: {g['definition']}")
                citations.append({"kind": "structured", "source": g.get("source") or "11_reference.md",
                                  "title": g["term"]})
                structured_hits += 1

    # ------------------------------------------------------- честный отказ
    if no_data:
        # Вопрос вне корпуса: не показываем случайные фрагменты как «факты».
        facts = []
        hypotheses = []
        actions = []
        citations = []
        chunks = []
        structured_hits = 0
        unknown = ["В корпусе нет сведений: " + ", ".join(no_data) + "."]

    # ------------------------------------------------------------ уверенность
    top_score = chunks[0]["score"] if chunks else 0.0
    score = 0.15 + 0.45 * min(1.0, top_score * 1.5) + 0.05 * min(6, structured_hits)
    if unknown:
        score = min(score, 0.68)
    if no_data:
        score = 0.3
    if not chunks and structured_hits == 0 and not no_data:
        score = 0.05
    score = round(min(0.95, score), 2)
    level = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
    abstained = (not chunks and structured_hits == 0) or bool(no_data)
    confidence = {
        "score": score,
        "level": level,
        "reason": (
            f"лучший фрагмент {top_score:.2f}, структурных попаданий {structured_hits}, "
            f"пробелов/неизвестного {len(unknown)}"
        ),
    }

    result = {
        "question": question,
        "intent": intent,
        "abstained": abstained,
        "facts": list(dict.fromkeys(facts)),
        "hypotheses": list(dict.fromkeys(hypotheses)),
        "unknown": list(dict.fromkeys(unknown)),
        "actions": list(dict.fromkeys(actions)),
        "citations": citations,
        "confidence": confidence,
        "chunks": chunks[:5],
        "mode": "deterministic",
    }
    result["answer"] = _build_text(
        question, intent, result["facts"], result["hypotheses"], result["unknown"],
        result["actions"], citations, chunks[:5], confidence,
    )

    if use_llm:
        context = "\n\n".join(f"[{c['doc']} :: {c['title']}]\n{c['text'][:1800]}" for c in chunks[:6])
        text = _call_deepseek(question, context, result["facts"])
        if text:
            result["answer"] = text
            result["mode"] = "llm+deterministic"
        else:
            result["mode"] = "deterministic (LLM недоступна)"
            result["unknown"].append("LLM-режим запрошен, но ключ/сеть недоступны — показан детерминированный ответ.")

    return result


def render(result: dict) -> str:
    return result["answer"]
