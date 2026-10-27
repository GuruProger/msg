"""Таксономия атрибутов знания.

Карточка отвечает не на «вопрос целиком», а на пару **субъект + атрибут**.
Таксономия — это список допустимых атрибутов и их синонимов на русском и
английском. Она нужна сразу двум модулям:

* ``cards`` — чтобы разложить корпус по полочкам «что именно здесь сказано»;
* ``coverage`` — чтобы понять, что именно спрашивают, и проверить, что
  найденная карточка отвечает на этот вопрос, а не просто содержит похожие
  слова.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from . import vectors
from .paths import DATA

ATTRIBUTES_PATH = DATA / "curated" / "attributes.json"
SHORT_SYNONYM = 4


@lru_cache(maxsize=1)
def payload() -> dict:
    return json.loads(ATTRIBUTES_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def attributes() -> dict[str, dict]:
    return {item["id"]: item for item in payload()["attributes"]}


@lru_cache(maxsize=1)
def order() -> list[str]:
    return [item["id"] for item in payload()["attributes"]]


@lru_cache(maxsize=1)
def families() -> dict[str, list[str]]:
    return payload().get("families", {})


def label(attribute: str) -> str:
    return attributes().get(attribute, {}).get("label", attribute)


def family(attribute: str) -> str:
    return attributes().get(attribute, {}).get("family", "generic")


def is_off_domain(attribute: str) -> bool:
    return family(attribute) == "off_domain"


def family_members(attribute: str) -> set[str]:
    """Атрибуты, которые действительно могут отвечать на запрошенный.

    Связи направленные: «условие срабатывания» может быть описано порогом,
    но «последствия» нельзя подменить определением узла. Раньше в каждое
    семейство добавлялся ``section_text``, из-за чего любой абзац считался
    ответом на любой вопрос.
    """
    out = {attribute}
    out.update(RELATED.get(attribute, set()))
    return out


RELATED: dict[str, set[str]] = {
    # Порог отвечает и числом, и последствием: «сколько попыток повторного
    # включения» записано в эффекте TH_OVERCURRENT, а не в самой метрике тока.
    "threshold_value": {"interlock_trigger", "interlock_effect"},
    "interlock_trigger": {"threshold_value"},
    # «Что произойдёт, если…» может быть описано правилом/запретом
    # («start_job только при bed_clear=true»), а не отдельной карточкой
    # последствия.
    "interlock_effect": {"policy_rule", "fault_forbidden", "fault_action"},
    "interlock_clear": set(),
    "fault_cause": {"fault_symptom"},
    "fault_symptom": {"fault_cause"},
    # FAQ «Что делать, если…» хранится как definition, но по смыслу это
    # готовый порядок действий. Направление важное: определение не должно
    # подменять действие, а действие может опираться на FAQ.
    "fault_action": {
        "procedure_steps", "fault_forbidden", "policy_rule",
        "procedure_precondition", "definition",
    },
    "fault_forbidden": {"policy_rule", "fault_action"},
    "procedure_steps": {"fault_action", "definition"},
    "procedure_precondition": {"policy_rule"},
    "policy_rule": {"fault_forbidden"},
    "definition": {"interlock_trigger", "policy_rule"},
    # «Какой параметр показывает заряд» — это и описание метрики, и порог,
    # в названии которого она записана.
    "measurement_spec": {"equipment_spec", "control_capability", "threshold_value"},
    "control_capability": {"measurement_spec"},
    "live_state": {"measurement_spec"},
    "history_event": {"fault_cause"},
    "equipment_spec": {"measurement_spec"},
    "gap": {"policy_rule"},
    "location": {"equipment_spec"},
    "quantity": {"threshold_value", "measurement_spec", "control_capability", "equipment_spec"},
    "comparison": {
        "definition", "threshold_value", "interlock_effect",
        "history_event", "policy_rule",
    },
}


@lru_cache(maxsize=4096)
def _synonym_re(synonym: str) -> re.Pattern[str]:
    normalized = vectors.normalize_text(synonym).strip()
    ascii_single = bool(re.fullmatch(r"[a-z0-9_]+", normalized))
    if (len(normalized) <= SHORT_SYNONYM and " " not in normalized) or (
        ascii_single and len(normalized) <= 8
    ):
        # «clear» не должен срабатывать внутри `bed_clear`: это поле
        # телеметрии, а не команда снятия интерлока.
        return re.compile(
            rf"(?<![a-zа-я0-9_]){re.escape(normalized)}(?![a-zа-я0-9_])"
        )
    return re.compile(re.escape(normalized))


def hits(text: str, synonym: str) -> bool:
    normalized = vectors.normalize_text(text)
    return bool(_synonym_re(synonym).search(normalized))


def score_text(text: str, attribute: str) -> float:
    """Насколько текст похож на формулировку данного атрибута."""
    item = attributes().get(attribute)
    if not item:
        return 0.0
    score = 0.0
    for synonym in item.get("synonyms", []):
        if hits(text, synonym):
            score += 1.0 + min(len(synonym), 24) / 24.0
    return score


def detect_attributes(text: str, limit: int = 3) -> list[str]:
    """Атрибуты вопроса, самые специфичные первыми.

    Специфичность — это вес атрибута плюс длина совпавшего синонима. Поэтому
    «серийный номер» (off_domain, вес 5) побеждает общее «сколько».
    """
    scored: list[tuple[float, int, str]] = []
    for position, attribute in enumerate(order()):
        item = attributes()[attribute]
        raw = score_text(text, attribute)
        if raw <= 0:
            continue
        weighted = raw * float(item.get("weight", 1))
        scored.append((weighted, -position, attribute))
    scored.sort(reverse=True)
    return [attribute for _, _, attribute in scored[:limit]]


def infer_attribute(text: str, default: str = "section_text") -> str:
    """Атрибут для карточки-раздела. Off-domain атрибуты здесь не ставятся."""
    best, best_score = default, 0.0
    for attribute in order():
        if is_off_domain(attribute) or attribute == "section_text":
            continue
        score = score_text(text, attribute)
        if score > best_score:
            best, best_score = attribute, score
    return best


def keyword_terms(text: str) -> list[str]:
    """Значимые слова вопроса без стоп-слов — для лексического перекрытия."""
    seen: list[str] = []
    for token in vectors.tokenize(text):
        if len(token) >= 4 and token not in seen:
            seen.append(token)
    return seen[:24]


def describe(attribute: str) -> str:
    item = attributes().get(attribute, {})
    return f"{attribute} ({item.get('label', '?')})"


def path() -> Path:
    return ATTRIBUTES_PATH
