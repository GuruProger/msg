"""Векторное представление чанков без внешних зависимостей.

Почему не нейросетевые эмбеддинги: в среде нет ``sentence-transformers`` и
numpy, а скачивать модель на 2 ГБ ради демонстрации неразумно. Здесь работает
разреженный TF-IDF по словам и символьным 4-граммам:

* слова дают точные совпадения (`resume`, `E07`, `bed_clear`);
* 4-граммы дают устойчивость к русской морфологии
  («температура» / «температуры» / «температурный»);
* IDF придаёт вес редким идентификаторам и гасит общие слова.

Формат ``build/vectors.json`` совместим с будущей заменой на плотные векторы:
``vectors`` — либо ``{term: weight}``, либо список чисел. Поиск умеет и то и другое.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from functools import lru_cache

from .paths import BUILD, CARD_VECTORS_PATH, CARDS_PATH, VECTORS_PATH

MODEL_NAME = "word+char4-tfidf-v1"
MAX_FEATURES_PER_CHUNK = 400

SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
TOKEN_RE = re.compile(r"[a-zа-я0-9]+")

STOP = {
    "и", "в", "во", "на", "с", "со", "по", "что", "как", "это", "для", "из",
    "не", "или", "а", "но", "при", "до", "после", "если", "то", "же", "ли",
    "у", "о", "об", "от", "за", "над", "под", "без", "более", "менее",
    "может", "можно", "нужно", "должен", "быть", "есть", "чем", "так",
    "его", "её", "их", "мы", "вы", "они", "он", "она", "оно", "я",
    "the", "and", "of", "to", "in", "is", "a",
}


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е").translate(SUBSCRIPT)
    return text


@lru_cache(maxsize=8192)
def _tokens(text: str) -> tuple[str, ...]:
    return tuple(t for t in TOKEN_RE.findall(normalize_text(text)) if t not in STOP)


def tokenize(text: str) -> list[str]:
    """Значимые слова. Кэш важен: один и тот же текст карточки проверяется
    несколькими каналами поиска и шестью осями покрытия за один запрос."""
    return list(_tokens(text))


# Грубая, но воспроизводимая основа слова. Нужна трём слоям сразу:
# сопоставлению сущностей (падежи), FTS-запросу и проверке покрытия. Раньше
# она жила в retrieval.py, из-за чего chunker/cards не могли ей пользоваться и
# теряли сущность в заголовке «падение солнечной генерации».
_ENDINGS = (
    "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими",
    "ая", "яя", "ое", "ее", "ые", "ие", "ых", "их", "ой", "ей", "ий", "ый",
    "ам", "ям", "ах", "ях", "ом", "ем",
    "а", "я", "ы", "и", "у", "ю", "е", "о", "ь",
)


@lru_cache(maxsize=65536)
def stem(token: str) -> str:
    token = normalize_text(token)
    for ending in _ENDINGS:
        if len(token) > len(ending) + 2 and token.endswith(ending):
            return token[: -len(ending)][:7]
    return token[:7]


@lru_cache(maxsize=8192)
def _stems(text: str) -> frozenset[str]:
    return frozenset(stem(t) for t in tokenize(text) if len(t) >= 3)


def stems(text: str) -> frozenset[str]:
    """Неизменяемый набор основ: вызывается на каждом сравнении карточки."""
    return _stems(text)


def features(text: str) -> Counter:
    """Слова + символьные 4-граммы слов длиной >= 4."""
    out: Counter = Counter()
    for tok in tokenize(text):
        out["w:" + tok] += 1
        if len(tok) >= 4 and not tok.isdigit():
            padded = "<" + tok + ">"
            for i in range(len(padded) - 3):
                out["c:" + padded[i:i + 4]] += 1
    return out


def _try_sentence_transformer():
    """Точка расширения: если появится sentence-transformers, включится сам."""
    if os.environ.get("DOMEKB_EMBEDDINGS", "").lower() not in ("st", "sentence-transformers"):
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    name = os.environ.get("DOMEKB_ST_MODEL", "intfloat/multilingual-e5-small")
    try:
        return name, SentenceTransformer(name)
    except Exception:
        return None


def build_vectors(chunks: list[dict], path=None) -> dict:
    path = path or VECTORS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    st = _try_sentence_transformer()
    if st:
        name, model = st
        texts = [c["search_text"] for c in chunks]
        emb = model.encode(texts, normalize_embeddings=True)
        payload = {
            "model": f"st:{name}",
            "n_docs": len(chunks),
            "idf": {},
            "vectors": {c["chunk_id"]: [round(float(x), 6) for x in v] for c, v in zip(chunks, emb)},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    docs = [features(c["search_text"]) for c in chunks]
    df: Counter = Counter()
    for f in docs:
        df.update(f.keys())
    n_docs = len(docs)
    idf = {t: round(math.log((n_docs + 1) / (d + 1)) + 1.0, 6) for t, d in df.items()}

    vectors: dict[str, dict] = {}
    for chunk, f in zip(chunks, docs):
        weights = {t: (1.0 + math.log(n)) * idf[t] for t, n in f.items()}
        top = sorted(weights.items(), key=lambda kv: -kv[1])[:MAX_FEATURES_PER_CHUNK]
        norm = math.sqrt(sum(v * v for _, v in top)) or 1.0
        vectors[chunk["chunk_id"]] = {t: round(v / norm, 5) for t, v in top}

    payload = {
        "model": MODEL_NAME,
        "n_docs": n_docs,
        "idf": idf,
        "vectors": vectors,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def query_vector(text: str, payload: dict) -> dict:
    """Вектор запроса в том же пространстве; неизвестные термины отбрасываются."""
    idf = payload.get("idf") or {}
    if not idf:
        return {}
    f = features(text)
    weights = {t: (1.0 + math.log(n)) * idf[t] for t, n in f.items() if t in idf}
    norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
    return {t: v / norm for t, v in weights.items()}


def cosine(a, b) -> float:
    if isinstance(a, list) or isinstance(b, list):
        if not isinstance(a, list) or not isinstance(b, list):
            return 0.0
        return float(sum(x * y for x, y in zip(a, b)))
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b.get(t, 0.0) for t, v in a.items()))


def cmd_vectors() -> None:
    chunks_path = BUILD.parent / "chunks" / "knowledge.jsonl"
    episodes_path = BUILD.parent / "chunks" / "episodes.jsonl"
    chunks: list[dict] = []
    for p in (chunks_path, episodes_path):
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                chunks += [json.loads(line) for line in fh if line.strip()]
    if not chunks:
        raise SystemExit("Нет чанков: сначала выполните `py kb.py chunk`.")
    payload = build_vectors(chunks)
    size = VECTORS_PATH.stat().st_size / 1024 / 1024
    print(f"vectors     {len(payload['vectors'])}  ({size:.1f} МБ)")
    print(f"model       {payload['model']}")
    print(f"features    {len(payload['idf'])}")

    if CARDS_PATH.exists():
        cards = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
        proxies = [
            {"chunk_id": card["card_id"], "search_text": card.get("search_text") or ""}
            for card in cards
        ]
        card_payload = build_vectors(proxies, path=CARD_VECTORS_PATH)
        card_size = CARD_VECTORS_PATH.stat().st_size / 1024 / 1024
        print(f"cards       {len(card_payload['vectors'])}  ({card_size:.1f} МБ)")
