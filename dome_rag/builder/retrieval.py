"""Гибридный поиск: FTS5 (BM25) + разреженный TF-IDF + переранжирование.

Лексический поиск обязателен: в предметной области полно точных
идентификаторов (`IL_CO2`, `E07`, `bed_clear`, `probe_z`), которые плотные
эмбеддинги часто «размывают». Векторная часть нужна для вопросов, где слов
в документе нет, но смысл близок.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path

from . import vectors as vec
from .chunker import AliasIndex
from .paths import BUILD, CARDS_PATH, CARD_VECTORS_PATH, DATA, DB_PATH, VECTORS_PATH

DEFAULT_COLLECTIONS = ("knowledge", "history", "episodes")
COLLECTION_WEIGHT = {"knowledge": 1.0, "history": 0.97, "episodes": 0.92}


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _jload(value, default):
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _stem(token: str) -> str:
    """Обратная совместимость: основа слова живёт в ``vectors``."""
    return vec.stem(token)


stems = vec.stems

def overlap(query_stems: set[str], text: str) -> float:
    if not query_stems:
        return 0.0
    doc = stems(text)
    if not doc:
        return 0.0
    return len(query_stems & doc) / len(query_stems)


NEG_RE = re.compile(r"\bне\s+([а-я]{4,})")


def negation_penalty(question: str, text: str) -> float:
    """«Не работает» не должно подтверждаться фразой «принтер работает»."""
    low = vec.normalize_text(text)
    for word in NEG_RE.findall(vec.normalize_text(question)):
        stem = word[:5]
        if stem in low and not re.search(rf"не\s+{re.escape(stem)}", low):
            return 0.35
    return 0.0


class KB:
    def __init__(
        self,
        db_path: Path | None = None,
        vectors_path: Path | None = None,
        *,
        check_same_thread: bool = True,
        data_dir: Path | None = None,
    ):
        self.db_path = Path(db_path or DB_PATH).resolve()
        self.data_dir = Path(data_dir) if data_dir else self.db_path.parent.parent / "data"
        build_dir = self.db_path.parent
        if not self.db_path.exists():
            raise SystemExit(
                f"База не собрана: {self.db_path}\nВыполните: py kb.py build"
            )
        self.con = sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True, check_same_thread=check_same_thread)
        self.con.row_factory = sqlite3.Row
        self._vectors = _load_json(Path(vectors_path or build_dir / "vectors.json"), None)
        nodes = [dict(r) for r in self.con.execute(
            "SELECT node_id, code_name, title, aliases_json FROM nodes"
        )]
        aliases = _load_json(self.data_dir / "curated" / "aliases.json", {})
        self.alias_index = AliasIndex(aliases, nodes)
        self.node_rows = nodes
        self.cards = self._load_cards()
        self.card_vectors = _load_json(build_dir / "card_vectors.json", None)
        self._card_by_id = {card["card_id"]: card for card in self.cards}
        # Обратный индекс разреженных векторов: запрос касается единиц
        # терминов, а не всех 1400+ карточек. Это снижает p95 и делает
        # векторный канал детерминированным при любом размере корпуса.
        self._card_postings: dict[str, list[tuple[str, float]]] = {}
        for card_id, weights in ((self.card_vectors or {}).get("vectors") or {}).items():
            if not isinstance(weights, dict):
                continue
            for term, weight in weights.items():
                self._card_postings.setdefault(term, []).append((card_id, float(weight)))
        self._cards_by_entity: dict[str, list[str]] = {}
        self._cards_by_code: dict[str, list[str]] = {}
        self._cards_by_attribute: dict[str, list[str]] = {}
        for card in self.cards:
            for entity in card.get("entities") or []:
                self._cards_by_entity.setdefault(entity, []).append(card["card_id"])
            for code in card.get("codes") or []:
                self._cards_by_code.setdefault(str(code), []).append(card["card_id"])
            self._cards_by_attribute.setdefault(card["attribute"], []).append(card["card_id"])
        # Пары основ каждой карточки считаются один раз при загрузке, а не в
        # каждом канале поиска. Это главный вклад в задержку p95.
        self._card_pairs: dict[str, frozenset[tuple[str, str]]] = {}
        for card in self.cards:
            tokens = [vec.stem(t) for t in vec.tokenize(self._card_text(card))]
            self._card_pairs[card["card_id"]] = frozenset(zip(tokens, tokens[1:]))

    def _load_cards(self) -> list[dict]:
        cards = _load_json(self.data_dir / "cards.json", [])
        if not cards:
            try:
                rows = self.con.execute("SELECT * FROM cards").fetchall()
            except sqlite3.OperationalError:
                rows = []
            cards = [
                {
                    "card_id": r["card_id"], "subject": r["subject"],
                    "subject_label": r["subject_label"], "attribute": r["attribute"],
                    "value": r["value"], "text": r["text"],
                    "entities": _jload(r["entities"], []),
                    "codes": _jload(r["codes"], []),
                    "source": {"doc": r["source_doc"], "title": r["source_title"],
                               "line": r["source_line"]},
                    "authority": r["authority"], "status": r["status"],
                    "canonical_id": r["canonical_id"], "chunk_id": r["chunk_id"],
                    "has_gaps": bool(r["has_gaps"]), "search_text": r["search_text"],
                }
                for r in rows
            ]
        return cards

    # ------------------------------------------------------------------ поиск

    def detect_entities(self, text: str) -> list[str]:
        """Сущности вопроса с учётом падежей и вложенных алиасов.

        Вся логика живёт в ``AliasIndex``: длинная фраза маскирует короткую,
        а падежные формы ловятся по основам. Дублировать её здесь нельзя —
        иначе «вытяжная вентиляция» снова добавляла бы «приточную».
        """
        return self.alias_index.detect(text)[:6]

    def _fts_query(self, text: str) -> str:
        """FTS-запрос: исходные слова с префиксом.

        Основа слова здесь не добавляется сознательно. Префиксы коротких
        основ («как»*, «влажнос»*) заставляют FTS5 перебирать почти весь
        словарь и стоят 20–35 мс на запрос, тогда как морфологию уже
        покрывают символьные 4-граммы векторного канала и проверка основ в
        покрытии. Исходная форма с ``*`` находит падежи, начинающиеся с того
        же корня, и стоит 2–3 мс.
        """
        toks = list(dict.fromkeys(vec.tokenize(text)))
        if not toks:
            return ""
        return " OR ".join(
            f'"{t}"*' if len(t) >= 3 else f'"{t}"' for t in toks
        )

    def search(
        self,
        question: str,
        k: int = 8,
        collections=DEFAULT_COLLECTIONS,
        types=None,
        doc: str | None = None,
        include_duplicates: bool = False,
    ) -> list[dict]:
        question = (question or "").strip()
        if not question:
            return []
        collections = tuple(collections) if collections else DEFAULT_COLLECTIONS
        types = (types,) if isinstance(types, str) else tuple(types or ())

        where = [f"c.collection IN ({','.join('?' * len(collections))})"]
        params: list = list(collections)
        if types:
            where.append(f"c.type IN ({','.join('?' * len(types))})")
            params += list(types)
        if doc:
            where.append("c.doc = ?")
            params.append(doc)
        if not include_duplicates:
            where.append("(c.is_duplicate_of IS NULL OR c.is_duplicate_of = '')")

        rows: list[sqlite3.Row] = []
        fts = self._fts_query(question)
        if fts:
            sql = (
                "SELECT c.*, bm25(chunks_fts) AS rank FROM chunks_fts "
                "JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id "
                f"WHERE chunks_fts MATCH ? AND {' AND '.join(where)} "
                "ORDER BY rank LIMIT 120"
            )
            try:
                rows = list(self.con.execute(sql, [fts] + params))
            except sqlite3.OperationalError:
                rows = []

        q_stems = stems(question)
        q_upper = question.upper()
        q_entities = set(self.alias_index.detect(question))
        q_vec = {}
        if self._vectors and self._vectors.get("idf"):
            q_vec = vec.query_vector(question, self._vectors)

        # если FTS не дал ничего (опечатка, редкая формулировка) — добираем по словам
        if not rows:
            like = " OR ".join("c.search_text LIKE ?" for _ in vec.tokenize(question)[:8])
            if like:
                like_params = [f"%{t}%" for t in vec.tokenize(question)[:8]]
                sql = f"SELECT c.*, 0.0 AS rank FROM chunks c WHERE ({like}) AND {' AND '.join(where)} LIMIT 120"
                try:
                    rows = list(self.con.execute(sql, like_params + params))
                except sqlite3.OperationalError:
                    rows = []

        if not rows:
            return []

        ranks = [-(r["rank"] or 0.0) for r in rows]
        max_rank = max(ranks) or 1.0

        # редкие идентификаторы (E07, IL_CO2, bed_clear) должны побеждать «общие» слова
        raw_terms = re.findall(r"[a-zа-я0-9_]+", vec.normalize_text(question))
        code_terms = [
            t for t in raw_terms
            if len(t) >= 3 and ("_" in t or any(ch.isdigit() for ch in t))
        ]
        norm_texts = [vec.normalize_text(r["text"]) for r in rows]
        term_df = {t: sum(1 for x in norm_texts if t in x) for t in code_terms}
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", question)

        scored: list[dict] = []
        for idx, (row, rank) in enumerate(zip(rows, ranks)):
            cid = row["chunk_id"]
            cos = 0.0
            if q_vec and self._vectors:
                cos = vec.cosine(q_vec, self._vectors["vectors"].get(cid, {}))
            bm25 = max(0.0, rank) / max_rank
            score = 0.55 * cos + 0.45 * bm25

            for term in code_terms:
                if term not in norm_texts[idx]:
                    continue
                score += 0.45 if term_df.get(term, 99) == 1 else 0.18 if term_df.get(term, 99) <= 3 else 0.0
            for d in dates:
                if d in norm_texts[idx]:
                    score += 0.40
            score -= negation_penalty(question, row["text"])

            canonical = row["canonical_id"] or ""
            if canonical and canonical.upper() in q_upper:
                score += 0.30
            entities = set(_jload(row["entities"], []))
            if entities & q_entities:
                score += min(0.24, 0.12 * len(entities & q_entities))
            title_ov = overlap(q_stems, row["title"] or "")
            score += 0.08 * title_ov
            score += 0.005 * ((row["source_priority"] or 3) - 3)
            if (row["title"] or "").strip().lower().startswith(("пример", "пример вызова", "пример ответа")):
                score -= 0.12
            score *= COLLECTION_WEIGHT.get(row["collection"], 0.9)
            if row["has_gaps"]:
                score -= 0.02

            scored.append(
                {
                    "chunk_id": cid,
                    "collection": row["collection"],
                    "doc": row["doc"],
                    "title": row["title"],
                    "heading_path": _jload(row["heading_path"], []),
                    "type": row["type"],
                    "canonical_id": row["canonical_id"],
                    "entities": sorted(entities),
                    "verification_status": row["verification_status"],
                    "has_gaps": bool(row["has_gaps"]),
                    "source_priority": row["source_priority"],
                    "line": row["line"],
                    "text": row["text"],
                    "score": round(score, 4),
                    "why": {
                        "cosine": round(cos, 4),
                        "bm25": round(bm25, 4),
                        "title_overlap": round(title_ov, 3),
                        "entities": sorted(entities & q_entities),
                    },
                }
            )

        scored.sort(key=lambda r: -r["score"])

        # один лучший чанк на каноническую сущность: не показывать 5 копий порога
        seen: set[str] = set()
        out = []
        for r in scored:
            key = r["canonical_id"] or (r["doc"] + "|" + (r["title"] or ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------- карточки v2

    def _card_text(self, card: dict) -> str:
        return " ".join(
            str(part or "")
            for part in (
                card.get("subject_label"), card.get("subject"), card.get("value"),
                card.get("text"), card.get("canonical_id"),
            )
        )

    def search_cards(self, question: str, k: int = 8, slots=None) -> list[dict]:
        """Поиск по карточкам: 5 каналов + RRF + детерминированный реранкер.

        Каналы:

        1. ``exact``      — точные коды, сущности, название субъекта;
        2. ``phrase``     — совпадение формулировки (соседние пары слов);
        3. ``fts``        — BM25 по FTS5 (исходные слова + основы);
        4. ``vector``     — разреженный TF-IDF (слова + символьные 4-граммы);
        5. ``structural`` — точная привязка «сущность × атрибут».

        Слияние — взвешенный Reciprocal Rank Fusion: каналы голосуют рангами,
        а не несравнимыми оценками. Финальный реранкер линейный и объяснимый.
        """
        from . import coverage

        question = (question or "").strip()
        if not question or not self.cards:
            return []
        slots = slots or coverage.extract_slots(question, entities=self.detect_entities(question))
        candidates: dict[str, float] = {}
        channels: dict[str, list[str]] = {}

        q_tokens = [vec.stem(t) for t in vec.tokenize(question)]
        q_pairs = set(zip(q_tokens, q_tokens[1:]))
        q_pair_count = max(1, len(q_pairs))

        def card_phrase(card: dict) -> float:
            return len(q_pairs & self._card_pairs.get(card["card_id"], frozenset())) / q_pair_count

        # ---- 1. exact: коды, сущности, точная тема
        qcodes = {str(c) for c in slots.codes}
        qents = set(slots.entities)
        exact: list[tuple[float, str]] = []
        for card in self.cards:
            score = 0.0
            codes = {str(c) for c in (card.get("codes") or [])}
            entities = set(card.get("entities") or [])
            if qcodes & codes:
                score += 1.6
            if qents & entities:
                score += (
                    0.95 if card.get("attribute") in slots.attributes else 0.25
                )
            subject = coverage.subject_match(question, card)
            if subject >= 0.99:
                score += 1.4
            elif subject >= 0.60:
                score += 0.35
            phrase = card_phrase(card)
            if phrase >= 0.45:
                score += 0.9 * phrase
            if score >= 0.40:
                exact.append((score, card["card_id"]))
        exact.sort(key=lambda item: (-item[0], item[1]))
        channels["exact"] = [cid for _, cid in exact[:160]]

        # ---- 2. phrase: формулировка, а не набор слов
        phrase_rows = [
            (card_phrase(card), card["card_id"]) for card in self.cards
        ]
        phrase_rows = [item for item in phrase_rows if item[0] >= 0.30]
        phrase_rows.sort(key=lambda item: (-item[0], item[1]))
        channels["phrase"] = [cid for _, cid in phrase_rows[:160]]

        # ---- 3. FTS5 / BM25
        fts_ranked: list[str] = []
        fts = self._fts_query(question)
        if fts:
            try:
                rows = self.con.execute(
                    "SELECT card_id, bm25(cards_fts) AS rank FROM cards_fts "
                    "WHERE cards_fts MATCH ? ORDER BY rank LIMIT 200",
                    [fts],
                )
                fts_ranked = [row["card_id"] for row in rows]
            except sqlite3.OperationalError:
                fts_ranked = []
        if not fts_ranked:
            # запасной лексический канал: не падать, если FTS-таблицы нет
            q_stems = stems(question)
            scored = [
                (overlap(q_stems, self._card_text(card)), card["card_id"])
                for card in self.cards
            ]
            scored.sort(key=lambda item: (-item[0], item[1]))
            fts_ranked = [cid for score, cid in scored[:160] if score > 0]
        channels["fts"] = fts_ranked

        # ---- 4. разреженный TF-IDF через обратный индекс
        vector_ranked: list[str] = []
        if self.card_vectors and self.card_vectors.get("idf"):
            q_vec = vec.query_vector(question, self.card_vectors)
            scores: dict[str, float] = {}
            for term, weight in q_vec.items():
                for card_id, card_weight in self._card_postings.get(term, ()):
                    scores[card_id] = scores.get(card_id, 0.0) + weight * card_weight
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            vector_ranked = [cid for cid, _ in ranked[:160] if _ > 0]
        channels["vector"] = vector_ranked

        # ---- 5. structural: сущность × атрибут, упорядоченно
        struct_scores: dict[str, float] = {}
        for entity in qents:
            for cid in self._cards_by_entity.get(entity, ()):
                struct_scores[cid] = struct_scores.get(cid, 0.0) + 1.2
        for code in qcodes:
            for cid in self._cards_by_code.get(code, ()):
                struct_scores[cid] = struct_scores.get(cid, 0.0) + 1.6
        wanted_attributes = set(slots.attributes)
        wanted_attributes.update(coverage.FALLBACK_ATTRIBUTES)
        for attribute in wanted_attributes:
            weight = 1.0 if attribute in slots.attributes else 0.35
            for cid in self._cards_by_attribute.get(attribute, ()):
                struct_scores[cid] = struct_scores.get(cid, 0.0) + weight
        structural = sorted(
            struct_scores.items(),
            key=lambda item: (
                -item[1],
                -int((self._card_by_id.get(item[0]) or {}).get("authority") or 3),
                item[0],
            ),
        )
        channels["structural"] = [cid for cid, _ in structural[:220]]

        # ---- взвешенный RRF
        RRF_K = 60
        channel_weight = {
            "exact": 1.40, "phrase": 1.25, "structural": 1.15,
            "fts": 1.00, "vector": 0.85,
        }
        for name, ranked in channels.items():
            weight = channel_weight.get(name, 1.0)
            for rank, cid in enumerate(ranked, 1):
                candidates[cid] = candidates.get(cid, 0.0) + weight / (RRF_K + rank)

        max_rrf = max(candidates.values()) or 1.0
        scored_cards: list[dict] = []
        for cid, rrf in candidates.items():
            card = self._card_by_id.get(cid)
            if card is None:
                continue
            cov = coverage.score_card(slots, card)
            authority = int(card.get("authority") or 3) / 5.0
            exact_attr = 1.0 if card.get("attribute") in slots.attributes else 0.0
            final = (
                0.44 * (rrf / max_rrf)
                + 0.38 * cov["total"]
                + 0.08 * authority
                + 0.05 * exact_attr
                + 0.05 * cov["lexical"]
            )
            if card.get("has_gaps"):
                final -= 0.02
            item = dict(card)
            item["score"] = round(final, 4)
            item["coverage"] = cov["total"]
            item["why"] = {
                "rrf": round(rrf, 5),
                "channels": sorted(
                    name for name, ranked in channels.items() if cid in set(ranked)
                ),
                "attribute_match": cov["attribute"],
                "entity_match": cov["entity"],
                "code_match": cov["code"],
                "lexical": cov["lexical"],
                "subject": cov["subject"],
                "phrase": cov["phrase"],
            }
            scored_cards.append(item)

        scored_cards.sort(
            key=lambda c: (-c["score"], -int(c.get("authority") or 3), c["card_id"])
        )

        # Один факт на пару (канонический id, атрибут). Копии одного и того же
        # утверждения из разных документов не должны занимать весь топ.
        out: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for card in scored_cards:
            key = (
                str(card.get("canonical_id") or card.get("subject") or card["card_id"]),
                str(card.get("attribute") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(card)
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------- структурные данные

    def nodes(self) -> list[dict]:
        return [dict(r) for r in self.con.execute(
            "SELECT node_id, code_name, title, system, system_ru FROM nodes ORDER BY node_id"
        )]

    def node_details(self, node_id: str) -> dict | None:
        row = self.con.execute(
            "SELECT node_id, code_name, title, system, system_ru, aliases_json, source "
            "FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if not row:
            return None
        node = dict(row)
        node["params"] = [dict(r) for r in self.con.execute(
            "SELECT key, unit, kind, trust, sample_json FROM node_params "
            "WHERE node_id = ? ORDER BY kind, key", (node_id,)
        )]
        node["controls"] = [dict(r) for r in self.con.execute(
            "SELECT action, label, allowed_for, api_confirmed, verify_by_json, "
            "value_schema_json, notes, preconditions_json FROM node_controls "
            "WHERE node_id = ? ORDER BY action", (node_id,)
        )]
        return node

    def find_nodes(self, text: str) -> list[dict]:
        ids = list(dict.fromkeys(self.alias_index.detect(text)))
        low = vec.normalize_text(text)
        for n in self.node_rows:
            if n["node_id"] in ids:
                continue
            for probe in (n["node_id"], n["code_name"], n["title"]):
                if probe and len(probe) >= 5 and vec.normalize_text(probe) in low:
                    ids.append(n["node_id"])
                    break
        out = []
        for nid in ids[:4]:
            d = self.node_details(nid)
            if d:
                out.append(d)
        return out

    def _rank_rows(self, sql: str, text: str, fields: tuple[str, ...], k: int = 3,
                   boost: tuple[str, ...] = (), key_fields: tuple[str, ...] = ()) -> list[dict]:
        q = stems(text)
        scored = []
        for row in self.con.execute(sql):
            d = dict(row)
            body = " ".join(str(d.get(f) or "") for f in fields)
            score = overlap(q, body)
            title_body = " ".join(str(d.get(f) or "") for f in boost)
            if title_body:
                score += 0.6 * overlap(q, title_body)
            for f in key_fields:
                score += 0.8 * overlap(q, str(d.get(f) or ""))
            if score > 0:
                score -= negation_penalty(text, body)
            if score > 0:
                d["_score"] = round(score, 3)
                scored.append(d)
        scored.sort(key=lambda d: -d["_score"])
        return scored[:k]

    def thresholds_for(self, text: str, k: int = 4) -> list[dict]:
        rows = self._rank_rows(
            "SELECT id, area, metric, op, value_json, for_s, window_s, effect, source, status, kind "
            "FROM thresholds", text, ("id", "area", "metric", "effect", "source"), k,
            boost=("effect", "id"), key_fields=("id", "metric"),
        )
        for d in rows:
            d["value"] = _jload(d.pop("value_json"), None)
        return rows

    def interlocks_for(self, text: str, k: int = 3) -> list[dict]:
        rows = self._rank_rows(
            "SELECT id, title, condition, threshold_id, affected_json, effect, clear_rule, source "
            "FROM interlocks", text,
            ("id", "title", "condition", "effect", "clear_rule"), k,
            boost=("title", "id"), key_fields=("id", "title"),
        )
        for d in rows:
            d["affected"] = _jload(d.pop("affected_json"), [])
        return rows

    def faults_for(self, text: str, k: int = 3) -> list[dict]:
        rows = self._rank_rows(
            "SELECT id, category, title, observability, what_happens, how_to_notice, "
            "human_action, agent_action, must_not, status FROM faults",
            text,
            ("id", "category", "title", "what_happens", "how_to_notice", "must_not"),
            k,
            boost=("title", "id"), key_fields=("id", "title"),
        )
        # явный номер отказа имеет приоритет
        m = re.search(r"\bF-?(\d+)\.(\d+)\b", text.upper())
        if m:
            fid = f"F-{m.group(1)}.{m.group(2)}"
            row = self.con.execute("SELECT * FROM faults WHERE id = ?", (fid,)).fetchone()
            if row:
                d = dict(row)
                d["_score"] = 1.0
                rows = [d] + [r for r in rows if r["id"] != fid]
        return rows[:k]

    def diagnostics_for(self, text: str, k: int = 2) -> list[dict]:
        checks = {}
        hyps = {}
        for r in self.con.execute("SELECT diag_id, check_text FROM diagnostic_checks ORDER BY diag_id, n"):
            checks.setdefault(r["diag_id"], []).append(r["check_text"])
        for r in self.con.execute("SELECT diag_id, text FROM diagnostic_hypotheses ORDER BY diag_id, n"):
            hyps.setdefault(r["diag_id"], []).append(r["text"])
        rows = self._rank_rows(
            "SELECT id, title, system, observation, next_steps, source, status FROM diagnostics",
            text, ("id", "title", "observation", "next_steps", "system"), k,
            boost=("title",), key_fields=("id", "title"),
        )
        for d in rows:
            d["checks"] = checks.get(d["id"], [])
            d["hypotheses"] = hyps.get(d["id"], [])
        return rows

    def procedures_for(self, text: str, k: int = 2) -> list[dict]:
        pre = {}
        steps = {}
        forb = {}
        for r in self.con.execute("SELECT proc_id, text FROM procedure_preconditions ORDER BY proc_id, n"):
            pre.setdefault(r["proc_id"], []).append(r["text"])
        for r in self.con.execute("SELECT proc_id, text FROM procedure_steps ORDER BY proc_id, n"):
            steps.setdefault(r["proc_id"], []).append(r["text"])
        for r in self.con.execute("SELECT proc_id, text FROM procedure_forbidden ORDER BY proc_id, n"):
            forb.setdefault(r["proc_id"], []).append(r["text"])
        rows = self._rank_rows(
            "SELECT id, title, system, source, status FROM procedures",
            text, ("id", "title", "system"), k, boost=("title",), key_fields=("id", "title"),
        )
        for d in rows:
            d["preconditions"] = pre.get(d["id"], [])
            d["steps"] = steps.get(d["id"], [])
            d["forbidden"] = forb.get(d["id"], [])
        return rows

    def policies_for(self, text: str, k: int = 4) -> list[dict]:
        rows = self._rank_rows(
            "SELECT id, scope, rule, verdict, applies_to_json, action, source FROM policies",
            text, ("id", "scope", "rule", "action", "applies_to_json"), k,
            boost=("rule",), key_fields=("id", "rule"),
        )
        for d in rows:
            d["applies_to"] = _jload(d.pop("applies_to_json"), [])
            d["match"] = round(d.pop("_score", 0.0), 3)
        return rows

    def gaps_for(self, text: str | None = None, k: int = 3) -> list[dict]:
        rows = self._rank_rows(
            "SELECT id, area, description, conflict, blocks_json, resolution, source FROM gaps",
            text or "", ("id", "area", "description", "conflict", "blocks_json"), k,
            boost=("description", "id"), key_fields=("id", "description"),
        )
        for d in rows:
            d["blocks"] = _jload(d.pop("blocks_json"), [])
            d["match"] = round(d.pop("_score", 0.0), 3)
        return rows

    def all_gaps(self) -> list[dict]:
        rows = []
        for r in self.con.execute("SELECT * FROM gaps ORDER BY id"):
            d = dict(r)
            d["blocks"] = _jload(d.pop("blocks_json"), [])
            rows.append(d)
        return rows

    def glossary_for(self, text: str, k: int = 2) -> list[dict]:
        return self._rank_rows(
            "SELECT term, definition, source FROM glossary", text, ("term", "definition"), k,
            boost=("term",), key_fields=("term",),
        )

    def stats(self) -> dict:
        tables = ["sources", "nodes", "node_params", "node_controls", "thresholds",
                  "interlocks", "faults", "diagnostics", "procedures", "policies",
                  "gaps", "edges", "faq", "glossary", "equipment", "chunks", "cards"]
        out = {t: self.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
        out["chunk_types"] = {
            r["type"]: r["n"] for r in self.con.execute(
                "SELECT type, COUNT(*) AS n FROM chunks GROUP BY type ORDER BY n DESC"
            )
        }
        out["duplicates"] = self.con.execute(
            "SELECT COUNT(*) FROM chunks WHERE is_duplicate_of IS NOT NULL AND is_duplicate_of != ''"
        ).fetchone()[0]
        out["vectors"] = len(self._vectors["vectors"]) if self._vectors else 0
        return out

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "KB":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Best effort: tests and CLI helpers occasionally drop a KB without
        # calling close(); do not let that leak a sqlite connection.
        connection = getattr(self, "con", None)
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
