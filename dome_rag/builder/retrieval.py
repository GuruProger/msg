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
from .paths import BUILD, DATA, DB_PATH, VECTORS_PATH

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


def stems(text: str) -> set[str]:
    """Грубые основы: первые 5 символов слова. Дёшево и переживает падежи."""
    return {t[:5] for t in vec.tokenize(text) if len(t) >= 3}


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
    def __init__(self, db_path: Path | None = None, vectors_path: Path | None = None):
        self.db_path = Path(db_path or DB_PATH)
        if not self.db_path.exists():
            raise SystemExit(
                f"База не собрана: {self.db_path}\nВыполните: py kb.py build"
            )
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self._vectors = _load_json(Path(vectors_path or VECTORS_PATH), None)
        nodes = [dict(r) for r in self.con.execute(
            "SELECT node_id, code_name, title, aliases_json FROM nodes"
        )]
        aliases = _load_json(DATA / "curated" / "aliases.json", {})
        self.alias_index = AliasIndex(aliases, nodes)
        self.node_rows = nodes

    # ------------------------------------------------------------------ поиск

    def _fts_query(self, text: str) -> str:
        toks = vec.tokenize(text)
        if not toks:
            return ""
        terms = []
        for t in dict.fromkeys(toks):
            # префиксный поиск только для достаточно длинных слов
            terms.append(f'"{t}"*' if len(t) >= 3 else f'"{t}"')
        return " OR ".join(terms)

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
                  "gaps", "edges", "faq", "glossary", "equipment", "chunks"]
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
