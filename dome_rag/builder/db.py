"""Сборка SQLite-ядра базы знаний.

Структура БД: справочники (узлы, команды, пороги, интерлоки), кейсы
(отказы, сценарии, процедуры), политики, пробелы, связи, плоский индекс
чанков и FTS5 для лексического поиска.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .paths import BUILD, CHUNKS, DATA

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=OFF;

CREATE TABLE IF NOT EXISTS sources(
  id TEXT PRIMARY KEY, path TEXT, kind TEXT, priority INT, checksum TEXT, bytes INT);

CREATE TABLE IF NOT EXISTS nodes(
  node_id TEXT PRIMARY KEY, code_name TEXT, title TEXT, system TEXT,
  system_ru TEXT, aliases_json TEXT, source TEXT);

CREATE TABLE IF NOT EXISTS node_params(
  node_id TEXT, key TEXT, unit TEXT, kind TEXT, trust TEXT,
  enum_hint TEXT, item_fields_json TEXT, sample_json TEXT,
  PRIMARY KEY(node_id, key));

CREATE TABLE IF NOT EXISTS node_controls(
  node_id TEXT, action TEXT, label TEXT, allowed_for TEXT,
  api_confirmed INT, verify_by_json TEXT, value_schema_json TEXT,
  notes TEXT, preconditions_json TEXT,
  PRIMARY KEY(node_id, action));

CREATE TABLE IF NOT EXISTS thresholds(
  id TEXT PRIMARY KEY, area TEXT, metric TEXT, op TEXT, value_json TEXT,
  for_s REAL, window_s REAL, effect TEXT, source TEXT, status TEXT, kind TEXT);

CREATE TABLE IF NOT EXISTS interlocks(
  id TEXT PRIMARY KEY, title TEXT, condition TEXT, threshold_id TEXT,
  affected_json TEXT, effect TEXT, clear_rule TEXT, source TEXT);

CREATE TABLE IF NOT EXISTS faults(
  id TEXT PRIMARY KEY, category TEXT, title TEXT, observability TEXT,
  what_happens TEXT, how_to_notice TEXT, human_action TEXT, agent_action TEXT,
  must_not TEXT, source TEXT, status TEXT);

CREATE TABLE IF NOT EXISTS fault_signals(
  fault_id TEXT, node TEXT, param TEXT, op TEXT, value_json TEXT);

CREATE TABLE IF NOT EXISTS diagnostics(
  id TEXT PRIMARY KEY, title TEXT, system TEXT, observation TEXT,
  next_steps TEXT, source TEXT, status TEXT);

CREATE TABLE IF NOT EXISTS diagnostic_checks(diag_id TEXT, n INT, check_text TEXT);
CREATE TABLE IF NOT EXISTS diagnostic_hypotheses(diag_id TEXT, n INT, text TEXT);

CREATE TABLE IF NOT EXISTS procedures(
  id TEXT PRIMARY KEY, title TEXT, system TEXT, source TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS procedure_steps(proc_id TEXT, n INT, text TEXT);
CREATE TABLE IF NOT EXISTS procedure_forbidden(proc_id TEXT, n INT, text TEXT);
CREATE TABLE IF NOT EXISTS procedure_preconditions(proc_id TEXT, n INT, text TEXT);

CREATE TABLE IF NOT EXISTS policies(
  id TEXT PRIMARY KEY, scope TEXT, rule TEXT, verdict TEXT,
  applies_to_json TEXT, condition TEXT, action TEXT, source TEXT);

CREATE TABLE IF NOT EXISTS gaps(
  id TEXT PRIMARY KEY, area TEXT, description TEXT, conflict TEXT,
  blocks_json TEXT, resolution TEXT, source TEXT);

CREATE TABLE IF NOT EXISTS edges(
  src TEXT, dst TEXT, relation TEXT, note TEXT);

CREATE TABLE IF NOT EXISTS faq(
  id TEXT PRIMARY KEY, question TEXT, answer TEXT, section TEXT, source TEXT);

CREATE TABLE IF NOT EXISTS glossary(
  term TEXT PRIMARY KEY, definition TEXT, source TEXT);

CREATE TABLE IF NOT EXISTS equipment(
  id INTEGER PRIMARY KEY AUTOINCREMENT, section TEXT, path_json TEXT,
  columns_json TEXT, row_json TEXT, source TEXT);

CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY, collection TEXT, doc TEXT, heading_path TEXT,
  title TEXT, type TEXT, canonical_id TEXT, entities TEXT,
  verification_status TEXT, has_gaps INT, source_priority INT,
  is_duplicate_of TEXT, text TEXT, search_text TEXT, line INT);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
  USING fts5(chunk_id UNINDEXED, search_text, tokenize='unicode61');
"""


def _j(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _load_json(name: str) -> list[dict]:
    """Сгенерированные файлы лежат в ``data/``, ручные — в ``data/curated/``."""
    for p in (DATA / name, DATA / "curated" / name):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return []


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_db(db_path: Path | None = None) -> Path:
    db_path = db_path or (BUILD / "kb.sqlite")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    # --- источники
    for s in _load_json("sources.json"):
        con.execute(
            "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?)",
            (s["id"], s["path"], s["kind"], s["priority"], s["checksum"], s["bytes"]),
        )

    # --- узлы
    for n in _load_json("nodes.json"):
        con.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?)",
            (n["node_id"], n["code_name"], n["title"], n["system"],
             n.get("system_ru"), _j(n.get("aliases", [])), n.get("source")),
        )
        for p in n.get("params", []):
            con.execute(
                "INSERT OR REPLACE INTO node_params VALUES (?,?,?,?,?,?,?,?)",
                (n["node_id"], p["key"], p.get("unit"), p.get("kind"),
                 p.get("trust"), p.get("enum_hint"),
                 _j(p.get("item_fields", [])), _j(p.get("sample"))),
            )
        for c in n.get("controls", []):
            con.execute(
                "INSERT OR REPLACE INTO node_controls VALUES (?,?,?,?,?,?,?,?,?)",
                (n["node_id"], c["action"], c.get("label"), c.get("allowed_for", "agent"),
                 1 if c.get("api_confirmed") else 0, _j(c.get("verify_by", [])),
                 _j(c.get("value_schema")), c.get("notes"), _j(c.get("preconditions", []))),
            )

    # --- пороги и интерлоки
    for t in _load_json("thresholds.json"):
        con.execute(
            "INSERT OR REPLACE INTO thresholds VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (t["id"], t.get("area"), t.get("metric"), t.get("op"),
             _j(t.get("value")), t.get("for_s"), t.get("window_s"),
             t.get("effect"), t.get("source"), t.get("status"), t.get("kind")),
        )
    for i in _load_json("interlocks.json"):
        con.execute(
            "INSERT OR REPLACE INTO interlocks VALUES (?,?,?,?,?,?,?,?)",
            (i["id"], i.get("title"), i.get("condition"), i.get("threshold"),
             _j(i.get("affected", [])), i.get("effect"), i.get("clear"), i.get("source")),
        )

    # --- отказы
    for f in _load_json("faults.json"):
        con.execute(
            "INSERT OR REPLACE INTO faults VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (f["id"], f.get("category"), f["title"], f.get("observability"),
             f.get("what_happens"), f.get("how_to_notice"), f.get("human_action"),
             f.get("agent_action"), f.get("must_not"), f.get("source"), f.get("status")),
        )
        for s in f.get("signals", []):
            con.execute(
                "INSERT INTO fault_signals VALUES (?,?,?,?,?)",
                (f["id"], s.get("node"), s.get("param"), s.get("op"), _j(s.get("value"))),
            )

    # --- диагностика
    for d in _load_json("diagnostics.json"):
        con.execute(
            "INSERT OR REPLACE INTO diagnostics VALUES (?,?,?,?,?,?,?)",
            (d["id"], d["title"], d.get("system"), d.get("observation"),
             d.get("next_steps"), d.get("source"), d.get("status")),
        )
        for n, text in enumerate(d.get("checks", []), 1):
            con.execute("INSERT INTO diagnostic_checks VALUES (?,?,?)", (d["id"], n, text))
        for n, text in enumerate(d.get("hypotheses", []), 1):
            con.execute("INSERT INTO diagnostic_hypotheses VALUES (?,?,?)", (d["id"], n, text))

    # --- процедуры (сгенерированные + уточнённые вручную)
    procs = _load_json("procedures.json") + _load_json("procedures_curated.json")
    seen = set()
    for p in procs:
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        con.execute(
            "INSERT OR REPLACE INTO procedures VALUES (?,?,?,?,?)",
            (p["id"], p["title"], p.get("system"), p.get("source"), p.get("status")),
        )
        for n, text in enumerate(p.get("preconditions", []), 1):
            con.execute("INSERT INTO procedure_preconditions VALUES (?,?,?)", (p["id"], n, _j(text) if isinstance(text, dict) else text))
        for n, text in enumerate(p.get("steps", []), 1):
            con.execute("INSERT INTO procedure_steps VALUES (?,?,?)", (p["id"], n, text))
        for n, text in enumerate(p.get("forbidden", []), 1):
            con.execute("INSERT INTO procedure_forbidden VALUES (?,?,?)", (p["id"], n, text))

    # --- политики, пробелы, связи
    for p in _load_json("policies.json"):
        con.execute(
            "INSERT OR REPLACE INTO policies VALUES (?,?,?,?,?,?,?,?)",
            (p["id"], p.get("scope"), p.get("rule"), p.get("verdict"),
             _j(p.get("applies_to", [])), p.get("condition"), p.get("action"), p.get("source")),
        )
    for g in _load_json("gaps.json"):
        con.execute(
            "INSERT OR REPLACE INTO gaps VALUES (?,?,?,?,?,?,?)",
            (g["id"], g.get("area"), g["description"], g.get("conflict"),
             _j(g.get("blocks", [])), g.get("resolution"), g.get("source")),
        )
    for e in _load_json("edges.json"):
        con.execute("INSERT INTO edges VALUES (?,?,?,?)",
                    (e["src"], e["dst"], e.get("relation"), e.get("note")))

    # --- справка
    for f in _load_json("faq.json"):
        con.execute("INSERT OR REPLACE INTO faq VALUES (?,?,?,?,?)",
                    (f["id"], f["question"], f["answer"], f.get("section"), f.get("source")))
    for g in _load_json("glossary.json"):
        con.execute("INSERT OR REPLACE INTO glossary VALUES (?,?,?)",
                    (g["term"], g["definition"], g.get("source")))
    for e in _load_json("equipment.json"):
        con.execute("INSERT INTO equipment(section,path_json,columns_json,row_json,source) VALUES (?,?,?,?,?)",
                    (e["section"], _j(e["path"]), _j(e["columns"]), _j(e["row"]), e["source"]))

    # --- чанки
    chunks = _load_jsonl(CHUNKS / "knowledge.jsonl") + _load_jsonl(CHUNKS / "episodes.jsonl")
    seen_ids = set()
    for c in chunks:
        cid = c["chunk_id"]
        if cid in seen_ids:
            cid = f"{cid}--{len(seen_ids)}"
        seen_ids.add(cid)
        con.execute(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, c["collection"], c["doc"], _j(c["heading_path"]), c["title"],
             c["type"], c.get("canonical_id"), _j(c.get("entities", [])),
             c["verification_status"], 1 if c.get("has_gaps") else 0,
             c.get("source_priority", 3), c.get("is_duplicate_of"),
             c["text"], c["search_text"], c.get("line", 0)),
        )
        con.execute("INSERT INTO chunks_fts(chunk_id, search_text) VALUES (?,?)",
                    (cid, c["search_text"]))

    con.commit()
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    con.commit()
    con.close()
    return db_path
