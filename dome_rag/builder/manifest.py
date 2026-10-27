"""Манифест корпуса: что именно попало в базу и в каком оно состоянии.

Зачем это нужно:

* сборка перестаёт зависеть от «магии» текущего каталога — корпус описан
  файлом, который можно положить в git;
* видно, какие документы прочитаны полностью, какие только частично
  (``image_only``), а какие вообще не извлечены;
* SHA-256 каждого файла позволяет отличить «база устарела» от «поиск плохой»
  (команда ``kb.py doctor``).

Манифест не является политикой безопасности и не даёт прав: он только
описывает источники.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import docx
from .paths import (
    BUILDER_VERSION,
    MANIFEST_PATH,
    PROJECT,
    SOURCES,
    Corpus,
    Source,
    rel,
    require_corpus,
    sources,
)

MANIFEST_VERSION = 2
SECRET_SUFFIXES = {".key", ".pem", ".env"}
SECRET_DIRS = {"keys", "secrets", ".ssh"}


@dataclass(slots=True)
class ScannedSource:
    path: Path
    rel_path: str
    kind: str
    authority: str
    priority: int
    status: str
    sha256: str
    bytes: int
    chars: int
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "path": self.rel_path,
            "kind": self.kind,
            "authority": self.authority,
            "priority": self.priority,
            "status": self.status,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "chars": self.chars,
            "note": self.note,
        }


def _is_secret(path: Path) -> bool:
    if path.suffix.casefold() in SECRET_SUFFIXES:
        return True
    return any(part.casefold() in SECRET_DIRS for part in path.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _measure(source: Source) -> ScannedSource:
    path = source.path
    digest = _sha256(path)
    status = source.status
    chars = 0
    note = ""
    if source.kind == "docx":
        try:
            extracted = docx.extract(path)
        except ValueError as exc:
            status = "not_extracted"
            note = str(exc)[:160]
        else:
            chars = len(extracted.text)
            status = extracted.status
            note = (
                f"рисунков {len(extracted.images)}, таблиц {extracted.tables}"
                if extracted.images or extracted.tables
                else ""
            )
    elif source.kind in {"md", "txt"}:
        chars = len(path.read_text(encoding="utf-8", errors="replace"))
    elif source.kind == "pdf":
        status = "not_extracted"
        note = "PDF не индексируется без OCR"
    return ScannedSource(
        path=path,
        rel_path=rel(path),
        kind=source.kind,
        authority=source.authority,
        priority=source.priority,
        status=status,
        sha256=digest,
        bytes=path.stat().st_size,
        chars=chars,
        note=note,
    )


def corpus_hash(items: list[ScannedSource]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda i: i.rel_path):
        digest.update(item.rel_path.encode("utf-8"))
        digest.update(item.sha256.encode("ascii"))
    return digest.hexdigest()[:32]


def scan(corpus: Corpus | None = None) -> list[ScannedSource]:
    corpus = corpus or require_corpus()
    out: list[ScannedSource] = []
    for source in sources():
        if _is_secret(source.path):
            # Ключи и секреты не индексируются и не попадают в манифест.
            continue
        out.append(_measure(source))
    return out


def build_manifest(corpus: Corpus | None = None) -> dict:
    corpus = corpus or SOURCES
    items = scan(corpus)
    knowledge = corpus.knowledge_dir
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "builder_version": BUILDER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_hash": corpus_hash(items),
        "knowledge_dir": _rel_path(knowledge) if knowledge else None,
        "logs_dir": _rel_path(corpus.logs_dir) if corpus.logs_dir else None,
        "root_docs": [_rel_path(p) for p in corpus.root_docs],
        "counts": {
            "documents": len(items),
            "ok": sum(1 for i in items if i.status == "ok"),
            "image_only": sum(1 for i in items if i.status == "image_only"),
            "not_extracted": sum(1 for i in items if i.status == "not_extracted"),
            "docx": sum(1 for i in items if i.kind == "docx"),
            "md": sum(1 for i in items if i.kind == "md"),
        },
        "sources": [item.as_dict() for item in items],
    }
    return payload


def _rel_path(path: Path) -> str:
    path = Path(path).resolve()
    for base in (PROJECT, PROJECT.parent):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def write_manifest(path: Path | None = None, corpus: Corpus | None = None) -> Path:
    path = Path(path or MANIFEST_PATH)
    payload = build_manifest(corpus)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def load(path: Path | None = None) -> dict | None:
    path = Path(path or MANIFEST_PATH)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def verify(path: Path | None = None) -> dict:
    """Сравнить текущий корпус с записанным манифестом."""
    recorded = load(path)
    current = build_manifest()
    if recorded is None:
        return {
            "ok": False,
            "reason": "манифест не найден",
            "changed": [],
            "added": sorted(item["path"] for item in current["sources"]),
            "removed": [],
            "current_hash": current["corpus_hash"],
        }
    old = {item["path"]: item for item in recorded.get("sources", [])}
    new = {item["path"]: item for item in current["sources"]}
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(
        name for name in set(old) & set(new) if old[name].get("sha256") != new[name].get("sha256")
    )
    return {
        "ok": not (added or removed or changed),
        "reason": "" if not (added or removed or changed) else "корпус изменился после сборки",
        "added": added,
        "removed": removed,
        "changed": changed,
        "recorded_hash": recorded.get("corpus_hash"),
        "current_hash": current["corpus_hash"],
        "builder_version": recorded.get("builder_version"),
        "current_builder_version": BUILDER_VERSION,
    }


def cmd_manifest(write: bool = True) -> dict:
    payload = build_manifest()
    if write:
        path = write_manifest()
        print(f"manifest    {path}")
    counts = payload["counts"]
    print(f"documents   {counts['documents']}")
    print(f"  md        {counts['md']}")
    print(f"  docx      {counts['docx']}")
    print(f"  ok        {counts['ok']}")
    print(f"  image_only {counts['image_only']}")
    print(f"  not_extr. {counts['not_extracted']}")
    print(f"hash        {payload['corpus_hash']}")
    print(f"builder     {payload['builder_version']}")
    for item in payload["sources"]:
        flag = "" if item["status"] == "ok" else f"  [{item['status']}]"
        print(f"  {item['path']:<52} {item['chars']:>6} chars{flag}")
    return payload


def cmd_doctor() -> dict:
    state = verify()
    print(f"corpus      {'OK' if state['ok'] else 'DRIFT'}")
    print(f"recorded    {state.get('recorded_hash') or '—'}")
    print(f"current     {state.get('current_hash') or '—'}")
    for key in ("added", "removed", "changed"):
        values = state.get(key) or []
        if values:
            print(f"{key:<11} {len(values)}")
            for value in values[:10]:
                print(f"  {value}")
    if not state["ok"]:
        print("Действие: пересобрать базу командой `python kb.py build`.")
    return state
