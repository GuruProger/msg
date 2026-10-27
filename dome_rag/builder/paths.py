"""Пути проекта DomeKB и поиск корпуса.

Раньше пути были жёстко привязаны к ``PROJECT/knowledge`` и ``PROJECT/Logs``.
Это ломалось при любом другом расположении файлов (G13). Теперь источник
корпуса определяется так:

1. ``corpus_manifest.json`` рядом с пакетом (если существует);
2. переменные окружения ``DOMEKB_KNOWLEDGE``, ``DOMEKB_LOGS``, ``DOMEKB_DOCS``;
3. автоматический поиск типовых расположений внутри проекта.

Пустой корпус — это ошибка, а не «успешная сборка пустой базы».
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

BUILDER_VERSION = "2.0.0"

ROOT = Path(__file__).resolve().parent.parent      # dome_rag/
PROJECT = ROOT.parent                              # msg-main/

DATA = ROOT / "data"
CHUNKS = ROOT / "chunks"
BUILD = ROOT / "build"
REPORTS = BUILD / "reports"
EVAL = ROOT / "eval"

DB_PATH = BUILD / "kb.sqlite"
VECTORS_PATH = BUILD / "vectors.json"
CARDS_PATH = DATA / "cards.json"
CARD_VECTORS_PATH = BUILD / "card_vectors.json"
MANIFEST_PATH = ROOT / "corpus_manifest.json"

ROOT_DOC_NAMES = (
    "00_Краткая_справка_API.md",
    "01_API.md",
    "dome_sandbox_nodes.md",
    "02_Задание_и_оценка.md",
    "03_Требования_к_агенту.md",
    "04_Как_создаётся_агент.md",
)

# Авторитет источника. Больше — важнее при равной релевантности.
AUTHORITY = {
    "api": 5,
    "req": 5,
    "regulation": 4,
    "reference": 4,
    "task": 4,
    "theory": 3,
    "scenario": 3,
    "registry": 4,
    "history": 2,
    "prep": 2,
    "overview": 2,
}

DOCX_AUTHORITY = {
    "задание и техкарта": ("task", 5),
    "итог проект пленок": ("theory", 4),
    "легенда реактор": ("scenario", 3),
    "материалы для подготовки": ("prep", 2),
    "окрашивание статья": ("history", 2),
}

_MD_AUTHORITY = {
    "00_Краткая_справка_API.md": ("api", 5),
    "01_API.md": ("api", 5),
    "dome_sandbox_nodes.md": ("registry", 4),
    "03_Требования_к_агенту.md": ("req", 5),
    "02_Задание_и_оценка.md": ("task", 4),
    "04_Как_создаётся_агент.md": ("overview", 2),
    "07_rules.md": ("regulation", 5),
    "08_safety.md": ("regulation", 5),
    "12_history.md": ("history", 2),
}


@dataclass(frozen=True, slots=True)
class Source:
    """Один документ корпуса."""

    path: Path
    kind: str                 # md | docx | pdf
    authority: str = "reference"
    priority: int = 4
    status: str = "ok"        # ok | image_only | not_extracted

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def authority_rank(self) -> int:
        return AUTHORITY.get(self.authority, 3)


@dataclass(slots=True)
class Corpus:
    knowledge_dir: Path | None = None
    logs_dir: Path | None = None
    root_docs: list[Path] = field(default_factory=list)
    origin: str = "discovery"
    warning: str = ""

    @property
    def ok(self) -> bool:
        return bool((self.knowledge_dir and _has_knowledge(self.knowledge_dir)) or any(p.is_file() for p in self.root_docs))


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


def _has_knowledge(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() and p.suffix.lower() in {".md", ".docx"} for p in path.iterdir())


def _has_logs(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*/telemetry.jsonl"))


def _candidate_dirs(name: str) -> list[Path]:
    return [
        PROJECT / name,
        PROJECT / "Dome_RAG-main" / name,
        PROJECT.parent / name,
        ROOT / name,
    ]


def _first(predicate, candidates: list[Path]) -> Path | None:
    for path in candidates:
        try:
            if predicate(path):
                return path.resolve()
        except OSError:
            continue
    return None


def _root_docs_in(base: Path) -> list[Path]:
    return [base / name for name in ROOT_DOC_NAMES if (base / name).exists()]


def authority_for(path: Path) -> tuple[str, int]:
    name = path.name
    if name in _MD_AUTHORITY:
        return _MD_AUTHORITY[name]
    if path.suffix.casefold() == ".docx":
        low = path.stem.casefold()
        for key, value in DOCX_AUTHORITY.items():
            if key in low:
                return value
        return ("reference", 3)
    if path.suffix.casefold() == ".pdf":
        return ("regulation", 5)
    if path.parent.name == "knowledge":
        return ("reference", 4)
    return ("overview", 2)


def _resolve_manifest_path(raw: str, manifest_path: Path) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    for base in (manifest_path.parent, PROJECT, ROOT):
        probe = base / candidate
        if probe.exists():
            return probe.resolve()
    return (manifest_path.parent / candidate).resolve()


def read_manifest(path: Path) -> Corpus:
    data = json.loads(path.read_text(encoding="utf-8"))
    knowledge = data.get("knowledge_dir")
    logs = data.get("logs_dir")
    root_docs = data.get("root_docs") or []
    corpus = Corpus(
        knowledge_dir=_resolve_manifest_path(knowledge, path) if knowledge else None,
        logs_dir=_resolve_manifest_path(logs, path) if logs else None,
        root_docs=[_resolve_manifest_path(item, path) for item in root_docs],
        origin=f"manifest:{path}",
    )
    if not corpus.ok:
        corpus.warning = f"манифест {path} не содержит корпуса"
    return corpus


def discover() -> Corpus:
    manifest = _env_path("DOMEKB_MANIFEST") or MANIFEST_PATH
    if manifest.exists():
        try:
            corpus = read_manifest(manifest)
            if corpus.ok:
                # Переменные окружения важнее манифеста: иначе собранный один
                # раз манифест «прибивал» бы базу к одному набору журналов.
                env_knowledge = _env_path("DOMEKB_KNOWLEDGE")
                env_logs = _env_path("DOMEKB_LOGS")
                if env_knowledge and _has_knowledge(env_knowledge):
                    corpus.knowledge_dir = env_knowledge.resolve()
                if env_logs and _has_logs(env_logs):
                    corpus.logs_dir = env_logs.resolve()
                if corpus.logs_dir is not None and not _has_logs(corpus.logs_dir):
                    corpus.logs_dir = _first(_has_logs, _candidate_dirs("Logs"))
                if corpus.logs_dir is None:
                    corpus.logs_dir = _first(_has_logs, _candidate_dirs("Logs"))
                return corpus
        except (OSError, ValueError, KeyError) as exc:
            corpus = Corpus(origin=f"manifest-error:{manifest}")
            corpus.warning = f"не удалось прочитать манифест: {exc}"
            # не сдаёмся: пробуем найти корпус автоматически
        else:
            pass
    else:
        corpus = Corpus()
        corpus.warning = "манифест не найден"

    env_knowledge = _env_path("DOMEKB_KNOWLEDGE")
    env_logs = _env_path("DOMEKB_LOGS")
    knowledge = env_knowledge if env_knowledge and _has_knowledge(env_knowledge) else None
    if knowledge is None:
        knowledge = _first(_has_knowledge, _candidate_dirs("knowledge"))
    logs = env_logs if env_logs and _has_logs(env_logs) else None
    if logs is None:
        logs = _first(_has_logs, _candidate_dirs("Logs"))

    docs: list[Path] = []
    for base in (PROJECT, PROJECT / "Dome_RAG-main", ROOT):
        docs = _root_docs_in(base)
        if docs:
            break

    found = Corpus(
        knowledge_dir=knowledge,
        logs_dir=logs,
        root_docs=docs,
        origin="env" if (env_knowledge or env_logs) else "discovery",
        warning=corpus.warning,
    )
    if not found.ok:
        found.warning = (
            "корпус не найден: нет ни knowledge/*.md, ни корневых документов. "
            "Задайте DOMEKB_KNOWLEDGE или создайте corpus_manifest.json"
        )
    return found


SOURCES = discover()
KNOWLEDGE = SOURCES.knowledge_dir or (PROJECT / "knowledge")
LOGS = SOURCES.logs_dir or (PROJECT / "Logs")
ROOT_DOCS = SOURCES.root_docs


def sources_exist(corpus: Corpus) -> bool:
    return (not corpus.knowledge_dir or _has_knowledge(corpus.knowledge_dir)) and all(p.is_file() for p in corpus.root_docs)


def require_corpus() -> Corpus:
    """Вернуть корпус или упасть с понятным сообщением."""
    if not SOURCES.ok:
        raise SystemExit(f"Корпус не найден: {SOURCES.warning}")
    if not sources_exist(SOURCES):
        raise SystemExit("Корпус неполный: отсутствуют файлы из манифеста. Действующая база сохранена.")
    return SOURCES


def sources() -> list[Source]:
    """Все документы корпуса в стабильном порядке."""
    corpus = require_corpus()
    out: list[Source] = []
    if corpus.knowledge_dir:
        for path in sorted(corpus.knowledge_dir.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.casefold()
            if suffix not in {".md", ".docx", ".pdf"}:
                continue
            authority, priority = authority_for(path)
            out.append(Source(path=path, kind=suffix.lstrip("."), authority=authority, priority=priority))
    for path in corpus.root_docs:
        if not path.exists():
            continue
        authority, priority = authority_for(path)
        out.append(Source(path=path, kind=path.suffix.lstrip("."), authority=authority, priority=priority))
    return out


def knowledge_docs() -> list[Path]:
    """Markdown-документы корпуса (для обратной совместимости)."""
    return [source.path for source in sources() if source.kind == "md"]


def all_docs() -> list[Source]:
    return sources()


def rel(path: Path) -> str:
    """Стабильное имя документа для метаданных и отчётов."""
    path = Path(path)
    corpus = SOURCES
    if corpus.knowledge_dir:
        try:
            return f"knowledge/{path.relative_to(corpus.knowledge_dir).as_posix()}"
        except ValueError:
            pass
    for doc in corpus.root_docs:
        if doc == path:
            return path.name
    try:
        return path.relative_to(PROJECT).as_posix()
    except ValueError:
        return path.name
