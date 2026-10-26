"""Пути проекта DomeKB.

Все модули работают с абсолютными путями, вычисленными от расположения пакета,
поэтому базу можно пересобрать из любого рабочего каталога.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # dome_kb/
PROJECT = ROOT.parent                              # Dome_RAG-main/

KNOWLEDGE = PROJECT / "knowledge"
LOGS = PROJECT / "Logs"

# Документы верхнего уровня: протокол, реестр узлов, регламент соревнования.
ROOT_DOCS = [
    PROJECT / "00_Краткая_справка_API.md",
    PROJECT / "01_API.md",
    PROJECT / "dome_sandbox_nodes.md",
    PROJECT / "02_Задание_и_оценка.md",
    PROJECT / "03_Требования_к_агенту.md",
    PROJECT / "04_Как_создаётся_агент.md",
]

DATA = ROOT / "data"
CHUNKS = ROOT / "chunks"
BUILD = ROOT / "build"
REPORTS = BUILD / "reports"
EVAL = ROOT / "eval"

DB_PATH = BUILD / "kb.sqlite"
VECTORS_PATH = BUILD / "vectors.json"


def knowledge_docs():
    """Все markdown-источники корпуса, в стабильном порядке."""
    docs = sorted(KNOWLEDGE.glob("*.md"))
    for p in ROOT_DOCS:
        if p.exists():
            docs.append(p)
    return [p for p in docs if p.exists()]


def rel(path: Path) -> str:
    """Короткое имя документа для метаданных и отчётов."""
    try:
        return path.relative_to(PROJECT).as_posix()
    except ValueError:
        return path.name
