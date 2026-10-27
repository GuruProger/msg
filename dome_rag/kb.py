#!/usr/bin/env python
"""DomeKB — командная строка базы знаний купола.

Примеры:

    py kb.py build
    py kb.py search "почему принтер встал на паузу"
    py kb.py ask "что делать при IL_CO2"
    py kb.py ask "какая линия питает принтер" --json
    py kb.py eval
    py kb.py drift
    py kb.py stats
    py kb.py live snapshot --host 192.168.8.10 --key <ключ>
"""
from __future__ import annotations

import argparse
import json
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

if __package__ in {None, ""}:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dome_rag.builder import cards as cards_mod
from dome_rag.builder import chunker, db, episodes, ingest, manifest, report, vectors
from dome_rag.builder.answer import answer
from dome_rag.builder.evaluator import cmd_eval
from dome_rag.builder.retrieval import KB


def _ms(t0: float) -> str:
    return f"{time.perf_counter() - t0:.2f}s"


def cmd_build(args) -> None:
    from dome_rag.builder.paths import require_corpus, LOGS, CHUNKS
    corpus = require_corpus()
    old_episodes = CHUNKS / "episodes.jsonl"
    if old_episodes.exists() and old_episodes.stat().st_size and not any(LOGS.glob("*/events.jsonl")):
        raise SystemExit("Нет исходных журналов для эпизодической памяти. Действующая база сохранена.")
    t0 = time.perf_counter()
    print("== ingest")
    ingest.cmd_ingest()
    print(f"   {_ms(t0)}")

    t = time.perf_counter()
    print("== chunk")
    chunker.cmd_chunk()
    episodes.cmd_episodes()
    print(f"   {_ms(t)}")

    t = time.perf_counter()
    print("== cards")
    cards_mod.cmd_cards()
    print(f"   {_ms(t)}")

    t = time.perf_counter()
    print("== vectors")
    vectors.cmd_vectors()
    print(f"   {_ms(t)}")

    t = time.perf_counter()
    print("== sqlite")
    path = db.build_db()
    print(f"   {path}  {_ms(t)}")

    t = time.perf_counter()
    print("== manifest")
    manifest.cmd_manifest(write=True)
    print(f"   {_ms(t)}")

    t = time.perf_counter()
    print("== report")
    report.cmd_report()
    print(f"   {_ms(t)}")

    print(f"== готово за {_ms(t0)}")
    _print_stats(KB())


def _print_stats(kb: KB) -> None:
    s = kb.stats()
    print("")
    print("Таблицы:")
    for key, value in s.items():
        if key in ("chunk_types", "duplicates", "vectors", "cards"):
            continue
        print(f"  {key:<14} {value}")
    print(f"  {'cards':<14} {s.get('cards', 0)}")
    print(f"  {'vectors':<14} {s['vectors']}")
    print(f"  {'duplicates':<14} {s['duplicates']}")
    print("Типы чанков:")
    for t, n in s["chunk_types"].items():
        print(f"  {t:<14} {n}")


def cmd_search(args) -> None:
    kb = KB()
    collections = tuple(args.collection) if args.collection else None
    hits = kb.search(
        args.query,
        k=args.k,
        collections=collections,
        types=args.type,
        doc=args.doc,
    )
    if not hits:
        print("Ничего не найдено.")
        return
    for i, h in enumerate(hits, 1):
        canon = f" [{h['canonical_id']}]" if h.get("canonical_id") else ""
        flags = []
        if h["has_gaps"]:
            flags.append("есть пробелы")
        if h["verification_status"] != "normative":
            flags.append(h["verification_status"])
        flag = ("  ⚠ " + ", ".join(flags)) if flags else ""
        print(f"{i}. {h['score']:.3f}  {h['doc']} :: {h['title']}{canon}{flag}")
        snippet = " ".join(h["text"].split())[:260]
        print(f"   {snippet}")
        print(f"   типы: {h['type']}; сущности: {', '.join(h['entities'][:6]) or '—'}")
    kb.close()


def cmd_ask(args) -> None:
    kb = KB()
    result = answer(args.query, kb, use_llm=args.llm, k=args.k)
    kb.close()
    if args.json or args.explain:
        payload = result if args.explain else result
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return
    print(result["answer"])
    print("")
    print(f"Режим: {result['mode']}; вердикт: {result['verdict']}; "
          f"уверенность: {result['confidence']['level']} "
          f"({result['confidence']['score']:.2f}); тип: {result['intent']}")


def cmd_cards(args) -> None:
    cards_mod.cmd_cards()


def cmd_manifest(args) -> None:
    if args.check:
        manifest.cmd_doctor()
    else:
        manifest.cmd_manifest(write=True)


def cmd_doctor(args) -> None:
    manifest.cmd_doctor()


def cmd_stats(args) -> None:
    kb = KB()
    _print_stats(kb)
    kb.close()


def cmd_drift(args) -> None:
    report.cmd_report()
    path = report.REPORTS / "drift.md"
    print(f"Открыть: {path}")


def cmd_live(args) -> None:
    from dome_rag.builder.live import DomeClient, WSError

    client = DomeClient(host=args.host, port=args.port, key=args.key)
    try:
        auth = client.connect()
        print(f"Авторизован: role={auth.get('role')} team={auth.get('team')}")
        if args.action == "snapshot":
            data = client.snapshot(wait_s=args.wait)
            nodes = data.get("nodes", {})
            if args.node:
                print(json.dumps(nodes.get(args.node, {}), ensure_ascii=False, indent=2))
            else:
                print(f"Узлов в снимке: {len(nodes)}")
                for nid in sorted(nodes)[:40]:
                    node = nodes[nid] or {}
                    print(f"  {nid:<24} status={node.get('status')} online={node.get('online')}")
        elif args.action == "describe":
            print(json.dumps(client.describe_nodes(), ensure_ascii=False, indent=2))
        elif args.action == "query":
            params = json.loads(args.params) if args.params else {}
            print(json.dumps(client.query(args.node, args.query, params), ensure_ascii=False, indent=2))
        elif args.action == "control":
            value = json.loads(args.value) if args.value else None
            print(json.dumps(client.control(args.node, args.command, value), ensure_ascii=False, indent=2))
        elif args.action == "say":
            print(json.dumps(client.say(args.text), ensure_ascii=False, indent=2))
    except (WSError, OSError) as exc:
        print(f"Живой API недоступен: {exc}")
        print("Проверьте DOME_HOST, DOME_KEY и сеть до купола.")
        raise SystemExit(2)
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kb.py", description="DomeKB — база знаний купола")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="полная пересборка: ingest → chunk → vectors → sqlite → report")
    sub.add_parser("ingest", help="извлечь структурированные данные из корпуса")
    sub.add_parser("chunk", help="нарезать корпус и журналы на чанки")
    sp = sub.add_parser("cards", help="пересобрать карточки знаний")
    sp.set_defaults(func=cmd_cards)

    sp = sub.add_parser("manifest", help="манифест корпуса: хеши и статусы источников")
    sp.add_argument("--check", action="store_true", help="только сравнить с текущим корпусом")
    sp.set_defaults(func=cmd_manifest)

    sp = sub.add_parser("doctor", help="проверить, не устарела ли база после правок корпуса")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("vectors", help="пересчитать TF-IDF векторы")
    sp.set_defaults(func=lambda a: vectors.cmd_vectors())

    sp = sub.add_parser("search", help="гибридный поиск по чанкам")
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=8)
    sp.add_argument("--type", action="append", help="фильтр по типу чанка (можно несколько)")
    sp.add_argument("--doc", help="фильтр по документу")
    sp.add_argument("--collection", action="append", choices=["knowledge", "history", "episodes"])
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("ask", help="ответ на вопрос с фактами, гипотезами и цитатами")
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=8)
    sp.add_argument("--llm", action="store_true", help="пересказ через DeepSeek (нужен DEEPSEEK_API_KEY)")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--explain", action="store_true", help="показать вердикт, слоты и карточки")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("stats", help="размеры таблиц и чанков")
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("drift", help="отчёт о расхождениях корпуса")
    sp.set_defaults(func=cmd_drift)

    sp = sub.add_parser("eval", help="оценка на eval/questions.json")
    sp.add_argument("--path")
    sp.add_argument("--k", type=int, default=5)
    sp.set_defaults(func=lambda a: cmd_eval(a.path, a.k))

    sp = sub.add_parser("live", help="живой API купола (WebSocket)")
    sp.add_argument("action", choices=["snapshot", "describe", "query", "control", "say"])
    sp.add_argument("--host")
    sp.add_argument("--port", type=int)
    sp.add_argument("--key")
    sp.add_argument("--wait", type=float, default=10.0)
    sp.add_argument("--node")
    sp.add_argument("--query")
    sp.add_argument("--params")
    sp.add_argument("--command")
    sp.add_argument("--value")
    sp.add_argument("--text")
    sp.set_defaults(func=cmd_live)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "ingest":
        ingest.cmd_ingest()
    elif args.cmd == "chunk":
        chunker.cmd_chunk()
        episodes.cmd_episodes()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
