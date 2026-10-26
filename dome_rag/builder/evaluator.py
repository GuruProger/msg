"""Оценка качества поиска и ответов на размеченном наборе вопросов.

Метрики намеренно простые и воспроизводимые:

* ``hit@1`` / ``hit@5`` — найден ли ожидаемый фрагмент в топе;
* ``MRR`` — средний обратный ранг первого попадания;
* ``answer_ok`` — попал ли ожидаемый маркер в текст ответа (факты/действия);
* ``abstention`` — доля корректных отказов на вопросы без ответа.
"""
from __future__ import annotations

import json
from pathlib import Path

from .answer import answer
from .retrieval import KB
from .paths import EVAL


def _haystack(chunk: dict) -> str:
    return " ".join(
        [
            chunk.get("canonical_id") or "",
            chunk.get("title") or "",
            chunk.get("doc") or "",
            chunk.get("text") or "",
        ]
    ).lower()


def _match(expect: list[str], text: str) -> bool:
    low = text.lower()
    return any(e.lower() in low for e in expect)


def evaluate(kb: KB, questions: list[dict], k: int = 5) -> dict:
    rows = []
    reciprocal = 0.0
    answerable = 0
    hit1 = hit5 = answer_ok = 0
    abst_total = abst_ok = 0

    for q in questions:
        question = q["question"]
        expect = q.get("expect_any") or []
        result = answer(question, kb)
        hits = kb.search(question, k=k)

        rank = 0
        for i, c in enumerate(hits, 1):
            if expect and _match(expect, _haystack(c)):
                rank = i
                break

        if expect:
            answerable += 1
            if rank == 1:
                hit1 += 1
            if 1 <= rank <= k:
                hit5 += 1
            reciprocal += 1.0 / rank if rank else 0.0
            answer_text = " ".join(
                result["facts"] + result["hypotheses"] + result["actions"]
                + [c.get("text", "") for c in result["chunks"]]
            )
            ok = _match(expect, answer_text)
            answer_ok += int(ok)
        else:
            abst_total += 1
            ok = bool(result.get("abstained"))
            abst_ok += int(ok)
            rank = 0

        rows.append(
            {
                "id": q.get("id"),
                "question": question,
                "rank": rank or None,
                "answer_ok": ok,
                "abstained": bool(result.get("abstained")),
                "confidence": result["confidence"],
                "note": q.get("note", ""),
            }
        )

    n = max(1, answerable)
    return {
        "n": len(questions),
        "answerable": answerable,
        "hit@1": round(hit1 / n, 3),
        "hit@5": round(hit5 / n, 3),
        "mrr": round(reciprocal / n, 3),
        "answer_ok": round(answer_ok / n, 3),
        "abstention": round(abst_ok / max(1, abst_total), 3),
        "rows": rows,
    }


def load_questions(path: Path | None = None) -> list[dict]:
    path = Path(path or (EVAL / "questions.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def cmd_eval(path: str | None = None, k: int = 5, verbose: bool = True) -> dict:
    kb = KB()
    questions = load_questions(Path(path) if path else None)
    metrics = evaluate(kb, questions, k=k)
    kb.close()

    print(f"questions   {metrics['n']}  (answerable {metrics['answerable']})")
    print(f"hit@1       {metrics['hit@1']:.3f}")
    print(f"hit@{k}       {metrics['hit@5']:.3f}")
    print(f"MRR         {metrics['mrr']:.3f}")
    print(f"answer_ok   {metrics['answer_ok']:.3f}")
    print(f"abstention  {metrics['abstention']:.3f}")
    if verbose:
        print("")
        for r in metrics["rows"]:
            mark = "OK " if r["answer_ok"] else "FAIL"
            abst = " [отказ]" if r["abstained"] else ""
            print(f"{mark} {r['id']:<4} rank={r['rank']} conf={r['confidence']['score']:.2f}{abst}  {r['question'][:70]}")
    return metrics
