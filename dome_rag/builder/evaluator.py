"""Оценка качества: честные метрики вместо «самоисполняющегося» answer_ok.

Что изменено по сравнению с v1:

* ``answer_ok`` больше не подмешивает тексты найденных фрагментов в проверяемый
  ответ. Иначе достаточно найти абзац со словом «температура», чтобы метрика
  стала 1.0, даже если ответ неверный.
* добавлены ``Recall@5`` и ``nDCG@5``;
* отказы считаются как precision/recall, а не одной «долей отказов»;
* отдельно считается ``false_abstention`` — цена слишком осторожного поиска;
* измеряется p95 задержки.

Набор вопросов может содержать ``negative: true`` (вопрос вне корпуса) либо
пустой ``expect_any``. Для положительных вопросов ``expect_any`` — подстроки,
которые обязаны появиться в ответе (факты/гипотезы/действия/цитаты), а не в
сырых чанках.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .answer import answer
from .coverage import ANSWERABLE, NO_RELEVANT_EVIDENCE, _attribute_match, extract_slots
from .paths import EVAL
from .retrieval import KB


def _card_haystack(card: dict) -> str:
    return " ".join(
        str(part or "")
        for part in (
            card.get("canonical_id"),
            card.get("subject_label"),
            card.get("subject"),
            card.get("value"),
            card.get("text"),
            (card.get("source") or {}).get("doc"),
            (card.get("source") or {}).get("title"),
        )
    ).lower()


def _answer_haystack(result: dict) -> str:
    """Только то, что видит пользователь как ответ.

    Цитаты сюда не входят: это ссылки на источники, а не утверждения. Если
    считать их текстом ответа, метрика снова начинает «подтверждаться» самим
    фактом находки фрагмента.
    """
    return " ".join(
        result.get("facts", [])
        + result.get("hypotheses", [])
        + result.get("actions", [])
        + result.get("unknown", [])
    ).lower()


def _match(expect: list[str], text: str) -> bool:
    low = text.lower()
    return any(str(e).lower() in low for e in expect)


def _ndcg(rank: int, k: int = 5) -> float:
    if not rank or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def evaluate(kb: KB, questions: list[dict], k: int = 5) -> dict:
    rows = []
    reciprocal = 0.0
    ndcg = 0.0
    positives = negatives = 0
    hit1 = hitk = answer_ok = strict_ok = 0
    true_abstain = false_abstain = wrong_confident = overanswer = 0
    abstain_total = 0
    citation_hits = citation_total = 0
    latencies: list[float] = []
    conf_correct: list[float] = []
    conf_wrong: list[float] = []

    for q in questions:
        question = q["question"]
        expect = q.get("expect_any") or []
        negative = bool(q.get("negative")) or not expect
        started = time.perf_counter()
        result = answer(question, kb)
        latencies.append((time.perf_counter() - started) * 1000)

        if negative:
            negatives += 1
            abstain_total += 1
            ok = result.get("verdict") == NO_RELEVANT_EVIDENCE
            true_abstain += int(ok)
            if not ok:
                overanswer += 1
                conf_wrong.append(float(result["confidence"]["score"]))
            rank = 0
        else:
            positives += 1
            slots = extract_slots(question, entities=kb.detect_entities(question))
            hits = kb.search_cards(question, k=max(k, 8), slots=slots)
            rank = 0
            gold_ids: set[str] = set()
            gold_canonicals: set[str] = set()
            for index, card in enumerate(hits, 1):
                if _match(expect, _card_haystack(card)):
                    gold_ids.add(card["card_id"])
                    if card.get("canonical_id"):
                        gold_canonicals.add(str(card["canonical_id"]).upper())
                    if rank == 0:
                        rank = index
            if rank == 1:
                hit1 += 1
            if 1 <= rank <= k:
                hitk += 1
            reciprocal += 1.0 / rank if rank else 0.0
            ndcg += _ndcg(rank, k)

            # Ответ засчитывается только по тому, что видит пользователь:
            # факты/гипотезы/действия. ``unknown`` — это ограничения, а не
            # утверждение; цитаты — ссылки. Раньше достаточно было найти
            # абзац, чтобы метрика стала 1.0.
            user_haystack = " ".join(
                result.get("facts", [])
                + result.get("hypotheses", [])
                + result.get("actions", [])
            ).lower()
            found = _match(expect, user_haystack)
            answered = result.get("verdict") != NO_RELEVANT_EVIDENCE
            ok = bool(found and answered)
            answer_ok += int(ok)
            # У вопроса может быть честно ожидаемый вердикт: например, когда
            # сам корпус документирует противоречие источников. Тогда строгая
            # проверка требует именно CONFLICT, а не ANSWERABLE.
            expected_verdict = q.get("expect_verdict") or ANSWERABLE
            strict_ok += int(found and result.get("verdict") == expected_verdict)
            if result.get("verdict") == NO_RELEVANT_EVIDENCE:
                false_abstain += 1
            if result.get("verdict") == ANSWERABLE and not found:
                wrong_confident += 1
            (conf_correct if found else conf_wrong).append(
                float(result["confidence"]["score"])
            )

            # Точность цитат: цитата релевантна, если её текст содержит
            # ожидаемый факт, ИЛИ она указывает на тот же источник, что и
            # найденная «золотая» карточка, ИЛИ её атрибут отвечает на
            # запрошенный аспект вопроса.
            requested_attributes = result.get("coverage", {}).get("attributes") or []
            for citation in result.get("citations", []):
                blob = " ".join(
                    str(citation.get(key) or "")
                    for key in ("source", "title", "text", "canonical_id")
                )
                canonical = str(citation.get("canonical_id") or "").upper()
                attribute = str(citation.get("attribute") or "")
                slot_relevant = any(
                    _attribute_match(wanted, attribute) >= 0.65
                    for wanted in requested_attributes
                )
                if (
                    _match(expect, blob)
                    or citation.get("card_id") in gold_ids
                    or (canonical and canonical in gold_canonicals)
                    or slot_relevant
                ):
                    citation_hits += 1
                citation_total += 1

        rows.append(
            {
                "id": q.get("id"),
                "question": question,
                "negative": negative,
                "rank": rank or None,
                "answer_ok": bool(ok),
                "strict_answer_ok": bool(
                    ok if negative
                    else (
                        result.get("verdict")
                        == (q.get("expect_verdict") or ANSWERABLE)
                        and ok
                    )
                ),
                "verdict": result.get("verdict"),
                "abstained": result.get("verdict") == NO_RELEVANT_EVIDENCE,
                "confidence": result["confidence"],
                "citations": len(result.get("citations", [])),
                "note": q.get("note", ""),
            }
        )

    pos = max(1, positives)
    neg = max(1, negatives)
    return {
        "n": len(questions),
        "answerable": positives,
        "negatives": negatives,
        "hit@1": round(hit1 / pos, 3),
        "recall@5": round(hitk / pos, 3),
        "hit@5": round(hitk / pos, 3),
        "mrr": round(reciprocal / pos, 3),
        "ndcg@5": round(ndcg / pos, 3),
        "answer_ok": round(answer_ok / pos, 3),
        "strict_answer_ok": round(strict_ok / pos, 3),
        "wrong_confident": round(wrong_confident / pos, 3),
        "wrong_confident_count": wrong_confident,
        "overanswer": round(overanswer / neg, 3),
        "overanswer_count": overanswer,
        "abstention_recall": round(true_abstain / neg, 3),
        "abstention_precision": round(true_abstain / max(1, true_abstain + false_abstain), 3),
        "false_abstention": round(false_abstain / pos, 3),
        "false_abstention_count": false_abstain,
        "citation_precision": round(citation_hits / citation_total, 3) if citation_total else None,
        "citation_total": citation_total,
        "conf_correct": round(sum(conf_correct) / len(conf_correct), 3) if conf_correct else None,
        "conf_wrong": round(sum(conf_wrong) / len(conf_wrong), 3) if conf_wrong else None,
        "p50_ms": round(_percentile(latencies, 50), 1),
        "p95_ms": round(_percentile(latencies, 95), 1),
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

    print(f"questions   {metrics['n']}  (answerable {metrics['answerable']}, "
          f"negatives {metrics['negatives']})")
    print(f"hit@1       {metrics['hit@1']:.3f}")
    print(f"recall@5    {metrics['recall@5']:.3f}")
    print(f"MRR         {metrics['mrr']:.3f}")
    print(f"nDCG@5      {metrics['ndcg@5']:.3f}")
    print(f"answer_ok   {metrics['answer_ok']:.3f}")
    print(f"strict_ok   {metrics['strict_answer_ok']:.3f}")
    print(f"wrong-conf. {metrics['wrong_confident']:.3f} "
          f"({metrics['wrong_confident_count']} вопросов)")
    print(f"overanswer  {metrics['overanswer']:.3f} "
          f"({metrics['overanswer_count']} вопросов)")
    print(f"abstention  recall {metrics['abstention_recall']:.3f}  "
          f"precision {metrics['abstention_precision']:.3f}")
    print(f"false-abst. {metrics['false_abstention']:.3f} "
          f"({metrics['false_abstention_count']} вопросов)")
    if metrics["citation_precision"] is not None:
        print(f"citation    {metrics['citation_precision']:.3f} "
              f"({metrics['citation_total']} цитат)")
    if metrics.get("conf_correct") is not None:
        print(f"confidence  верные {metrics['conf_correct']:.2f} / "
              f"неверные {metrics['conf_wrong'] or 0:.2f}")
    print(f"latency     p50 {metrics['p50_ms']:.1f} ms / p95 {metrics['p95_ms']:.1f} ms")
    if verbose:
        print("")
        for row in metrics["rows"]:
            mark = "OK " if row["answer_ok"] else "FAIL"
            kind = "NEG" if row["negative"] else f"rank={row['rank']}"
            print(
                f"{mark} {str(row['id']):<5} {kind:<9} "
                f"verdict={row['verdict']:<21} conf={row['confidence']['score']:.2f}  "
                f"{row['question'][:64]}"
            )
    return metrics
