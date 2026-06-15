import json
from collections import Counter

import pandas as pd

from engine.config import OUTPUT_DIR
from engine.domain_pack import DomainPack
from engine.schemas import TestCase, AssistantAnswer, RetrievedChunk
from engine.retrieval_metrics import recall_at_k, mrr


def load_records(label: str = "baseline") -> list[dict]:
    path = OUTPUT_DIR / f"audit_{label}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def failure_type_summary(records: list[dict]) -> pd.DataFrame:
    """Which failure types occur, how often, and under which conditions."""
    counter: Counter = Counter()
    for r in records:
        if r["evaluation"]["passed"]:
            continue
        for ft in r["evaluation"]["failure_types"]:
            counter[(r["condition"], ft)] += 1
    rows = [{"condition": c, "failure_type": ft, "count": n}
            for (c, ft), n in counter.most_common()]
    return pd.DataFrame(rows)


def attribute_failures(records: list[dict], pack: DomainPack, retriever,
                       assistant, evaluator) -> pd.DataFrame:
    """Oracle-retrieval re-run: was the failure retrieval's fault or the generator's?

    For every failed case, answer again with the GOLD chunks forced into
    context. If the case now passes -> retrieval failure. If it still
    fails with perfect evidence -> generation failure.
    """
    rows = []
    for r in records:
        if r["evaluation"]["passed"]:
            continue

        case = TestCase(
            case_id=r["case_id"], user_id=r["user_id"], split=r["split"],
            condition=r["condition"], question=r["question"],
            profile=r["profile"], visible=r["visible"],
            ground_truth=r["ground_truth"],
        )
        gold = retriever.get_chunks_by_ids(case.ground_truth["gold_chunk_ids"])

        # Re-answer with oracle context. We monkey-patch retrieval for this
        # one call by answering on a question that the assistant retrieves
        # for normally, then swapping in the gold chunks for evaluation,
        # and ALSO let assistants that accept forced context use it.
        if hasattr(assistant, "answer_with_context"):
            oracle_answer = assistant.answer_with_context(case.question, gold)
        else:
            raw = assistant.answer(case.question)
            oracle_answer = AssistantAnswer(answer_text=raw.answer_text,
                                            retrieved_chunks=gold)

        rec = recall_at_k(gold, case.ground_truth["gold_chunk_ids"])
        rr = mrr(gold, case.ground_truth["gold_chunk_ids"])
        oracle_eval = evaluator.evaluate(case, oracle_answer, rec, rr)

        rows.append({
            "case_id": case.case_id,
            "condition": case.condition,
            "split": case.split,
            "original_score": r["evaluation"]["overall_score"],
            "oracle_score": oracle_eval.overall_score,
            "verdict": "retrieval_failure" if oracle_eval.passed else "generation_failure",
        })
    return pd.DataFrame(rows)
