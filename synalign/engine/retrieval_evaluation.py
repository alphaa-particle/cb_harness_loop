"""Measures of search quality, independent of the answer-generation scorer."""

import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from engine.enforcement import build_messages
from engine.schemas import RetrievedChunk


# --- per-answer measures used by the audit --------------------------------

def recall_at_k(retrieved: list[RetrievedChunk], gold_chunk_ids: list[str]) -> float:
    if not gold_chunk_ids:
        return 1.0
    got = {c.chunk_id for c in retrieved}
    return sum(1 for g in gold_chunk_ids if g in got) / len(gold_chunk_ids)


def mrr(retrieved: list[RetrievedChunk], gold_chunk_ids: list[str]) -> float:
    if not gold_chunk_ids:
        return 1.0
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.chunk_id in gold_chunk_ids:
            return 1.0 / rank
    return 0.0


# --- labelled-question evaluation -----------------------------------------

def load_questions(path: str | Path, split: str | None = None) -> list[dict]:
    cases, ids, group_splits = [], set(), {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", row.get("case_id", line_number)))
            if query_id in ids:
                raise ValueError(f"Duplicate query ID: {query_id}")
            ids.add(query_id)
            group_id = str(row.get("group_id", row.get("user_id", query_id)))
            row_split = row.get("split")
            if group_id in group_splits and group_splits[group_id] != row_split:
                raise ValueError(f"Query group {group_id!r} leaks across splits")
            group_splits[group_id] = row_split
            if split is not None and row_split != split:
                continue
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Nonempty question required for {query_id}")
            gold = row.get("gold_chunk_ids", row.get("ground_truth", {}).get("gold_chunk_ids", []))
            relevance = row.get("relevance", {cid: 1 for cid in gold})
            if not isinstance(relevance, dict) or not all(
                isinstance(cid, str) and isinstance(grade, (float, int)) and
                math.isfinite(grade) and grade >= 0 for cid, grade in relevance.items()
            ):
                raise ValueError(f"Invalid relevance judgments for {query_id}")
            relevance = {cid: float(grade) for cid, grade in relevance.items() if grade > 0}
            abstain = row.get("expect_abstain", False)
            if not isinstance(abstain, bool) or bool(relevance) == abstain:
                raise ValueError(f"{query_id} needs positive labels OR expect_abstain=true with no positive labels")
            required = row.get("required_chunk_ids", [])
            if not isinstance(required, list) or not set(required).issubset(relevance):
                raise ValueError(f"Required evidence must have positive relevance labels for {query_id}")
            cases.append({"query_id": query_id, "question": question, "relevance": relevance,
                          "expect_abstain": abstain, "required_chunk_ids": required,
                          "condition": row.get("condition", "unspecified"), "split": row.get("split"),
                          "group_id": group_id})
    if not cases:
        raise ValueError("No labelled questions selected; check the input file and split")
    return cases


def load_beir_questions(directory: str | Path, split: str = "test") -> list[dict]:
    directory = Path(directory)
    queries = {}
    with (directory / "queries.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["_id"] in queries:
                raise ValueError(f"Duplicate query ID: {row['_id']}")
            queries[row["_id"]] = row["text"]
    judgments = {}
    with (directory / "qrels" / f"{split}.tsv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if float(row["score"]) > 0:
                judgments.setdefault(row["query-id"], {})[row["corpus-id"]] = float(row["score"])
    if not judgments:
        raise ValueError("No positive relevance judgments in the requested split")
    return [{"query_id": qid, "question": queries[qid], "relevance": rel,
             "expect_abstain": False, "required_chunk_ids": [], "condition": "independent_benchmark",
             "split": split, "group_id": qid} for qid, rel in sorted(judgments.items())]


def ranked_metrics(ids: list[str], relevance: dict[str, float], k: int) -> dict:
    if not relevance:
        raise ValueError("Ranking metrics require positive judgments; score abstention separately")
    if k <= 0 or len(ids) != len(set(ids)):
        raise ValueError("k must be positive and retrieved IDs must be unique")
    top = ids[:k]
    hits = sum(cid in relevance for cid in top)
    reciprocal_rank = next((1 / rank for rank, cid in enumerate(top, 1) if cid in relevance), 0.0)
    dcg = sum((2 ** relevance.get(cid, 0) - 1) / math.log2(rank + 1)
              for rank, cid in enumerate(top, 1))
    ideal = sum((2 ** grade - 1) / math.log2(rank + 1)
                for rank, grade in enumerate(sorted(relevance.values(), reverse=True)[:k], 1))
    return {"hit_rate": float(hits > 0), "recall": hits / len(relevance),
            "precision": hits / k, "mrr": reciprocal_rank, "ndcg": dcg / ideal}


def _bootstrap(records, key, repeats=1000):
    """Cluster bootstrap; variants of one user/intent stay together."""
    groups = {}
    for record in records:
        groups.setdefault(record["group_id"], []).append(record[key])
    sums = np.array([sum(values) for values in groups.values()])
    sizes = np.array([len(values) for values in groups.values()])
    rng = np.random.default_rng(42)
    values = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(groups), len(groups))
        values.append(sums[chosen].sum() / sizes[chosen].sum())
    return [round(float(x), 4) for x in np.percentile(values, [2.5, 97.5])]


def evaluate_retrieval(retriever, cases: list[dict], cutoffs=(1, 3, 5, 10)) -> dict:
    cutoffs = sorted(set(cutoffs))
    if not cases or not cutoffs or min(cutoffs) <= 0:
        raise ValueError("Nonempty cases and positive cutoffs are required")
    known = set(retriever.chunk_ids)
    missing = {cid for case in cases for cid in case["relevance"]} - known
    if missing:
        raise ValueError(f"Judgments reference missing corpus IDs: {sorted(missing)[:10]}")
    records = []
    for case in cases:
        started = time.perf_counter()
        chunks = retriever.retrieve(case["question"], top_k=max(cutoffs))
        latency = (time.perf_counter() - started) * 1000
        ids = [chunk.chunk_id for chunk in chunks]
        record = {**case, "retrieved_chunk_ids": ids, "scores": [chunk.score for chunk in chunks],
                  "latency_ms": latency, "abstained": not ids}
        if not case["expect_abstain"]:
            for k in cutoffs:
                for name, value in ranked_metrics(ids, case["relevance"], k).items():
                    record[f"{name}@{k}"] = value
                if case["required_chunk_ids"]:
                    record[f"complete_required_evidence@{k}"] = float(set(case["required_chunk_ids"]).issubset(ids[:k]))
        records.append(record)

    def summarize(rows):
        positive = [r for r in rows if not r["expect_abstain"]]
        negative = [r for r in rows if r["expect_abstain"]]
        summary = {"questions": len(rows), "answerable": len(positive), "unanswerable": len(negative)}
        if positive:
            for k in cutoffs:
                for name in ("hit_rate", "recall", "precision", "mrr", "ndcg"):
                    key = f"{name}@{k}"
                    summary[key] = round(float(np.mean([r[key] for r in positive])), 4)
                required = [r for r in positive if r["required_chunk_ids"]]
                if required:
                    key = f"complete_required_evidence@{k}"
                    summary[key] = round(float(np.mean([r[key] for r in required])), 4)
        if negative:
            summary["abstention_accuracy"] = round(sum(r["abstained"] for r in negative) / len(negative), 4)
            summary["false_positive_rate"] = round(1 - summary["abstention_accuracy"], 4)
        return summary

    summary = summarize(records)
    positive = [r for r in records if not r["expect_abstain"]]
    if positive:
        summary["bootstrap_95ci"] = {f"{metric}@{k}": _bootstrap(positive, f"{metric}@{k}")
                                     for metric in ("hit_rate", "recall") for k in cutoffs}
    latencies = [r["latency_ms"] for r in records]
    return {"corpus_sha256": retriever.corpus_fingerprint,
            "questions_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
            "chunks": len(retriever.chunk_ids), "method": retriever.method, "min_score": retriever.min_score,
            "cutoffs": cutoffs, "summary": summary,
            "by_condition": {condition: summarize([r for r in records if r["condition"] == condition])
                             for condition in sorted({r["condition"] for r in records})},
            "latency": {"median_ms": round(float(np.median(latencies)), 3),
                        "p95_ms": round(float(np.percentile(latencies, 95)), 3)},
            "includes_model_inference": False, "records": records}


def evaluate_context(pack, retriever, cases: list[dict], enforce: bool = True) -> dict:
    """Does the prompt the model would be given contain the right text? No model is run.

    For each answerable question the prompt is built exactly as the assistant
    builds it, and must contain the source text of every required section
    (or, where none is marked required, of at least one labelled section).
    `enforce=False` builds the prompt from plain top-k search instead, to show
    what rules-first selection changes.
    """
    if not cases:
        raise ValueError("Nonempty cases are required")
    missing = {cid for case in cases for cid in case["relevance"]} - set(retriever.chunk_ids)
    if missing:
        raise ValueError(f"Judgments reference missing corpus IDs: {sorted(missing)[:10]}")
    search = retriever.retrieve_evidence if enforce else retriever.retrieve
    source = dict(zip(retriever.chunk_ids, retriever.chunks))
    records = []
    for case in cases:
        started = time.perf_counter()
        chunks = search(case["question"])
        prompt = build_messages(pack, case["question"], chunks)[1]["content"] if chunks else ""
        latency = (time.perf_counter() - started) * 1000
        record = {"query_id": case["query_id"], "condition": case["condition"], "question": case["question"],
                  "context_chunk_ids": [c.chunk_id for c in chunks], "model_called": bool(chunks),
                  "prompt_characters": len(prompt), "latency_ms": latency,
                  "expect_abstain": case["expect_abstain"]}
        if not case["expect_abstain"]:
            present = [source[cid] in prompt for cid in (case["required_chunk_ids"] or case["relevance"])]
            record["right_text_in_prompt"] = all(present) if case["required_chunk_ids"] else any(present)
        records.append(record)

    def summarize(rows):
        positive = [r for r in rows if not r["expect_abstain"]]
        negative = [r for r in rows if r["expect_abstain"]]
        summary = {"questions": len(rows)}
        if positive:
            correct = sum(r["right_text_in_prompt"] for r in positive)
            summary.update({"answerable": len(positive), "right_text_in_prompt": correct,
                            "right_text_in_prompt_rate": round(correct / len(positive), 4)})
        if negative:
            skipped = sum(not r["model_called"] for r in negative)
            summary.update({"unanswerable": len(negative), "model_not_called": skipped,
                            "model_not_called_rate": round(skipped / len(negative), 4)})
        return summary

    called = [r for r in records if r["model_called"]]
    latencies = [r["latency_ms"] for r in records]
    return {"corpus_sha256": retriever.corpus_fingerprint, "chunks": len(retriever.chunk_ids),
            "method": retriever.method, "top_k": retriever.top_k, "min_score": retriever.min_score,
            "rules_first": enforce, "summary": summarize(records),
            "by_condition": {condition: summarize([r for r in records if r["condition"] == condition])
                             for condition in sorted({r["condition"] for r in records})},
            "mean_sections_in_prompt": round(float(np.mean([len(r["context_chunk_ids"]) for r in called])), 2) if called else 0,
            "mean_prompt_characters": round(float(np.mean([r["prompt_characters"] for r in called]))) if called else 0,
            "latency": {"median_ms": round(float(np.median(latencies)), 3),
                        "p95_ms": round(float(np.percentile(latencies, 95)), 3)},
            "includes_model_inference": False,
            "failures": [r for r in records if (r["model_called"] if r["expect_abstain"]
                                                else not r["right_text_in_prompt"])]}
