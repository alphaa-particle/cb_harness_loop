"""Benchmark single-node retrieval with synthetic or existing domain documents.

Synthetic mode checks capacity and named-scheme retrieval, not real-world
answer accuracy. Use --domain and --questions for a labelled real corpus.
No model is loaded and existing audit/training files are never written.
"""

import argparse
import json
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.corpus import CorpusPack
from engine.domain_pack import DomainPack
from engine.retriever import Retriever


def synthetic_corpus(path, count, queries, chunks_per_scheme=1):
    states = ["Punjab", "Haryana", "Delhi", "Bihar", "Maharashtra"]
    topics = ["Education Scholarship", "Medical Treatment", "Housing Repair",
              "Agriculture Equipment", "Worker Pension", "Skill Training",
              "Disability Assistance", "Business Credit", "Child Nutrition", "Livelihood Support"]
    sections, examples = [], []
    for i in range(count):
        cid = f"synthetic_{i:05d}"
        state, topic = states[i % len(states)], topics[(i // len(states)) % len(topics)]
        name = f"{state} {topic} Programme {i:05d}"
        income = 5000 + (i % 40) * 1000
        sections.append(
            f"## Scheme: {name}\nScheme ID: {cid}\n"
            f"Purpose: {topic} for applicants residing in {state}.\n"
            f"Eligibility rules:\n- Applicant must reside in {state}.\n"
            f"- Age must be between {18 + i % 10} and {50 + i % 20}.\n"
            f"- Monthly household income must be {income} or below.\n"
            "- Applicants must provide identity proof, residence evidence and an income certificate.\n"
            f"Benefits: {topic}, subject to verification and available funding.\n"
            "Application: Submit the completed application and supporting documents to the designated "
            "office. Officials review the documents and may request additional information.\n"
            "Important caution: Final approval depends on official verification. Never guarantee "
            "approval or payment. Ask about missing eligibility details before confirming eligibility."
        )
        extra_topics = ["Application documents", "Benefit disbursement", "Review and appeals"]
        for section in range(1, chunks_per_scheme):
            section_topic = extra_topics[(section - 1) % len(extra_topics)]
            sections.append(
                f"## Scheme: {name} — {section_topic}\nChunk ID: {cid}_section_{section}\n"
                f"This section describes {section_topic.lower()} for {name} in {state}.\n"
                "An applicant must submit the designated form and supporting certificates to the local office. "
                "The office verifies the submission and records its decision. Payments depend on approval "
                "and the availability of funds. An unsuccessful applicant may request a review through "
                "the designated office. The case reference should be quoted in any correspondence. "
                "Keep copies of submitted documents and acknowledgments."
            )
        examples.append({"question": f"What are the eligibility rules for {name}?",
                         "gold_chunk_ids": [cid]})
    (path / "synthetic_schemes.md").write_text("\n\n---\n\n".join(sections), encoding="utf-8")
    random.Random(42).shuffle(examples)
    return CorpusPack(path), examples[:queries]


def measure(pack, examples, source):
    started = time.perf_counter()
    retriever = Retriever(pack)
    if retriever.method != "char_tfidf" or retriever.min_score != 0:
        raise ValueError("Legacy speed comparison requires char_tfidf and min_score=0; use evaluate_retrieval.py for other methods")
    build_seconds = time.perf_counter() - started
    # Verify every indexed section can be supplied as original gold context.
    direct = retriever.get_chunks_by_ids(retriever.chunk_ids)
    if [(c.chunk_id, c.text) for c in direct] != list(zip(retriever.chunk_ids, retriever.chunks)):
        raise AssertionError("Gold context IDs or source text changed")

    known_ids = set(retriever.chunk_ids)
    unknown_gold = {cid for example in examples for cid in example["gold_chunk_ids"]} - known_ids
    if unknown_gold:
        raise ValueError(f"Questions reference unindexed gold chunks: {sorted(unknown_gold)[:10]}")

    def legacy(query):
        qv = retriever.vectorizer.transform([query])
        scores = cosine_similarity(qv, retriever._search_matrix.T).flatten()
        indices = scores.argsort()[::-1][:retriever.top_k]
        return [retriever.chunk_ids[i] for i in indices if scores[i] > 0]

    for example in examples[:5]:
        retriever.retrieve(example["question"])
        legacy(example["question"])

    elapsed, old_elapsed, hits, old_hits, recalls = [], [], 0, 0, []
    failed_examples = []
    for example in examples:
        question, expected = example["question"], set(example["gold_chunk_ids"])
        if not expected:
            raise ValueError("Benchmark questions must identify at least one gold chunk")
        started = time.perf_counter()
        result = retriever.retrieve(question)
        elapsed.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        old_ids = legacy(question)
        old_elapsed.append((time.perf_counter() - started) * 1000)
        found = {c.chunk_id for c in result}
        hits += bool(found & expected)
        old_hits += bool(set(old_ids) & expected)
        recalls.append(len(found & expected) / len(expected))
        if not found & expected and len(failed_examples) < 5:
            failed_examples.append({"question": question, "expected": sorted(expected),
                                    "retrieved": [c.chunk_id for c in result]})

    def latency(values):
        return {"median_ms": round(float(np.median(values)), 3),
                "p95_ms": round(float(np.percentile(values, 95)), 3)}

    def matrix_bytes(matrix):
        return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes

    return {
        "source": source,
        "platform": platform.machine(),
        "python": platform.python_version(),
        "corpus_sha256": retriever.corpus_fingerprint,
        "chunks": len(retriever.chunks),
        "questions": len(examples),
        "top_k": retriever.top_k,
        "build_seconds": round(build_seconds, 3),
        "sparse_matrices_mib": round(matrix_bytes(retriever._search_matrix) / 1024 ** 2, 2),
        "current_retrieval": latency(elapsed),
        "previous_retrieval": latency(old_elapsed),
        "hit_rate_at_k": round(hits / len(examples), 4),
        "previous_hit_rate_at_k": round(old_hits / len(examples), 4),
        "mean_recall_at_k": round(float(np.mean(recalls)), 4),
        "gold_chunks_verified": len(direct),
        "failed_examples": failed_examples,
        "concurrency": 1,
        "includes_model_inference": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemes", type=int, default=5000, help="Number of synthetic schemes")
    parser.add_argument("--chunks-per-scheme", type=int, default=1, help="Synthetic sections per scheme, including its rules")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3, help="Evidence count for the synthetic benchmark")
    parser.add_argument("--domain", help="Use this existing domain instead of synthetic documents")
    parser.add_argument("--questions", type=Path, help="JSONL with question and gold_chunk_ids (or audit records)")
    parser.add_argument("--output", type=Path, help="Save the benchmark measurements as JSON")
    args = parser.parse_args()
    if args.schemes <= 0 or args.queries <= 0 or args.chunks_per_scheme <= 0 or args.top_k <= 0:
        parser.error("--schemes, --chunks-per-scheme, --top-k and --queries must be positive")
    if bool(args.domain) != bool(args.questions):
        parser.error("--domain and --questions must be supplied together")

    if args.domain:
        records = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines()
                   if line.strip()][:args.queries]
        if not records:
            parser.error("The question file is empty")
        examples = [{"question": r["question"],
                     "gold_chunk_ids": r.get("gold_chunk_ids", r.get("ground_truth", {}).get("gold_chunk_ids", []))}
                    for r in records]
        result = measure(DomainPack(args.domain), examples, f"domain:{args.domain}")
    else:
        with tempfile.TemporaryDirectory(prefix="synalign-retrieval-") as directory:
            pack, examples = synthetic_corpus(Path(directory), args.schemes, args.queries, args.chunks_per_scheme)
            pack.eval_config["retrieval"]["top_k"] = args.top_k
            result = measure(pack, examples, "synthetic named-scheme questions; not real-world accuracy")
            result.update({"schemes": args.schemes, "chunks_per_scheme": args.chunks_per_scheme})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
