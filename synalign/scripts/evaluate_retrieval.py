"""Evaluate a swappable local corpus against independently supplied labels."""

import argparse
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.corpus import CorpusPack
from engine.retrieval_evaluation import evaluate_retrieval, load_questions, load_beir_questions
from engine.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path)
    source.add_argument("--beir", type=Path, help="Directory containing corpus.jsonl, queries.jsonl and qrels/")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--split", help="Exact labelled split; BEIR defaults to test")
    parser.add_argument("--method", choices=["char_tfidf", "word_tfidf", "hybrid"], default="char_tfidf")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--output", type=Path, help="Write complete report, including per-query evidence")
    args = parser.parse_args()
    if args.corpus and not args.questions:
        parser.error("--corpus requires --questions with reviewed relevance labels")
    if args.beir and args.questions:
        parser.error("--beir reads its own queries and qrels; omit --questions")
    cases = (load_beir_questions(args.beir, args.split or "test") if args.beir else
             load_questions(args.questions, args.split))
    source = args.beir / "corpus.jsonl" if args.beir else args.corpus
    started = time.perf_counter()
    retriever = Retriever(CorpusPack(source, method=args.method, min_score=args.min_score))
    build_seconds = time.perf_counter() - started
    result = evaluate_retrieval(retriever, cases, args.cutoffs)
    result.update({"source": str(source), "split": args.split or ("test" if args.beir else "all"),
                   "build_seconds": round(build_seconds, 3), "python": platform.python_version(),
                   "platform": platform.machine()})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
