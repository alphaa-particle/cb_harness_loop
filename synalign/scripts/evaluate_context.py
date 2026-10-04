"""Check, without a model, that the right scheme text reaches the model's prompt.

    python scripts/build_scheme_corpus.py --output /tmp/schemes
    python scripts/evaluate_context.py --corpus /tmp/schemes/corpus.jsonl \
        --questions /tmp/schemes/questions.jsonl --split test --method hybrid --top-k 3

Each question's prompt is built exactly as the assistant builds it. A question
passes when the prompt contains the labelled source text; a question the corpus
cannot answer passes when the model would not be called at all. The same
questions are also run through plain top-k search, to show what putting each
scheme's rules first changes.
"""

import argparse
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import ACTIVE_DOMAIN
from engine.domain_pack import DomainPack
from engine.retrieval_evaluation import evaluate_context, load_questions
from engine.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", help="Exact labelled split; all questions if omitted")
    parser.add_argument("--method", choices=["char_tfidf", "word_tfidf", "hybrid"], default="char_tfidf")
    parser.add_argument("--top-k", type=int, default=3, help="Most sections allowed in the prompt")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help="Write the report, including every question that failed")
    args = parser.parse_args()
    pack = DomainPack(ACTIVE_DOMAIN, corpus_path=args.corpus)
    pack.eval_config["retrieval"].update(method=args.method, top_k=args.top_k, min_score=args.min_score)
    cases = load_questions(args.questions, args.split)
    started = time.perf_counter()
    retriever = Retriever(pack)
    build_seconds = round(time.perf_counter() - started, 3)
    result = {"source": str(args.corpus), "split": args.split or "all", "build_seconds": build_seconds,
              "python": platform.python_version(), "platform": platform.machine(),
              "rules_first": evaluate_context(pack, retriever, cases, enforce=True),
              "plain_top_k": evaluate_context(pack, retriever, cases, enforce=False)}
    del result["plain_top_k"]["failures"]  # the comparison only needs its totals
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del result["rules_first"]["failures"]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
