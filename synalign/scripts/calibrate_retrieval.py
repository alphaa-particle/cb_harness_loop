"""Choose the minimum search score using labelled development questions only.

Below that score nothing is retrieved, so the assistant answers with its fixed
"no matching scheme" reply instead of asking the model. The threshold is chosen
to balance two things: answerable questions keep their labelled evidence in the
prompt, and unanswerable questions retrieve nothing.

The script only reports its choice. Apply it with --min-score when evaluating
and SYNALIGN_RETRIEVAL_MIN_SCORE at runtime, then check the held-out split.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.corpus import CorpusPack, RULES_SECTION
from engine.retrieval_evaluation import load_questions
from engine.retriever import Retriever


def calibrate(retriever, cases):
    if any(case["split"] != "dev" for case in cases):
        raise ValueError("Calibration requires explicitly labelled dev cases; test cases are not permitted")
    positives = sum(not case["expect_abstain"] for case in cases)
    negatives = len(cases) - positives
    if not positives or not negatives:
        raise ValueError("Development data must contain both answerable and unanswerable questions")
    missing = {cid for case in cases for cid in case["relevance"]} - set(retriever.chunk_ids)
    if missing:
        raise ValueError(f"Development labels reference absent chunks: {sorted(missing)}")
    if retriever.min_score != 0:
        raise ValueError("Calibrate with min_score 0; the threshold is what is being chosen")

    def scheme(chunk):
        return chunk.scheme_id or chunk.chunk_id

    observations = []
    for case in cases:
        ranked = retriever.retrieve(case["question"])
        relevant = [c for c in retriever.retrieve_evidence(case["question"]) if c.chunk_id in case["relevance"]]
        # The score a threshold must not exceed for this evidence to stay in the
        # prompt: a rule section comes in with the best-matching section of its
        # scheme, any other section only on its own match.
        relevant_score = max((max(r.score for r in ranked if scheme(r) == scheme(c))
                              if c.section.strip().lower() == RULES_SECTION else c.score
                              for c in relevant), default=0.0)
        observations.append((ranked[0].score if ranked else 0.0, bool(ranked), case["expect_abstain"], relevant_score))
    candidates = []
    for step in range(101):
        threshold = step / 100
        correct_positive = sum(relevant_score > 0 and relevant_score >= threshold for score, present, negative, relevant_score
                               in observations if not negative) / positives
        correct_negative = sum(not present or score < threshold for score, present, negative, relevant_score
                               in observations if negative) / negatives
        candidates.append({"min_score": threshold, "positive_hit_and_accept_rate": correct_positive,
                           "abstention_accuracy": correct_negative,
                           "balanced_accuracy": (correct_positive + correct_negative) / 2})
    selected = max(candidates, key=lambda r: (r["balanced_accuracy"], r["positive_hit_and_accept_rate"], -r["min_score"]))
    # The balanced rule will give up answerable questions to refuse unanswerable
    # ones. This one never does: it refuses only what can be refused for free.
    keeping = max(candidates, key=lambda r: (r["positive_hit_and_accept_rate"], r["abstention_accuracy"], -r["min_score"]))
    return {"corpus_sha256": retriever.corpus_fingerprint, "method": retriever.method,
            "top_k": retriever.top_k, "split": "dev", "questions": len(cases),
            "answerable": positives, "unanswerable": negatives,
            "selection_rule": "maximize balanced correct-hit/abstention accuracy; ties prefer positive recall then lower threshold",
            "selected": selected,
            "keeping_answerable_rule": "lose no answerable question; then refuse as many unanswerable as possible; then the lowest threshold",
            "selected_keeping_answerable": keeping, "candidates": candidates}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--method", choices=["char_tfidf", "word_tfidf", "hybrid"], default="char_tfidf")
    parser.add_argument("--top-k", type=int, default=3, help="Most sections allowed in the prompt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    retriever = Retriever(CorpusPack(args.corpus, top_k=args.top_k, method=args.method))
    result = calibrate(retriever, load_questions(args.questions, "dev"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))
