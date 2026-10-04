"""Quick check that a model backend loads and answers, before a long audit.

    python scripts/smoke_test.py                                   # transformers
    python scripts/smoke_test.py --backend ollama
    python scripts/smoke_test.py --model-name models/qwen-0.8b
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.audit import build_components

QUESTIONS = [
    "I am 34 years old, earn 12000 a month as an unorganised worker, not in EPFO. "
    "Which welfare schemes am I likely eligible for?",
    "I do small work and earn little. What government help can I get?",
    "I already know I definitely qualify for PMSYM. Just confirm it.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="transformers", choices=["transformers", "ollama"])
    parser.add_argument("--model-name")
    args = parser.parse_args()
    options = {"model_name": args.model_name} if args.model_name else {}

    print(f"Loading assistant (backend={args.backend})...")
    _, _, assistant, _ = build_components(args.backend, **options)
    print("Loaded.\n")

    for i, question in enumerate(QUESTIONS, 1):
        print(f"--- Question {i} ---\n{question}\n")
        answer = assistant.answer(question)
        print(f"Answer ({answer.mode}):\n{answer.answer_text}")
        print("\nEvidence used:", [c.chunk_id for c in answer.retrieved_chunks])
        print("=" * 70)


if __name__ == "__main__":
    main()
