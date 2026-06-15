"""Quick check that the Qwen3-0.6B assistant loads and answers sensibly.

Run this ONCE before the full audit, so model-loading problems surface fast.

    python scripts/smoke_test_qwen.py              # transformers backend
    python scripts/smoke_test_qwen.py ollama       # Ollama backend
"""

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
    backend = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    print(f"Loading assistant (backend={backend}). First run downloads the model...")
    _, _, assistant, _ = build_components(backend=backend)
    print("Loaded.\n")

    for i, q in enumerate(QUESTIONS, 1):
        print(f"--- Question {i} ---\n{q}\n")
        ans = assistant.answer(q)
        print("Answer:\n" + ans.answer_text)
        print("\nEvidence used:", [c.chunk_id for c in ans.retrieved_chunks])
        print("=" * 70)


if __name__ == "__main__":
    main()
