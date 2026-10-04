"""Explain an audit's failures: what kind, and was it the search or the writing?

    python scripts/run_diagnosis.py                       # the naive baseline audit
    python scripts/run_diagnosis.py --label my_run --backend transformers --model-name models/qwen-0.8b

Attribution answers every failed case again with the correct evidence forced
into the prompt, so a model backend runs the model once more per failure.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.assistant import BACKENDS
from engine.audit import build_components
from engine.config import ASSISTANT_BACKEND
from engine.diagnosis import load_records, failure_type_summary, attribute_failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--backend", default=ASSISTANT_BACKEND, choices=BACKENDS)
    parser.add_argument("--model-name")
    args = parser.parse_args()
    options = {"model_name": args.model_name} if args.model_name else {}

    records = load_records(args.label)
    failures = failure_type_summary(records)
    print("=== FAILURE TYPES BY CONDITION ===")
    print(failures.to_string(index=False) if len(failures) else "(no failures)")

    _, retriever, assistant, evaluator = build_components(args.backend, **options)
    attribution = attribute_failures(records, retriever, assistant, evaluator)
    if len(attribution):
        print("\n=== FAILURE ATTRIBUTION (correct evidence forced in) ===")
        print(attribution["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
