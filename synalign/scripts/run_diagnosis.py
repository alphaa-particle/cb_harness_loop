import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.audit import build_components
from engine.diagnosis import load_records, failure_type_summary, attribute_failures


if __name__ == "__main__":
    records = load_records("baseline")
    pack, retriever, assistant, evaluator = build_components()

    print("=== FAILURE TYPES BY CONDITION ===")
    print(failure_type_summary(records).to_string(index=False))

    attr = attribute_failures(records, pack, retriever, assistant, evaluator)
    if len(attr):
        print("\n=== FAILURE ATTRIBUTION ===")
        print(attr["verdict"].value_counts().to_string())
