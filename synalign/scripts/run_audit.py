import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.audit import run_audit, summarize, worst_case


if __name__ == "__main__":
    df = run_audit(label="baseline")

    for split in ("train", "dev", "test"):
        print(f"\n=== SYNALIGN AUDIT — {split.upper()} SPLIT ===")
        print(summarize(df, split).to_string(index=False))

    print("\n=== WORST CASE (dev) ===")
    print(worst_case(df, "dev"))

    print("\nData of record: data/outputs/audit_baseline.jsonl")
