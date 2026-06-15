"""Run the SYNALIGN audit on the Qwen3-0.6B RAG assistant.

    python scripts/run_audit_qwen.py                 # transformers, 20 users
    python scripts/run_audit_qwen.py --users 8       # smaller/faster
    python scripts/run_audit_qwen.py --backend ollama
    python scripts/run_audit_qwen.py --sample         # nucleus sampling on

Because a 0.6B model is slow on CPU, this defaults to a small user sample
(N_USERS_LLM). Use a GPU and raise --users for a more stable estimate.
The full statistical machinery (splits, per-condition scores, bootstrap CIs,
gates, retrieval metrics) is identical to the baseline audit.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import N_USERS_LLM
from engine.audit import run_audit, summarize, worst_case
from engine.domain_pack import DomainPack
from engine.config import ACTIVE_DOMAIN
from engine.diagnosis import load_records, failure_type_summary, attribute_failures
from engine.audit import build_components


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=N_USERS_LLM)
    ap.add_argument("--backend", default="qwen", choices=["qwen", "ollama"])
    ap.add_argument("--sample", action="store_true", help="use nucleus sampling")
    ap.add_argument("--label", default="qwen")
    args = ap.parse_args()

    print(f"Auditing Qwen3-0.6B  | backend={args.backend} | "
          f"users={args.users} ({args.users * 5} cases) | "
          f"decoding={'sampling' if args.sample else 'greedy'}")
    t0 = time.time()

    df = run_audit(
        n_users=args.users,
        backend=args.backend,
        label=args.label,
        do_sample=args.sample,
    )
    print(f"Audit finished in {time.time() - t0:.0f}s.\n")

    for split in ("train", "dev", "test"):
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        print(f"=== {split.upper()} SPLIT ===")
        print(summarize(df, split).to_string(index=False))
        print()

    print("=== WORST CASE (all splits) ===")
    print(worst_case(df))
    print()

    # Diagnosis: failure types + retrieval-vs-generation attribution.
    records = load_records(args.label)
    print("=== FAILURE TYPES BY CONDITION ===")
    fts = failure_type_summary(records)
    print(fts.to_string(index=False) if len(fts) else "(no failures)")

    pack, retriever, assistant, evaluator = build_components(backend=args.backend,
                                                             do_sample=args.sample)
    attr = attribute_failures(records, pack, retriever, assistant, evaluator)
    if len(attr):
        print("\n=== FAILURE ATTRIBUTION (oracle retrieval re-run) ===")
        print(attr["verdict"].value_counts().to_string())

    print(f"\nData of record: data/outputs/audit_{args.label}.jsonl")


if __name__ == "__main__":
    main()
