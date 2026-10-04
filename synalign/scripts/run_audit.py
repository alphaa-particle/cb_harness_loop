"""Run the audit: synthetic users ask, the assistant answers, the evaluator grades.

    python scripts/run_audit.py                                  # naive baseline, no model
    python scripts/run_audit.py --backend transformers --model-name models/qwen-0.8b \
        --users 100 --label my_run --progress-every 10
    python scripts/run_audit.py --backend ollama --users 20
    python scripts/run_audit.py --user-source population --label population_baseline

Writes data/outputs/audit_<label>.jsonl (the record) and a flat CSV copy.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.assistant import BACKENDS, NO_MODEL_BACKENDS
from engine.audit import run_audit, summarize, worst_case
from engine.config import ASSISTANT_BACKEND, N_SYNTHETIC_USERS, N_USERS_LLM, USER_SOURCE, USER_SOURCES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default=ASSISTANT_BACKEND, choices=BACKENDS)
    parser.add_argument("--users", type=int, help="Synthetic users; each asks five questions")
    parser.add_argument("--user-source", default=USER_SOURCE, choices=USER_SOURCES,
                        help="Invent users from the schema's ranges, or draw them from its population file")
    parser.add_argument("--label", help="Names the output files (default: baseline, or the backend)")
    parser.add_argument("--model-name", help="Local model path (transformers) or model tag (ollama)")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N cases")
    args = parser.parse_args()

    naive = args.backend in NO_MODEL_BACKENDS
    users = args.users or (N_SYNTHETIC_USERS if naive else N_USERS_LLM)
    label = args.label or ("baseline" if args.backend == "naive" else args.backend)
    options = {name: value for name, value in (("model_name", args.model_name),
                                               ("max_new_tokens", args.max_new_tokens)) if value}
    if naive and options:
        parser.error("--model-name and --max-new-tokens need a model backend")

    print(f"Audit | backend={args.backend} | users={users} ({users * 5} cases) from {args.user_source} "
          f"| label={label}", flush=True)
    started = time.time()
    df = run_audit(n_users=users, backend=args.backend, label=label,
                   progress_every=args.progress_every, user_source=args.user_source, **options)
    print(f"Finished in {time.time() - started:.0f}s.")

    for split in ("train", "dev", "test"):
        if (df["split"] == split).any():
            print(f"\n=== {split.upper()} SPLIT ===")
            print(summarize(df, split).to_string(index=False))

    print("\n=== WORST CASE (dev) ===")
    print(worst_case(df, "dev") if (df["split"] == "dev").any() else worst_case(df))
    print(f"\nData of record: data/outputs/audit_{label}.jsonl")
    print(f"Next: python scripts/run_diagnosis.py --label {label} --backend {args.backend}")


if __name__ == "__main__":
    main()
