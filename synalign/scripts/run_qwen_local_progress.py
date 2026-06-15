"""Run a progress-reporting local Qwen audit.

This is useful for larger local runs where the standard `run_audit_qwen.py`
stays quiet until the whole audit finishes.

Example:
    python scripts/run_qwen_local_progress.py --cases 500
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Work around a broken optional pyarrow import in the requested conda env.
# Pandas imports cleanly first; sklearn later only needs pyarrow.__version__.
import pandas as pd

if "pyarrow" not in sys.modules:
    pyarrow_stub = types.ModuleType("pyarrow")
    pyarrow_stub.__version__ = "17.0.0"
    for name in ("Table", "ChunkedArray", "Array"):
        setattr(pyarrow_stub, name, type(name, (), {}))
    sys.modules["pyarrow"] = pyarrow_stub
else:
    pyarrow_mod = sys.modules["pyarrow"]
    if not hasattr(pyarrow_mod, "__version__"):
        pyarrow_mod.__version__ = "17.0.0"
    for name in ("Table", "ChunkedArray", "Array"):
        if not hasattr(pyarrow_mod, name):
            setattr(pyarrow_mod, name, type(name, (), {}))

from engine.config import ACTIVE_DOMAIN, OUTPUT_DIR, RANDOM_SEED
from engine.domain_pack import DomainPack
from engine.synthesis import make_synthetic_users
from engine.splits import assign_splits
from engine.perturbations import make_test_cases
from engine.retriever import TfidfRetriever
from engine.evaluator import Evaluator
from engine.retrieval_metrics import recall_at_k, mrr
from engine.audit import summarize, worst_case
from engine.qwen_assistant import QwenTransformersAssistant


def print_cuda_status(stage: str):
    try:
        import torch

        if not torch.cuda.is_available():
            print(f"{stage}: CUDA unavailable", flush=True)
            return
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(
            f"{stage}: cuda_available=True | device={torch.cuda.get_device_name(0)} | "
            f"allocated={allocated:.2f} GiB | reserved={reserved:.2f} GiB",
            flush=True,
        )
    except Exception as exc:
        print(f"{stage}: CUDA status unavailable: {exc}", flush=True)


def parse_args():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=500)
    ap.add_argument("--label", default="qwen_500_cases")
    ap.add_argument("--model-name", default=str(root / "models" / "qwen3_0.6"))
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--progress-every", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    conditions_per_user = 5
    if args.cases % conditions_per_user != 0:
        raise SystemExit(
            f"--cases must be divisible by {conditions_per_user}; got {args.cases}"
        )
    n_users = args.cases // conditions_per_user

    print(
        f"Local Qwen audit | users={n_users} | cases={args.cases} | "
        f"model={args.model_name} | device={args.device} | "
        f"max_new_tokens={args.max_new_tokens}",
        flush=True,
    )

    pack = DomainPack(ACTIVE_DOMAIN)
    retriever = TfidfRetriever(pack)
    print_cuda_status("before_model_load")
    assistant = QwenTransformersAssistant(
        pack,
        retriever,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    model_device = next(assistant.model.parameters()).device
    print(f"model_parameter_device={model_device}", flush=True)
    print_cuda_status("after_model_load")
    evaluator = Evaluator(pack)

    users = make_synthetic_users(pack.profile_schema, n_users, seed=RANDOM_SEED)
    splits = assign_splits([u["user_id"] for u in users])
    cases = make_test_cases(pack, users, splits)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records_path = OUTPUT_DIR / f"audit_{args.label}.jsonl"

    rows = []
    t0 = time.time()
    with open(records_path, "w", encoding="utf-8") as f:
        for idx, case in enumerate(cases, start=1):
            answer = assistant.answer(case.question)
            rec = recall_at_k(answer.retrieved_chunks, case.ground_truth["gold_chunk_ids"])
            rr = mrr(answer.retrieved_chunks, case.ground_truth["gold_chunk_ids"])
            result = evaluator.evaluate(case, answer, rec, rr)

            record = {
                **case.to_dict(),
                "answer": answer.answer_text,
                "retrieved_chunk_ids": [c.chunk_id for c in answer.retrieved_chunks],
                "evaluation": result.to_dict(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append({**result.to_dict(), "question": case.question})

            if idx % args.progress_every == 0 or idx == len(cases):
                elapsed = time.time() - t0
                rate = elapsed / idx
                remaining = rate * (len(cases) - idx)
                print_cuda_status(f"gpu_after_{idx}_cases")
                print(
                    f"progress {idx}/{len(cases)} cases | "
                    f"elapsed={elapsed/60:.1f}m | eta={remaining/60:.1f}m",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / f"audit_{args.label}_flat.csv", index=False)

    print("\n=== QWEN LOCAL AUDIT SUMMARY (ALL SPLITS) ===")
    print(summarize(df).to_string(index=False))
    for split in ("train", "dev", "test"):
        sub = df[df["split"] == split]
        if len(sub):
            print(f"\n=== {split.upper()} SPLIT ===")
            print(summarize(df, split).to_string(index=False))
    print("\n=== WORST CASE (all splits) ===")
    print(worst_case(df))
    print(f"\nData of record: {records_path}")


if __name__ == "__main__":
    main()
