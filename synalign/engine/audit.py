import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config import (OUTPUT_DIR, N_SYNTHETIC_USERS, ACTIVE_DOMAIN, ASSISTANT_BACKEND,
                           RANDOM_SEED, USER_SOURCE, USER_SOURCES)
from engine.domain_pack import DomainPack
from engine.synthesis import make_synthetic_users
from engine.splits import assign_splits
from engine.perturbations import make_test_cases
from engine.retriever import Retriever
from engine.assistant import make_assistant
from engine.evaluator import Evaluator
from engine.retrieval_evaluation import recall_at_k, mrr


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for the mean, via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def build_components(backend: str | None = None, **assistant_kwargs):
    """Assemble the pack, retriever, assistant, and evaluator.

    `backend` is "naive", "transformers" or "ollama" (default: ASSISTANT_BACKEND).
    `assistant_kwargs` go to the model backends (e.g. model_name, max_new_tokens).
    """
    pack = DomainPack(ACTIVE_DOMAIN)
    retriever = Retriever(pack)
    assistant = make_assistant(backend or ASSISTANT_BACKEND, pack, retriever, **assistant_kwargs)
    return pack, retriever, assistant, Evaluator(pack)


def run_audit(n_users: int = N_SYNTHETIC_USERS, backend: str | None = None,
              label: str = "baseline", output_dir: Path | None = None,
              progress_every: int = 0, user_source: str | None = None,
              **assistant_kwargs) -> pd.DataFrame:
    user_source = user_source or USER_SOURCE
    if user_source not in USER_SOURCES:
        raise ValueError(f"Unknown user source: {user_source!r}; choose from {USER_SOURCES}")
    pack, retriever, assistant, evaluator = build_components(backend, **assistant_kwargs)
    if pack.build_ground_truth is None:
        raise ValueError(f"Domain {pack.name!r} has no ground_truth.py, so it cannot run the invented-user audit")

    population = pack.load_population() if user_source == "population" else None
    users = make_synthetic_users(pack.profile_schema, n_users, seed=RANDOM_SEED, population=population)
    splits = assign_splits([u["user_id"] for u in users])
    cases = make_test_cases(pack, users, splits)

    missing = {cid for case in cases for cid in case.ground_truth["gold_chunk_ids"]} - set(retriever.chunk_ids)
    if missing:
        raise ValueError(
            f"Domain ground truth references chunks absent from the selected corpus: {sorted(missing)}. "
            "Use reviewed labels for this corpus or a matching domain pack; replacing documents "
            "does not create new eligibility ground truth."
        )

    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"audit_{label}.jsonl"

    rows = []
    started = time.time()
    with open(records_path, "w", encoding="utf-8") as f:
        for done, case in enumerate(cases, start=1):
            answer = assistant.answer(case.question)          # question ONLY
            rec = recall_at_k(answer.retrieved_chunks, case.ground_truth["gold_chunk_ids"])
            rr = mrr(answer.retrieved_chunks, case.ground_truth["gold_chunk_ids"])
            result = evaluator.evaluate(case, answer, rec, rr)

            record = {
                **case.to_dict(),
                "answer": answer.answer_text,
                "answer_mode": answer.mode,
                "retrieved_chunk_ids": [c.chunk_id for c in answer.retrieved_chunks],
                "evaluation": result.to_dict(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append({**result.to_dict(), "question": case.question})

            if progress_every and (done % progress_every == 0 or done == len(cases)):
                elapsed = time.time() - started
                print(f"progress {done}/{len(cases)} cases | elapsed={elapsed / 60:.1f}m | "
                      f"eta={elapsed / done * (len(cases) - done) / 60:.1f}m", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"audit_{label}_flat.csv", index=False)  # human convenience only
    return df


def summarize(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Per-condition means with 95% CIs, pass rates, and gate counts."""
    if split:
        df = df[df["split"] == split]
    out = []
    for condition, g in df.groupby("condition"):
        lo, hi = bootstrap_ci(list(g["overall_score"]))
        out.append({
            "condition": condition,
            "n": len(g),
            "pass_rate": round(g["passed"].mean(), 3),
            "overall": round(g["overall_score"].mean(), 3),
            "overall_ci95": f"[{lo:.3f}, {hi:.3f}]",
            "coverage": round(g["coverage_score"].mean(), 3),
            "followup": round(g["followup_score"].mean(), 3),
            "groundedness": round(g["groundedness_score"].mean(), 3),
            "recall@k": round(g["retrieval_recall"].mean(), 3),
            "mrr": round(g["retrieval_mrr"].mean(), 3),
            "gate_violations": int(g["gate_violations"].apply(len).sum()),
        })
    return pd.DataFrame(out).sort_values("overall").reset_index(drop=True)


def worst_case(df: pd.DataFrame, split: str | None = None) -> dict:
    """The single number that must improve: the weakest condition."""
    s = summarize(df, split)
    row = s.iloc[0]
    return {"condition": row["condition"], "overall": row["overall"],
            "pass_rate": row["pass_rate"]}
