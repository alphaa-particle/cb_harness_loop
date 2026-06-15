import json

import numpy as np
import pandas as pd

from engine.config import OUTPUT_DIR, N_SYNTHETIC_USERS, ACTIVE_DOMAIN, RANDOM_SEED
from engine.domain_pack import DomainPack
from engine.synthesis import make_synthetic_users
from engine.splits import assign_splits
from engine.perturbations import make_test_cases
from engine.retriever import TfidfRetriever
from engine.assistant import NaiveBaselineAssistant
from engine.evaluator import Evaluator
from engine.retrieval_metrics import recall_at_k, mrr


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for the mean, via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def build_components(assistant_cls=None, backend: str | None = None, **assistant_kwargs):
    """Assemble the pack, retriever, assistant, and evaluator.

    Pick the assistant by `backend` ("naive" | "qwen" | "ollama"), or pass an
    explicit `assistant_cls`. `assistant_kwargs` are forwarded to LLM backends
    (e.g. model_name, do_sample, host).
    """
    from engine.config import ASSISTANT_BACKEND

    pack = DomainPack(ACTIVE_DOMAIN)
    retriever = TfidfRetriever(pack)

    if assistant_cls is None:
        backend = (backend or ASSISTANT_BACKEND).lower()
        if backend == "naive":
            assistant_cls = NaiveBaselineAssistant
        elif backend == "qwen":
            from engine.qwen_assistant import QwenTransformersAssistant
            assistant_cls = QwenTransformersAssistant
        elif backend == "ollama":
            from engine.qwen_assistant import QwenOllamaAssistant
            assistant_cls = QwenOllamaAssistant
        else:
            raise ValueError(f"Unknown backend: {backend}")

    assistant = assistant_cls(pack, retriever, **assistant_kwargs)
    evaluator = Evaluator(pack)
    return pack, retriever, assistant, evaluator


def run_audit(n_users: int = N_SYNTHETIC_USERS, assistant_cls=None,
              backend: str | None = None, label: str = "baseline",
              **assistant_kwargs) -> pd.DataFrame:
    pack, retriever, assistant, evaluator = build_components(
        assistant_cls=assistant_cls, backend=backend, **assistant_kwargs
    )

    users = make_synthetic_users(pack.profile_schema, n_users, seed=RANDOM_SEED)
    splits = assign_splits([u["user_id"] for u in users])
    cases = make_test_cases(pack, users, splits)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records_path = OUTPUT_DIR / f"audit_{label}.jsonl"

    rows = []
    with open(records_path, "w", encoding="utf-8") as f:
        for case in cases:
            answer = assistant.answer(case.question)          # question ONLY
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

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / f"audit_{label}_flat.csv", index=False)  # human convenience only
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
