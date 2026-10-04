"""Fresh non-LLM audit plus explicitly identified re-scoring of saved LLM answers.

This measures the existing demo's heuristic behavioral checks, not factual
answer accuracy on an external corpus. No model weights are downloaded.
"""

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.audit import run_audit
from engine.domain_pack import DomainPack
from engine.evaluator import Evaluator
from engine.retrieval_evaluation import recall_at_k, mrr
from engine.retriever import Retriever
from engine.schemas import TestCase, AssistantAnswer


def summarize(rows):
    def subset(group):
        return {"cases": len(group), "passed": sum(r["passed"] for r in group),
                "pass_rate": round(sum(r["passed"] for r in group) / len(group), 4),
                "mean_retrieval_recall": round(sum(r["retrieval_recall"] for r in group) / len(group), 4),
                "mean_followup": round(sum(r["followup_score"] for r in group) / len(group), 4),
                "gate_violations": sum(len(r["gate_violations"]) for r in group)}
    return {"all": subset(rows),
            "by_split": {split: subset([r for r in rows if r["split"] == split])
                         for split in sorted({r["split"] for r in rows})},
            "test_by_condition": {condition: subset([r for r in rows if r["split"] == "test" and r["condition"] == condition])
                                  for condition in sorted({r["condition"] for r in rows if r["split"] == "test"})}}


def replay(path, retriever, evaluator):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    old, updated, changes = [], [], []
    for row in records:
        case = TestCase(**{f.name: row[f.name] for f in fields(TestCase)})
        ids = row["retrieved_chunk_ids"]
        chunks = retriever.get_chunks_by_ids(ids)
        if [c.chunk_id for c in chunks] != ids:
            raise ValueError(f"Saved answer references evidence missing from corpus: {path}")
        result = evaluator.evaluate(case, AssistantAnswer(row["answer"], chunks),
                                    recall_at_k(chunks, case.ground_truth["gold_chunk_ids"]),
                                    mrr(chunks, case.ground_truth["gold_chunk_ids"])).to_dict()
        old.append(row["evaluation"])
        updated.append(result)
        if result != row["evaluation"]:
            changes.append({"case_id": case.case_id, "original": row["evaluation"], "rescored": result})
    return {"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mode": "saved_answer_replay_no_new_inference", "original": summarize(old),
            "rescored": summarize(updated), "changed_records": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True, help="Directory for fresh baseline JSONL and CSV")
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--saved-audit", type=Path, action="append", default=[])
    args = parser.parse_args()
    pack = DomainPack("welfare_demo")
    if pack.corpus_path != pack.documents_dir.resolve():
        parser.error("This evaluation uses demo ground truth; unset SYNALIGN_CORPUS_PATH for this command")
    # The active domain of run_audit must also be the demo.
    from engine.config import ACTIVE_DOMAIN
    if ACTIVE_DOMAIN != "welfare_demo":
        parser.error("Set SYNALIGN_DOMAIN=welfare_demo for this demo-only evaluation")
    retriever, evaluator = Retriever(pack), Evaluator(pack)
    df = run_audit(n_users=args.users, backend="naive", label="research_baseline", output_dir=args.audit_dir)
    result = {"corpus_sha256": retriever.corpus_fingerprint, "method": retriever.method,
              "fresh_baseline": {"backend": "naive", "new_inference": True,
                                 "records_path": str(args.audit_dir / "audit_research_baseline.jsonl"),
                                 **summarize(df.to_dict("records"))},
              "saved_audits": [replay(path, retriever, evaluator) for path in args.saved_audit],
              "limitation": "Behavioral heuristic scores for two demo schemes; not factual correctness or a fresh LLM evaluation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    printable = {**result, "saved_audits": [{k: v for k, v in run.items() if k != "changed_records"}
                                           for run in result["saved_audits"]]}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
