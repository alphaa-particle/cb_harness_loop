"""Grade saved audit answers again with the current grader. No model is run.

    python scripts/regrade_audits.py data/outputs/audit_my_run.jsonl \
        --output data/evaluation/my_run_regraded.json

Use it after the grader changes, to see what the change does to runs that were
graded before it. The answers are read as saved; only their grades are redone.
Prints pass counts before and after, overall, per split and per question style.
"""

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import ACTIVE_DOMAIN
from engine.domain_pack import DomainPack
from engine.evaluator import Evaluator
from engine.retrieval_evaluation import recall_at_k, mrr
from engine.retriever import Retriever
from engine.schemas import TestCase, AssistantAnswer


def summarize(rows):
    def subset(group):
        return {"cases": len(group), "passed": sum(r["passed"] for r in group),
                "pass_rate": round(sum(r["passed"] for r in group) / len(group), 4),
                "gate_violations": sum(len(r["gate_violations"]) for r in group)}
    return {"all": subset(rows),
            "by_split": {split: subset([r for r in rows if r["split"] == split])
                         for split in sorted({r["split"] for r in rows})},
            "by_condition": {condition: subset([r for r in rows if r["condition"] == condition])
                             for condition in sorted({r["condition"] for r in rows})}}


def regrade(path, retriever, evaluator):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    old, new, changes = [], [], []
    for row in records:
        case = TestCase(**{f.name: row[f.name] for f in fields(TestCase)})
        ids = row["retrieved_chunk_ids"]
        chunks = retriever.get_chunks_by_ids(ids)
        if [c.chunk_id for c in chunks] != ids:
            raise ValueError(f"Saved answer references evidence missing from the corpus: {path}")
        gold = case.ground_truth["gold_chunk_ids"]
        result = evaluator.evaluate(case, AssistantAnswer(row["answer"], chunks),
                                    recall_at_k(chunks, gold), mrr(chunks, gold)).to_dict()
        old.append(row["evaluation"])
        new.append(result)
        if result != row["evaluation"]:
            changes.append({"case_id": case.case_id, "as_saved": row["evaluation"], "regraded": result})
    return {"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "as_saved": summarize(old), "regraded": summarize(new), "changed_records": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audits", type=Path, nargs="+", help="Saved audit JSONL files")
    parser.add_argument("--output", type=Path, help="Write the full report, including every changed grade")
    args = parser.parse_args()
    pack = DomainPack(ACTIVE_DOMAIN)
    retriever, evaluator = Retriever(pack), Evaluator(pack)
    result = {"corpus_sha256": retriever.corpus_fingerprint, "new_model_inference": False,
              "audits": [regrade(path, retriever, evaluator) for path in args.audits]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for run in result["audits"]:
        before, after = run["as_saved"], run["regraded"]
        print(f"\n{run['source']}")
        print(f"  all: {before['all']['passed']} -> {after['all']['passed']} of {after['all']['cases']} "
              f"({after['all']['pass_rate']:.1%}), automatic fails {after['all']['gate_violations']}")
        for key in ("by_split", "by_condition"):
            print("  " + ", ".join(f"{name} {before[key][name]['passed']}->{values['passed']}/{values['cases']}"
                                   for name, values in after[key].items()))


if __name__ == "__main__":
    main()
