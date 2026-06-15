import json

from engine.config import OUTPUT_DIR
from engine.domain_pack import DomainPack


def write_ideal_answer(record: dict, pack: DomainPack) -> str:
    """Compose an ideal answer from the behavioral spec.

    MVP: template using human-readable entity names from aliases.yaml.
    Stage 2: replace the template with a strong-LLM call conditioned on
    ground_truth + gold chunk text, keeping the human-review step.
    """
    gt = record["ground_truth"]
    names = {eid: aliases[0] for eid, aliases in pack.entity_aliases().items()}
    field_names = {fid: aliases[0] for fid, aliases in pack.field_aliases().items()}

    lines = ["Here is a careful assessment based on the official documents."]

    if gt["likely_eligible"]:
        lines.append("Likely options for you:")
        for eid in gt["likely_eligible"]:
            lines.append(f"- {names.get(eid, eid)}: you appear to meet the stated criteria.")

    if gt["unknown_due_to_missing_info"]:
        lines.append("Cannot be confirmed yet:")
        for eid in gt["unknown_due_to_missing_info"]:
            lines.append(f"- {names.get(eid, eid)}: a key detail is missing, so I can't confirm this yet.")

    for fid in gt["must_ask_about"]:
        lines.append(f"Could you tell me your {field_names.get(fid, fid)}?")

    if gt["ineligible"]:
        for eid in gt["ineligible"]:
            lines.append(f"{names.get(eid, eid)} does not appear to fit based on what you shared.")

    lines.append("Please note this is not a final approval; official verification is required. "
                 "A good next step is to gather your documents and check the official portal.")
    return "\n".join(lines)


def create_sft_data(pack: DomainPack, label: str = "baseline") -> str:
    """SFT examples from FAILED TRAIN-split cases only. Output requires expert review."""
    in_path = OUTPUT_DIR / f"audit_{label}.jsonl"
    out_path = OUTPUT_DIR / "sft_train.jsonl"

    n = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            if r["split"] != "train" or r["evaluation"]["passed"]:
                continue
            example = {
                "messages": [
                    {"role": "system",
                     "content": "You are a careful document-grounded assistant. Ask follow-up "
                                "questions when key information is missing, never guarantee "
                                "outcomes, and ground every claim in the provided documents."},
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": write_ideal_answer(r, pack)},
                ],
                "case_id": r["case_id"],
                "needs_expert_review": True,
            }
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} SFT examples (train-split failures) to {out_path}")
    return str(out_path)


def create_preference_data(pack: DomainPack, label: str = "baseline") -> str:
    """Good/bad pairs from TRAIN split: model's failing answer = rejected,
    ideal answer = chosen. Gate violations are always rejected."""
    in_path = OUTPUT_DIR / f"audit_{label}.jsonl"
    out_path = OUTPUT_DIR / "preference_train.jsonl"

    n = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            if r["split"] != "train":
                continue
            ev = r["evaluation"]
            if ev["passed"]:
                # Passing answers become positive examples.
                fout.write(json.dumps({
                    "prompt": r["question"], "completion": r["answer"],
                    "label": True, "score": ev["overall_score"],
                }, ensure_ascii=False) + "\n")
            else:
                # Failing answer = negative; ideal answer = paired positive.
                fout.write(json.dumps({
                    "prompt": r["question"], "completion": r["answer"],
                    "label": False, "score": ev["overall_score"],
                    "gate_violations": ev["gate_violations"],
                }, ensure_ascii=False) + "\n")
                fout.write(json.dumps({
                    "prompt": r["question"],
                    "completion": write_ideal_answer(r, pack),
                    "label": True, "score": 1.0, "synthetic_ideal": True,
                }, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote preference data for {n} train cases to {out_path}")
    return str(out_path)
