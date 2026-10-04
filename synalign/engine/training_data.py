"""Training data built from audit records.

Each example pairs the prompt the assistant really builds (engine/enforcement.py)
with the answer the ground truth calls for. The evidence in that prompt is the
labelled gold evidence, looked up by ID — never a search prediction.
"""

from engine.domain_pack import DomainPack
from engine.enforcement import build_messages
from engine.schemas import RetrievedChunk

_DECISIONS = ("likely_eligible", "unknown_due_to_missing_info", "ineligible")


def gold_chunks(record: dict, retriever) -> list[RetrievedChunk]:
    """Fetch labelled evidence, never replace it with the retriever's prediction."""
    ids = record["ground_truth"]["gold_chunk_ids"]
    if not ids:
        raise ValueError(f"No gold evidence supplied for {record.get('case_id', 'record')}")
    chunks = retriever.get_chunks_by_ids(ids)
    if [chunk.chunk_id for chunk in chunks] != ids:
        missing = set(ids) - {chunk.chunk_id for chunk in chunks}
        raise ValueError(f"Gold evidence missing from the selected corpus: {sorted(missing)}")
    return chunks


def scheme_names(pack: DomainPack, retriever) -> dict[str, str]:
    """How each scheme is named in a target answer: corpus titles, then the pack's own names."""
    names: dict[str, str] = {}
    for doc in retriever.documents:
        if doc.scheme_key not in names or doc.is_rules:
            names[doc.scheme_key] = doc.title or doc.scheme_key
    if pack.uses_own_documents():
        names.update({key: aliases[0] for key, aliases in pack.entity_aliases().items() if aliases})
    return names


def _ineligibility_reasons(record: dict, pack: DomainPack) -> dict[str, str]:
    gt = record["ground_truth"]
    if "ineligibility_reasons" in gt:
        return gt["ineligibility_reasons"]
    # Audits saved before reasons were recorded: the pack's rules can restate
    # them, but only for the documents those rules were written for.
    if pack.uses_own_documents() and "visible" in record:
        return pack.build_ground_truth(record.get("profile", {}), record["visible"]).get(
            "ineligibility_reasons", {})
    return {}


def ideal_answer(record: dict, pack: DomainPack, names: dict[str, str] | None = None) -> str:
    """The answer the ground truth calls for, in the structure the system prompt asks for."""
    gt = record["ground_truth"]
    names = names or {}
    labels = pack.profile_schema.get("field_labels", {})

    def listed(ids):
        return ", ".join(names.get(x, x) for x in ids)

    none = "none based on the provided information"
    parts = [f"Likely eligible: {listed(gt['likely_eligible']) or none}.",
             f"Needs confirmation: {listed(gt['unknown_due_to_missing_info']) or 'none'}.",
             f"Not likely eligible: {listed(gt['ineligible']) or none}."]
    if gt["ineligible"]:
        reasons = _ineligibility_reasons(record, pack)
        parts.append("Reason: " + " | ".join(
            f"{names.get(s, s)}: "
            f"{reasons.get(s) or 'the reviewed eligibility label indicates that the criteria are not met'}"
            for s in gt["ineligible"]) + ".")
    # Ask in the order the pack lists its fields, so targets are stable.
    order = pack.profile_schema.get("question_fields", [])
    asked = sorted(gt["must_ask_about"], key=lambda f: (order.index(f) if f in order else len(order), f))
    missing = ", ".join(labels.get(f, f) for f in asked)
    parts.append(f"Missing details: please share your {missing}." if missing else "Missing details: none.")
    parts.append(pack.prompts["closing"])
    return " ".join(parts)


def make_example(record: dict, pack: DomainPack, retriever, names: dict[str, str] | None = None) -> dict:
    """One supervised fine-tuning example: enforced prompt -> ideal answer."""
    gt = record["ground_truth"]
    for field in (*_DECISIONS, "must_ask_about"):
        if field not in gt or not isinstance(gt[field], list) or not all(isinstance(x, str) for x in gt[field]):
            raise ValueError(f"Reviewed ground_truth.{field} list required for {record.get('case_id')}")
    decisions = [s for field in _DECISIONS for s in gt[field]]
    if len(set(decisions)) != len(decisions):
        raise ValueError(f"Conflicting or duplicate eligibility labels for {record.get('case_id')}")
    messages = build_messages(pack, record["question"], gold_chunks(record, retriever))
    messages.append({"role": "assistant", "content": ideal_answer(record, pack, names)})
    return {"messages": messages, "case_id": record["case_id"], "split": record["split"],
            "condition": record["condition"]}


def make_preference_pair(record: dict, example: dict) -> dict | None:
    """For a failed case: the assistant's own answer (rejected) against the ideal one (chosen)."""
    evaluation = record.get("evaluation")
    if not evaluation or evaluation["passed"] or "answer" not in record:
        return None
    return {"messages": example["messages"][:-1], "chosen": example["messages"][-1]["content"],
            "rejected": record["answer"], "gate_violations": evaluation["gate_violations"],
            "case_id": record["case_id"]}
