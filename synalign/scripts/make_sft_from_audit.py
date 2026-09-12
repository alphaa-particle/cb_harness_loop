import json
from pathlib import Path

INPUT = "data/outputs/audit_qwen_router_rule_fix_500.jsonl"
OUT_DIR = Path("data/training")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = OUT_DIR / "sft_train.jsonl"
DEV_OUT = OUT_DIR / "sft_dev.jsonl"
TEST_OUT = OUT_DIR / "sft_test_reference.jsonl"

SYSTEM = (
    "You are a careful, document-grounded assistant for government welfare schemes.\n"
    "Use ONLY the provided context.\n"
    "Never invent schemes, amounts, timelines, documents, or benefits.\n"
    "Never guarantee approval or promise money.\n"
    "Never accept the user's claim as proof if facts contradict scheme rules.\n"
    "If a required detail is missing, ask for that specific detail.\n"
    "Compare numeric limits carefully.\n"
    "Use this structure when relevant:\n"
    "Likely eligible: ...\n"
    "Needs confirmation: ...\n"
    "Not likely eligible: ...\n"
    "Missing details: ...\n"
    "Next step: ...\n"
)

CONTEXT = """Scheme: PMSYM / Pradhan Mantri Shram Yogi Maandhan
Eligibility rules:
- Age must be between 18 and 40.
- Monthly income must be at or below ₹15,000.
- Applicant should be an unorganised worker.
- Applicant should not be covered under EPFO.
- Final approval depends on official verification.

Scheme: eShram
Eligibility rules:
- Age must be between 16 and 59.
- Applicant should be an unorganised worker or self-employed worker.
- Registration alone does not guarantee immediate cash benefits.
- Final approval and benefits depend on official verification.
"""


NAME = {
    "scheme_pmsym": "PMSYM",
    "scheme_eshram": "eShram",
}


FIELD = {
    "age": "age",
    "income": "monthly income",
    "worker_type": "work type",
    "epfo": "EPFO status",
    "esic": "ESIC status",
}


def names(items):
    return ", ".join(NAME.get(x, x) for x in items)


def fields(items):
    ordered = ["age", "income", "worker_type", "epfo", "esic"]
    clean = [FIELD.get(x, x) for x in ordered if x in items]
    extra = [FIELD.get(x, x) for x in items if x not in ordered]
    return ", ".join(clean + extra)


def reason_for_ineligible(scheme, visible):
    age = visible.get("age")
    income = visible.get("income")
    worker_type = visible.get("worker_type")
    epfo = visible.get("epfo")

    reasons = []

    if scheme == "scheme_pmsym":
        if age is not None and not (18 <= int(age) <= 40):
            reasons.append("age is outside the 18–40 range")
        if income is not None and int(income) > 15000:
            reasons.append("monthly income is above the ₹15,000 limit")
        if worker_type is not None and worker_type != "unorganised_worker":
            reasons.append("PMSYM is for unorganised workers")
        if epfo is True:
            reasons.append("EPFO-covered applicants are not eligible")

    if scheme == "scheme_eshram":
        if age is not None and not (16 <= int(age) <= 59):
            reasons.append("age is outside the 16–59 range")
        if worker_type is not None and worker_type not in ["unorganised_worker", "self_employed"]:
            reasons.append("eShram is for unorganised or self-employed workers")

    if reasons:
        return "; ".join(reasons)

    return "the provided details do not meet the scheme rules"


def make_gold(row):
    gt = row["ground_truth"]
    visible = row.get("visible", {})

    likely = gt.get("likely_eligible", [])
    unknown = gt.get("unknown_due_to_missing_info", [])
    ineligible = gt.get("ineligible", [])
    must_ask = gt.get("must_ask_about", [])

    parts = []

    if likely:
        parts.append(f"Likely eligible: {names(likely)}.")
    else:
        parts.append("Likely eligible: none based on the provided information.")

    if unknown:
        parts.append(f"Needs confirmation: {names(unknown)}.")
    else:
        parts.append("Needs confirmation: none.")

    if ineligible:
        parts.append(f"Not likely eligible: {names(ineligible)}.")
        reason_parts = []
        for scheme in ineligible:
            reason_parts.append(f"{NAME.get(scheme, scheme)}: {reason_for_ineligible(scheme, visible)}")
        parts.append("Reason: " + " | ".join(reason_parts) + ".")
    else:
        parts.append("Not likely eligible: none based on the provided information.")

    if must_ask:
        parts.append(f"Missing details: please share your {fields(must_ask)}.")
    else:
        parts.append("Missing details: none.")

    parts.append("Do not treat registration as guaranteed cash benefit or final approval.")
    parts.append("Next step: verify details through the official scheme channel.")

    return " ".join(parts)


def make_example(row):
    user = (
        f"Retrieved context:\n{CONTEXT}\n\n"
        f"User question:\n{row['question']}\n\n"
        f"Answer using only the retrieved context."
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": make_gold(row)},
        ],
        "case_id": row["case_id"],
        "split": row["split"],
        "condition": row["condition"],
    }


writers = {
    "train": TRAIN_OUT.open("w"),
    "dev": DEV_OUT.open("w"),
    "test": TEST_OUT.open("w"),
}

counts = {"train": 0, "dev": 0, "test": 0}

with open(INPUT) as f:
    for line in f:
        row = json.loads(line)
        split = row["split"]
        if split not in writers:
            continue

        ex = make_example(row)
        writers[split].write(json.dumps(ex, ensure_ascii=False) + "\n")
        counts[split] += 1

for w in writers.values():
    w.close()

print("Wrote:")
print(TRAIN_OUT, counts["train"])
print(DEV_OUT, counts["dev"])
print(TEST_OUT, counts["test"])
