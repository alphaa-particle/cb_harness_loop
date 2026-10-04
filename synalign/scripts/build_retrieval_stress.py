"""Build a reproducible fictional welfare corpus with labelled difficult queries.

Labels are authored fixtures, not predictions from the retriever. This is an
engineering stress test, not a sample of real schemes or real user traffic.
"""

import argparse
import json
from pathlib import Path
import random
import shutil


def build(output: Path, chunks: int = 5000):
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/retrieval"
    records = [json.loads(line) for line in (fixture / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    if chunks < len(records):
        raise ValueError(f"At least {len(records)} chunks are needed to retain all labelled evidence")
    states = ["Assam", "Goa", "Sikkim", "Odisha"]
    benefits = ["college textbook reimbursement", "technical diploma tuition fees",
                "kidney dialysis treatment", "cataract eye surgery", "damaged house roof repairs",
                "wheelchairs for locomotor disability", "construction worker accident compensation",
                "monthly old-age pension", "sewing machine tailoring equipment",
                "street vendor working-capital loans", "drip irrigation installation",
                "college student hostel accommodation", "pregnancy nutrition and antenatal care",
                "storm-damaged fishing boat repairs"]
    for i in range(chunks - len(records)):
        state, benefit = states[i % len(states)], benefits[(i // len(states)) % len(benefits)]
        cid = f"distractor_{i:05d}"
        title = f"Fictional {state} local support {i:05d}"
        records.append({"chunk_id": cid, "scheme_id": cid, "title": title,
                        "text": f"{title}. This fictional programme provides {benefit} only to residents "
                        f"of {state}, municipal zone {i:05d}. Residents of other states are not eligible. "
                        "Applicants submit identity documents, local residence proof and relevant "
                        "certificates at their local office. "
                        f"Annual family income must be below {100000 + i % 30 * 10000} rupees. "
                        "Benefits are subject to verification; approval is not guaranteed."})
    random.Random(147).shuffle(records)
    output.mkdir(parents=True, exist_ok=True)
    (output / "corpus.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    shutil.copyfile(fixture / "questions.jsonl", output / "questions.jsonl")
    return {"chunks": len(records), "corpus": str(output / "corpus.jsonl"),
            "questions": str(output / "questions.jsonl"), "kind": "fictional engineering stress test"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.chunks), indent=2))
