"""Build a fictional corpus of up to 5,000 schemes with labelled questions.

    python scripts/build_scheme_corpus.py --output /tmp/schemes --schemes 5000

Every scheme has four sections (eligibility rules, how to apply, payment,
appeals), so 5,000 schemes are 20,000 sections. Each scheme is a unique
combination of a state, a group of workers and a benefit, with its own age
range, income limit and amounts.

The questions and their correct evidence are written from the same scheme
record, not from any search result, so they are independent labels:

    named      asks for a scheme's rules by its full name
    documents  asks how to apply; needs the rules AND the application section
    typo       the named question with keyboard typos
    described  describes the person and the need without naming the scheme
    absent     names a scheme in a state the corpus does not cover
    off_topic  has nothing to do with schemes

This is an engineering fixture. The schemes are invented; it shows whether the
right text reaches the model's prompt, not how real users phrase questions.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.perturbations import add_typos

STATES = ["Punjab", "Haryana", "Delhi", "Bihar", "Maharashtra", "Kerala", "Gujarat", "Odisha", "Assam",
          "Goa", "Sikkim", "Rajasthan", "Karnataka", "Telangana", "Jharkhand", "Tripura", "Manipur",
          "Mizoram", "Nagaland", "Meghalaya", "Uttarakhand", "Chhattisgarh", "Tamil Nadu",
          "West Bengal", "Madhya Pradesh"]
ABSENT_STATES = ["Uttar Pradesh", "Andhra Pradesh", "Himachal Pradesh", "Arunachal Pradesh", "Ladakh"]
# (name used in the scheme title, how a person describes their own work)
GROUPS = [("Weavers", "a weaver"), ("Fishers", "a fisher"), ("Street Vendors", "a street vendor"),
          ("Construction Workers", "a construction worker"), ("Farm Labourers", "a farm labourer"),
          ("Domestic Workers", "a domestic worker"), ("Potters", "a potter"), ("Carpenters", "a carpenter"),
          ("Tailors", "a tailor"), ("Rickshaw Pullers", "a rickshaw puller"),
          ("Brick Kiln Workers", "a brick kiln worker"), ("Salt Pan Workers", "a salt pan worker"),
          ("Beedi Rollers", "a beedi roller"), ("Tea Garden Workers", "a tea garden worker"),
          ("Sanitation Workers", "a sanitation worker"), ("Porters", "a porter"), ("Cobblers", "a cobbler"),
          ("Blacksmiths", "a blacksmith"), ("Dairy Farmers", "a dairy farmer"), ("Barbers", "a barber")]
# (name used in the scheme title, what the scheme gives, how a person describes the need)
BENEFITS = [("Old Age Pension", "a monthly pension after the age of 60", "money every month when I am old"),
            ("Housing Repair", "a grant to repair a damaged house", "fixing my damaged house"),
            ("Education Scholarship", "a scholarship for children's school fees", "paying my children's school fees"),
            ("Medical Treatment", "cashless hospital treatment", "my hospital bills"),
            ("Tool Kit", "a free kit of work tools", "buying tools for my work"),
            ("Accident Insurance", "insurance cover for accidents at work", "getting hurt at work"),
            ("Maternity Support", "cash support during pregnancy", "support during pregnancy"),
            ("Skill Training", "free skill training with a stipend", "learning new skills"),
            ("Working Capital Loan", "a low-interest working capital loan", "a small loan for my business"),
            ("Funeral Assistance", "assistance with funeral expenses", "funeral costs in my family")]
OFF_TOPIC = ["What is the capital of France?", "Tell me a joke about cats.", "How do I reset my phone password?",
             "Who won the cricket world cup in 2011?", "What is 17 multiplied by 23?",
             "Recommend a good film for the weekend.", "How far is the moon from the earth?",
             "Write a poem about the monsoon.", "Which is the tallest mountain in the world?",
             "How do I boil an egg?", "What time is it in Tokyo?", "Translate good morning into French.",
             "Why is the sky blue?", "Who wrote Hamlet?", "How many days are there in a leap year?",
             "What should I cook for dinner tonight?", "Explain how a bicycle gear works.",
             "Is it going to rain tomorrow?", "What is the boiling point of water?",
             "Suggest a name for my puppy.", "How do aeroplanes stay in the air?",
             "What is the square root of 144?", "Who painted the Mona Lisa?",
             "How do I tie a tie?", "Which planet is closest to the sun?",
             "What is the speed of light?", "How do I learn to swim?",
             "Give me a recipe for lemonade.", "What is the largest ocean?", "How do magnets work?"]
MAX_SCHEMES = len(STATES) * len(GROUPS) * len(BENEFITS)


def _bucket(i: int) -> int:
    """Ten equal buckets, each spread evenly over every state, group and benefit."""
    return (i // len(BENEFITS) + i) % 10


def _split(i: int) -> str:
    return "dev" if _bucket(i) == 3 else "test"


def scheme_sections(i: int) -> tuple[str, list[dict]]:
    """The scheme's name and its four sections. Index i picks a unique combination."""
    benefit, purpose, _ = BENEFITS[i % len(BENEFITS)]
    group, worker = GROUPS[(i // len(BENEFITS)) % len(GROUPS)]
    state = STATES[i // (len(BENEFITS) * len(GROUPS))]
    name = f"{state} {group} {benefit} Scheme"
    sid = f"scheme_{i:04d}"
    low, high = 18 + i % 7, 45 + (i * 7) % 20
    income, amount = 60000 + (i % 37) * 5000, 1000 + (i % 53) * 250
    parts = {
        "rules": ("eligibility rules",
                  f"{name}. Purpose: {purpose} for {group.lower()} in {state}.\n"
                  f"Eligibility rules:\n- The applicant must live in {state}.\n"
                  f"- The applicant must work as {worker}, registered with the {state} labour department.\n"
                  f"- Age must be between {low} and {high}.\n"
                  f"- Annual family income must be {income} rupees or below.\n"
                  f"- The applicant must not already receive {purpose} from another government scheme.\n"
                  "Final approval depends on official verification."),
        "application": ("how to apply",
                        f"How to apply for the {name}. Applicants to the {name} submit the application form at "
                        f"the {state} district labour office with an identity card, proof of residence in {state}, "
                        f"an income certificate and their registration card as {worker}. The {name} office issues "
                        f"an acknowledgement slip with a reference number. Forms are accepted until day {1 + i % 28} "
                        "of each month."),
        "payment": ("payment",
                    f"Payment under the {name}. The {name} pays {amount} rupees into the applicant's own bank "
                    f"account within {15 + i % 30} days of approval. The {name} benefit stops if the applicant "
                    f"moves out of {state}. Payment is subject to available funds."),
        "appeals": ("review and appeals",
                    f"Review and appeals for the {name}. An applicant refused under the {name} may appeal to the "
                    f"{state} labour commissioner within {30 + i % 60} days, quoting the reference number. "
                    f"The {name} appeal decision is final."),
    }
    return name, [{"chunk_id": f"{sid}_{section}", "scheme_id": sid, "section": section,
                   "title": f"{name} — {heading}", "text": text}
                  for section, (heading, text) in parts.items()]


def scheme_questions(i: int, name: str, rng, variants: bool) -> list[dict]:
    sid, split = f"scheme_{i:04d}", _split(i)
    _, _, need = BENEFITS[i % len(BENEFITS)]
    _, worker = GROUPS[(i // len(BENEFITS)) % len(GROUPS)]
    state = STATES[i // (len(BENEFITS) * len(GROUPS))]
    named = f"What are the eligibility rules for the {name}?"
    rules, apply = f"{sid}_rules", f"{sid}_application"

    def case(condition, question, gold):
        return {"query_id": f"{sid}_{condition}", "group_id": sid, "split": split, "condition": condition,
                "question": question, "gold_chunk_ids": gold, "required_chunk_ids": gold}

    cases = [case("named", named, [rules])]
    if variants:
        cases += [case("documents", f"Which documents do I need to apply for the {name}?", [rules, apply]),
                  case("typo", add_typos(named, 0.18, rng), [rules]),
                  case("described", f"I am {worker} from {state} and I need help with {need}. "
                                    "Can I get anything?", [rules])]
    return cases


def build(output: Path, schemes: int = MAX_SCHEMES, seed: int = 42) -> dict:
    if not 0 < schemes <= MAX_SCHEMES:
        raise ValueError(f"schemes must be between 1 and {MAX_SCHEMES}")
    rng = np.random.default_rng(seed)
    sections, questions = [], []
    # Spread the chosen schemes over every state, group and benefit.
    for i in sorted(int(x) for x in rng.choice(MAX_SCHEMES, schemes, replace=False)):
        name, parts = scheme_sections(i)
        sections += parts
        # Every scheme is asked for by name; a tenth of each split also gets the harder variants.
        questions += scheme_questions(i, name, rng, variants=_bucket(i) in (0, 3))
    for n, state in enumerate(ABSENT_STATES):
        for m in range(60):
            group, benefit = GROUPS[(n * 7 + m) % len(GROUPS)][0], BENEFITS[m % len(BENEFITS)][0]
            questions.append({"query_id": f"absent_{n}_{m}", "group_id": f"absent_{n}_{m}",
                              "split": _split(m), "condition": "absent",
                              "question": f"What are the eligibility rules for the {state} {group} {benefit} Scheme?",
                              "gold_chunk_ids": [], "expect_abstain": True})
    for n, question in enumerate(OFF_TOPIC):
        questions.append({"query_id": f"off_topic_{n}", "group_id": f"off_topic_{n}",
                          "split": "dev" if n % 3 == 0 else "test", "condition": "off_topic",
                          "question": question, "gold_chunk_ids": [], "expect_abstain": True})
    order = rng.permutation(len(sections))  # sections of one scheme are not neighbours on disk
    output.mkdir(parents=True, exist_ok=True)
    (output / "corpus.jsonl").write_text(
        "".join(json.dumps(sections[i], ensure_ascii=False) + "\n" for i in order), encoding="utf-8")
    (output / "questions.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
    return {"schemes": schemes, "sections": len(sections), "questions": len(questions),
            "corpus": str(output / "corpus.jsonl"), "questions_file": str(output / "questions.jsonl"),
            "kind": "fictional engineering fixture; labels written from the scheme records"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schemes", type=int, default=MAX_SCHEMES)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.schemes), indent=2))
