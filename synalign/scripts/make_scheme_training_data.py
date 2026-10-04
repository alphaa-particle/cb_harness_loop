"""Fine-tuning examples that teach the answering model the skills it gets wrong.

    python scripts/make_scheme_training_data.py --output-dir data/training/india_schemes

The evaluation showed the small model's main errors with the right rules in
front of it: saying "yes" to someone outside an age or income limit, and
answering when the information shown is about a different scheme. This writes
examples of exactly those, each in the prompt layout the chatbot really uses
(engine/enforcement.py), with the answer worked out from the scheme's checked
limits:

    eligibility   "I am 52. Can I join X?"  ->  "No. X is for people aged 18 to 40, and 52 is more than 40."
                  in English and Hindi, many ages and incomes on both sides of each limit
    not_here      a question about one scheme shown another scheme's rules  ->  "I could not find this ..."

Only schemes in the dev split are used, so the test schemes the results are
reported on are never trained on. The rules are used to write training answers,
never to answer real questions. Train with scripts/train_qwen_lora_sft.py on a
GPU machine, then measure the trained model on the test split and keep it only
if scripts/compare_runs.py says KEEP.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import DOMAINS_DIR, ROOT_DIR
from engine.domain_pack import DomainPack
from engine.enforcement import build_messages
from engine.scheme_records import load_records, record_chunks
from engine.schemas import RetrievedChunk
from scripts.build_eval_questions import split_of

NOT_HERE = {"en": "I could not find this in the scheme information.",
            "hi": "मुझे योजना की जानकारी में यह नहीं मिला।"}


def names(record):
    en = record.get("short_name") or record["name"]
    return en, (record.get("names_hi") or [en])[0]


def age_examples(record, rng):
    rules = record.get("rules") or {}
    low, high = rules.get("age_min"), rules.get("age_max")
    if not isinstance(low, int) and not isinstance(high, int):
        return []
    en, hi = names(record)
    if isinstance(low, int) and isinstance(high, int):
        rule_en, rule_hi = f"is for people aged {low} to {high}", f"{low} से {high} वर्ष की उम्र के लोगों के लिए है"
    elif isinstance(low, int):
        rule_en, rule_hi = f"is for people aged {low} or more", f"{low} वर्ष या उससे अधिक उम्र के लोगों के लिए है"
    else:
        rule_en, rule_hi = f"is for people aged up to {high}", f"{high} वर्ष तक की उम्र के लोगों के लिए है"
    ages = {}
    if isinstance(high, int):
        for _ in range(4):
            ages[high + rng.randint(1, 25)] = "over"
        ages[high] = "inside"
    if isinstance(low, int) and low > 1:
        for _ in range(2):
            ages[max(1, low - rng.randint(1, 8))] = "under"
        ages[low] = "inside"
    inside_top = high if isinstance(high, int) else (low or 18) + 40
    for _ in range(4):
        ages.setdefault(rng.randint(low or 1, inside_top), "inside")
    out = []
    for age, side in sorted(ages.items()):
        if side == "over":
            answers = (f"No. {en} {rule_en}, and {age} is more than {high}.",
                       f"नहीं। {hi} {rule_hi}, और {age} वर्ष {high} से अधिक है।")
        elif side == "under":
            answers = (f"No. {en} {rule_en}, and {age} is less than {low}.",
                       f"नहीं। {hi} {rule_hi}, और {age} वर्ष {low} से कम है।")
        else:
            answers = (f"Yes, on age: {en} {rule_en}, and {age} meets that. The other conditions in the rules must also be met.",
                       f"हाँ, उम्र के हिसाब से: {hi} {rule_hi}, और {age} वर्ष इस शर्त को पूरा करता है। "
                       "नियमों की बाकी शर्तें भी पूरी होनी चाहिए।")
        out.append(("en", rng.choice([f"I am {age} years old. Can I join {en}?", f"My age is {age}. Am I eligible for {en}?"]),
                    answers[0]))
        out.append(("hi", rng.choice([f"मेरी उम्र {age} साल है। क्या मैं {hi} में शामिल हो सकता हूँ?",
                                      f"मैं {age} साल का हूँ। क्या मुझे {hi} मिल सकती है?"]), answers[1]))
    return out


def income_examples(record, rng):
    rules, out = record.get("rules") or {}, []
    en, hi = names(record)
    for key, en_period, hi_period in (("monthly_income_max", "a month", "महीने"), ("annual_income_max", "a year", "साल")):
        limit = rules.get(key)
        if not isinstance(limit, int) or limit <= 0:
            continue
        over, under = int(limit * rng.uniform(1.1, 2.0)), int(limit * rng.uniform(0.3, 0.95))
        out += [("en", f"I earn Rs {over:,} {en_period}. Am I eligible for {en}?",
                 f"No. {en} has an income limit of Rs {limit:,} {en_period}, and Rs {over:,} is more than that."),
                ("hi", f"मेरी आय {hi_period} में {over} रुपये है। क्या मैं {hi} के लिए पात्र हूँ?",
                 f"नहीं। {hi} की आय सीमा {hi_period} में {limit} रुपये है, और {over} रुपये इससे अधिक है।"),
                ("en", f"I earn Rs {under:,} {en_period}. Am I eligible for {en}?",
                 f"Yes, on income: {en} has a limit of Rs {limit:,} {en_period}, and Rs {under:,} is within it. "
                 "The other conditions in the rules must also be met."),
                ("hi", f"मेरी आय {hi_period} में {under} रुपये है। क्या मैं {hi} के लिए पात्र हूँ?",
                 f"हाँ, आय के हिसाब से: {hi} की आय सीमा {hi_period} में {limit} रुपये है, और {under} रुपये इसके "
                 "अंदर है। नियमों की बाकी शर्तें भी पूरी होनी चाहिए।")]
    return out


def build(domain: str, rng: random.Random) -> list[dict]:
    pack = DomainPack(domain)
    records = [r for r in load_records(DOMAINS_DIR / domain / "schemes") if split_of(r["id"]) == "dev"]
    chunks = {r["id"]: record_chunks(r) for r in records}

    def evidence(record_id):
        """What search would put in the prompt for this scheme: its rules first, then its overview."""
        parts = chunks[record_id]
        rules = [c for c in parts if c["section"] == "rules"][:2]
        return [RetrievedChunk(c["chunk_id"], c["text"], 1.0, c["title"], c["scheme_id"], c["section"], c["source"])
                for c in rules + [c for c in parts if c["section"] == "overview"]][:3]

    examples = []
    for record in records:
        shown = evidence(record["id"])
        for language, question, answer in age_examples(record, rng) + income_examples(record, rng):
            examples.append({"messages": [*build_messages(pack, question, shown), {"role": "assistant", "content": answer}],
                             "kind": "eligibility", "language": language, "scheme_id": record["id"]})
        en, hi = names(record)
        for other in rng.sample([r for r in records if r["id"] != record["id"]], k=min(3, len(records) - 1)):
            for language, question in (("en", rng.choice([f"What are the eligibility rules for {en}?",
                                                           f"How much money does {en} give?"])),
                                       ("hi", rng.choice([f"{hi} के लिए कौन पात्र है?", f"{hi} में कितना पैसा मिलता है?"]))):
                examples.append({"messages": [*build_messages(pack, question, evidence(other["id"])),
                                              {"role": "assistant", "content": NOT_HERE[language]}],
                                 "kind": "not_here", "language": language, "scheme_id": record["id"]})
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", default="india_schemes")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "data" / "training" / "india_schemes")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    examples = build(args.domain, rng)
    rng.shuffle(examples)
    # Hold back four schemes to watch the loss on schemes the model was not trained on.
    schemes = sorted({e["scheme_id"] for e in examples}, key=lambda sid: hashlib.sha1(sid.encode()).hexdigest())
    held = set(schemes[:4])
    train = [e for e in examples if e["scheme_id"] not in held]
    dev = [e for e in examples if e["scheme_id"] in held]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sft_train.jsonl", train), ("sft_dev.jsonl", dev)):
        (args.output_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                            encoding="utf-8")
    kinds = {}
    for e in examples:
        kinds[f"{e['kind']}/{e['language']}"] = kinds.get(f"{e['kind']}/{e['language']}", 0) + 1
    print(json.dumps({"schemes": len({e['scheme_id'] for e in examples}), "train": len(train), "dev": len(dev),
                      "by_kind": kinds, "output_dir": str(args.output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
