"""Write the frozen test questions for the real-schemes corpus.

    python scripts/build_eval_questions.py            # domains/india_schemes/evaluation/questions.jsonl

Every question carries its expected answer, worked out from the checked scheme
records, never from what the search returns:

    fact         a scheme's age limit, amount, premium...  in English, Hindi, and Hindi typed in Latin letters
    typo         the English fact question with keyboard typos
    situation    a person describing their need without naming the scheme
    eligibility  "I am 52. Can I join APY?"  expected yes-or-unsure, or a clear no, from the scheme's limits
    absent       a real scheme that is NOT in the corpus: the right answer is "not found"
    off_topic    nothing to do with schemes: the right answer is "not found"

Questions are split into dev (for tuning) and test (for the reported numbers) by
scheme, so no scheme is in both. The file's SHA-256 is printed so results can
name exactly which questions they were measured on. The fact and situation
questions were drafted by the research agents from official pages; they have
not been checked by a native Hindi speaker.

    python scripts/build_eval_questions.py --usecases  # evaluation/usecase_questions.jsonl

The use-case questions cover what people come to the chatbot for beyond single facts. They
are written from the structure of the scheme files with a few fixed English phrasings, never
per scheme, and each names the sections its answer needs (gold_groups; "all" of them, or
"any" one for category questions):

    apply              "How do I apply for X?"                    the how-to-apply section
    documents          "What documents do I need for X?"          the documents section
    benefits           "What benefits does X give?"               the benefits section
    exclusions         "Who is not eligible for X?"               the rules part that lists exclusions
    apply_eligibility  "Who can apply for X, and how?"            the rules AND the how-to-apply section
    compare            "What is the difference between X and Y?"  something from X AND from Y
    category           "Which schemes help with <a need>?"        at least one scheme of that kind

Compare questions pair two schemes of the same kind and the same split. Category questions are
split by category (every third, alphabetically, is dev); their schemes come from both splits, as a
real catalogue question would.
The mixed categories ("social_assistance", "other") get no category question.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import DOMAINS_DIR
from engine.perturbations import add_typos
from engine.scheme_records import load_records, record_chunks, record_questions

# Real schemes that are deliberately not in the corpus (state schemes, and central schemes left
# out because they are closed or their status is unclear). Asking about them must not get an answer
# about some other scheme.
ABSENT = [
    ("ladli_behna", "Ladli Behna Yojana of Madhya Pradesh", "मध्य प्रदेश की लाड़ली बहना योजना"),
    ("majhi_ladki_bahin", "Mukhyamantri Majhi Ladki Bahin Yojana of Maharashtra", "महाराष्ट्र की मुख्यमंत्री माझी लाडकी बहीण योजना"),
    ("lakshmir_bhandar", "Lakshmir Bhandar scheme of West Bengal", "पश्चिम बंगाल की लक्ष्मीर भंडार योजना"),
    ("gruha_lakshmi", "Gruha Lakshmi scheme of Karnataka", "कर्नाटक की गृह लक्ष्मी योजना"),
    ("magalir_urimai", "Kalaignar Magalir Urimai Thogai scheme of Tamil Nadu", "तमिलनाडु की कलैग्नर मगलिर उरिमै योजना"),
    ("kalia", "KALIA scheme for farmers in Odisha", "ओडिशा की कालिया योजना"),
    ("rythu_bharosa", "Rythu Bharosa scheme of Telangana", "तेलंगाना की रायथु भरोसा योजना"),
    ("orunodoi", "Orunodoi scheme of Assam", "असम की ओरुनोदोई योजना"),
    ("mahtari_vandan", "Mahtari Vandan Yojana of Chhattisgarh", "छत्तीसगढ़ की महतारी वंदन योजना"),
    ("kanya_sumangala", "Mukhyamantri Kanya Sumangala Yojana of Uttar Pradesh", "उत्तर प्रदेश की मुख्यमंत्री कन्या सुमंगला योजना"),
    ("ladli_laxmi", "Ladli Laxmi Yojana of Madhya Pradesh", "मध्य प्रदेश की लाड़ली लक्ष्मी योजना"),
    ("chiranjeevi", "Mukhyamantri Chiranjeevi Swasthya Bima Yojana of Rajasthan", "राजस्थान की मुख्यमंत्री चिरंजीवी स्वास्थ्य बीमा योजना"),
    ("kanyashree", "Kanyashree Prakalpa of West Bengal", "पश्चिम बंगाल की कन्याश्री प्रकल्प योजना"),
    ("subhadra", "Subhadra Yojana of Odisha", "ओडिशा की सुभद्रा योजना"),
    ("stand_up_india", "Stand-Up India scheme", "स्टैंड-अप इंडिया योजना"),
    ("pm_kusum", "PM-KUSUM solar pump scheme", "पीएम-कुसुम सोलर पंप योजना"),
    ("pmfme", "PM Formalisation of Micro Food Processing Enterprises scheme", "पीएम सूक्ष्म खाद्य उद्योग उन्नयन योजना"),
    ("pm_poshan", "PM POSHAN mid-day meal scheme", "पीएम पोषण मध्याह्न भोजन योजना"),
]
OFF_TOPIC = [
    "What is the capital of France?", "Tell me a joke about cats.", "How do I reset my phone password?",
    "Who won the cricket world cup in 2011?", "What is 17 multiplied by 23?", "Recommend a film for the weekend.",
    "How far is the moon from the earth?", "Write a poem about the monsoon.", "How do I boil an egg?",
    "Why is the sky blue?", "आज मौसम कैसा रहेगा?", "मुझे एक चुटकुला सुनाओ।", "भारत की राजधानी क्या है?",
    "क्रिकेट मैच का स्कोर क्या है?", "चाय बनाने का तरीका बताओ।", "aaj mausam kaisa hai?",
    "mujhe ek gaana sunao", "cricket ka score kya hai?", "Which planet is closest to the sun?", "Who painted the Mona Lisa?",
]


def split_of(key: str) -> str:
    """Three in ten schemes go to dev, the rest to test; fixed by the scheme id, not by chance."""
    return "dev" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10 < 3 else "test"


def eligibility_questions(record: dict, chunk_ids: list[str]) -> list[dict]:
    rules, name = record.get("rules") or {}, record.get("short_name") or record["name"]
    hindi = (record.get("names_hi") or [name])[0]
    cases = []
    if isinstance(rules.get("age_max"), int):
        cases += [("age_over", rules["age_max"] + 5, "no")]
    if isinstance(rules.get("age_min"), int) and rules["age_min"] >= 10:
        cases += [("age_under", rules["age_min"] - 3, "no")]
    if isinstance(rules.get("age_min"), int) or isinstance(rules.get("age_max"), int):
        low, high = rules.get("age_min", 18), rules.get("age_max", (rules.get("age_min", 18) + 20))
        cases += [("age_inside", (low + high) // 2, "not_no")]
    questions = []
    for kind, age, gold in cases:
        for language, text in (("en", f"I am {age} years old. Can I apply for {name}?"),
                               ("hi", f"मेरी उम्र {age} साल है। क्या मैं {hindi} के लिए आवेदन कर सकता हूँ?")):
            questions.append({"type": "eligibility", "language": language, "question": text, "variant": kind,
                              "facts_given": {"age": age}, "gold_verdict": gold})
    for key, period in (("monthly_income_max", "monthly"), ("annual_income_max", "yearly")):
        limit = rules.get(key)
        if not isinstance(limit, int):
            continue
        for kind, income, gold in (("income_over", int(limit * 1.5), "no"), ("income_inside", int(limit * 0.5), "not_no")):
            en = (f"My monthly income is Rs {income:,}. Am I eligible for {name}?" if period == "monthly" else
                  f"My family's yearly income is Rs {income:,}. Am I eligible for {name}?")
            hi = (f"मेरी मासिक आय {income} रुपये है। क्या मैं {hindi} के लिए पात्र हूँ?" if period == "monthly" else
                  f"मेरे परिवार की सालाना आय {income} रुपये है। क्या मैं {hindi} के लिए पात्र हूँ?")
            for language, text in (("en", en), ("hi", hi)):
                questions.append({"type": "eligibility", "language": language, "question": text, "variant": kind,
                                  "facts_given": {key: income}, "gold_verdict": gold})
    if rules.get("gender") == "female":
        for language, text in (("en", f"I am a man. Can I apply for {name}?"),
                               ("hi", f"मैं पुरुष हूँ। क्या मैं {hindi} के लिए आवेदन कर सकता हूँ?")):
            questions.append({"type": "eligibility", "language": language, "question": text, "variant": "gender",
                              "facts_given": {"gender": "male"}, "gold_verdict": "no"})
    return [{**q, "gold_chunk_ids": chunk_ids} for q in questions]


# One plain description of the need each kind of scheme serves; used only for category questions.
CATEGORY_NEEDS = {
    "pension": "a monthly pension", "insurance": "insurance for my life, an accident or my crops",
    "health": "free or cheaper medical treatment", "housing": "help to build or buy a house",
    "agriculture": "support for my farming or fishing", "women_child": "support for a pregnant woman, a mother or a girl child",
    "education": "a scholarship or a loan for my studies", "skills_employment": "job training or work",
    "credit_enterprise": "a loan to start or grow a small business", "food": "free or cheap food grains or nutrition",
    "energy": "a cooking gas connection or solar power for my home",
    "financial_inclusion": "a bank account or a safe place to save money",
}
PHRASINGS = {
    "apply": ("How do I apply for {name}?", "Where can I register for {name}, and what is the process?"),
    "documents": ("What documents do I need for {name}?", "Which papers should I keep ready to apply for {name}?"),
    "benefits": ("What benefits does {name} give?", "How much money or help does {name} provide?"),
    "exclusions": ("Who is not eligible for {name}?", "Who cannot apply for {name}?"),
    "apply_eligibility": ("Who can apply for {name}, and how do I apply?",),
    "compare": ("What is the difference between {a} and {b}?", "Should I go for {a} or {b}? How are they different?"),
    "category": ("Which government schemes can help me get {need}?", "I need {need}. What schemes are there for me?"),
}


def usecase_questions(records: list[dict]) -> list[dict]:
    """Questions for the uses beyond single facts, with the sections each answer needs."""
    rows = []

    def add(qid, group, split, kind, text, groups, need, schemes):
        rows.append({"query_id": qid, "scheme_id": schemes[0] if len(schemes) == 1 else "", "group_id": group,
                     "split": split, "type": kind, "language": "en", "question": text, "expect_abstain": False,
                     "gold_groups": groups, "need": need, "gold_chunk_ids": sorted({c for g in groups for c in g}),
                     "gold_answer": {"schemes": schemes}})

    chunks = {r["id"]: record_chunks(r) for r in records}
    for record in records:
        sid, parts = record["id"], chunks[record["id"]]
        names = (record["name"], record.get("short_name") or record["name"])
        by_section = {}
        for c in parts:
            by_section.setdefault(c["section"], []).append(c["chunk_id"])
        exclusions = [s.strip() for s in record.get("sections", {}).get("exclusions") or []]
        excl_ids = [c["chunk_id"] for c in parts if c["section"] == "rules" and any(e in c["text"] for e in exclusions)]
        wanted = {"apply": by_section.get("application"), "documents": by_section.get("documents"),
                  "benefits": by_section.get("benefits"), "exclusions": excl_ids}
        base = {"group": sid, "split": split_of(sid)}
        for kind, gold in wanted.items():
            if not gold:
                continue
            for n, phrasing in enumerate(PHRASINGS[kind]):
                add(f"{sid}__{kind}_{n + 1}", base["group"], base["split"], kind,
                    phrasing.format(name=names[n % 2]), [gold], "all", [sid])
        if by_section.get("rules") and by_section.get("application"):
            add(f"{sid}__apply_eligibility", sid, base["split"], "apply_eligibility",
                PHRASINGS["apply_eligibility"][0].format(name=names[1]),
                [by_section["rules"], by_section["application"]], "all", [sid])
    kinds = {}
    for record in records:
        kinds.setdefault((record["category"], split_of(record["id"])), []).append(record)
    for (category, split), members in sorted(kinds.items()):
        members = sorted(members, key=lambda r: r["id"])
        for a, b in zip(members[::2], members[1::2]):
            for n, phrasing in enumerate(PHRASINGS["compare"]):
                pick = (lambda r: r.get("short_name") or r["name"]) if n == 0 else (lambda r: r["name"])
                add(f"compare__{a['id']}__{b['id']}_{n + 1}", f"compare__{a['id']}__{b['id']}", split, "compare",
                    phrasing.format(a=pick(a), b=pick(b)),
                    [[c["chunk_id"] for c in chunks[a["id"]]], [c["chunk_id"] for c in chunks[b["id"]]]],
                    "all", [a["id"], b["id"]])
    # Twelve categories are too few for the 3-in-10 hash to be sure of giving dev any, so every
    # third category in alphabetical order is dev instead.
    for n_cat, category in enumerate(sorted(CATEGORY_NEEDS)):
        need = CATEGORY_NEEDS[category]
        members = sorted(r["id"] for r in records if r["category"] == category)
        if not members:
            continue
        for n, phrasing in enumerate(PHRASINGS["category"]):
            add(f"category__{category}_{n + 1}", f"category__{category}", "dev" if n_cat % 3 == 0 else "test",
                "category", phrasing.format(need=need), [[c["chunk_id"] for c in chunks[m]] for m in members],
                "any", members)
    return rows


def build(domain: str, include_unverified: bool) -> list[dict]:
    rng = np.random.default_rng(7)
    rows = []
    for record in load_records(DOMAINS_DIR / domain / "schemes"):
        if record["review"]["status"] != "verified" and not include_unverified:
            continue
        sid = record["id"]
        base = {"scheme_id": sid, "group_id": sid, "split": split_of(sid), "expect_abstain": False}
        rules_ids = [c["chunk_id"] for c in record_chunks(record) if c["section"] == "rules"]
        all_ids = [c["chunk_id"] for c in record_chunks(record)]
        for q in record_questions(record):
            rows.append({**base, "query_id": q["query_id"], "type": "fact", "language": q["language"],
                         "question": q["question"], "gold_chunk_ids": q["gold_chunk_ids"],
                         "gold_answer": {"value": q["answer"], "accept": q["accept"]}})
            if q["language"] == "en":
                rows.append({**rows[-1], "query_id": q["query_id"].replace("__en", "__typo"), "type": "typo",
                             "question": add_typos(q["question"], 0.18, rng)})
        for fact in record.get("facts") or []:
            if fact.get("status") == "verified" and fact.get("question_hi_latn"):
                gold = next(q for q in record_questions(record) if q["fact"] == fact["key"])
                rows.append({**base, "query_id": f"{sid}__{fact['key']}__hi_latn", "type": "fact", "language": "hi_latn",
                             "question": fact["question_hi_latn"], "gold_chunk_ids": gold["gold_chunk_ids"],
                             "gold_answer": {"value": gold["answer"], "accept": gold["accept"]}})
        for n, situation in enumerate(record.get("situations") or [], 1):
            rows.append({**base, "query_id": f"{sid}__situation_{n}", "type": "situation",
                         "language": situation["language"], "question": situation["question"],
                         "gold_chunk_ids": all_ids, "gold_answer": {"scheme": sid}})
        for n, q in enumerate(eligibility_questions(record, rules_ids), 1):
            rows.append({**base, "query_id": f"{sid}__eligibility_{n}", **q})
    for key, en, hi in ABSENT:
        for language, text in (("en", f"What are the eligibility rules for the {en}?"),
                               ("hi", f"{hi} के लिए कौन पात्र है?")):
            rows.append({"query_id": f"absent__{key}__{language}", "scheme_id": "", "group_id": f"absent__{key}",
                         "split": split_of(f"absent__{key}"), "type": "absent", "language": language, "question": text,
                         "gold_chunk_ids": [], "expect_abstain": True})
    for n, text in enumerate(OFF_TOPIC):
        rows.append({"query_id": f"off_topic__{n}", "scheme_id": "", "group_id": f"off_topic__{n}",
                     "split": split_of(f"off_topic__{n}"), "type": "off_topic",
                     "language": "hi" if any("ऀ" <= ch <= "ॿ" for ch in text) else "en",
                     "question": text, "gold_chunk_ids": [], "expect_abstain": True})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", default="india_schemes")
    parser.add_argument("--verified-only", action="store_true",
                        help="Leave out schemes marked needs_review (must match how the corpus was built)")
    parser.add_argument("--usecases", action="store_true",
                        help="Write the use-case questions (usecase_questions.jsonl) instead of questions.jsonl")
    args = parser.parse_args()
    if args.usecases:
        records = [r for r in load_records(DOMAINS_DIR / args.domain / "schemes")
                   if not args.verified_only or r["review"]["status"] == "verified"]
        rows = usecase_questions(records)
        out = DOMAINS_DIR / args.domain / "evaluation" / "usecase_questions.jsonl"
    else:
        rows = build(args.domain, include_unverified=not args.verified_only)
        out = DOMAINS_DIR / args.domain / "evaluation" / "questions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts = {}
    for r in rows:
        counts.setdefault(r["split"], {}).setdefault(r["type"], 0)
        counts[r["split"]][r["type"]] += 1
    print(json.dumps({"questions": len(rows), "by_split_and_type": counts, "file": str(out),
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
