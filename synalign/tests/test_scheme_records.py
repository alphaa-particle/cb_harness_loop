"""Scheme records: validation catches wrong or unsupported facts; chunks are built from the records."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.corpus import load_corpus
from engine.scheme_records import (build_corpus, load_records, record_chunks, record_questions,
                                   validate_record, validate_records)

RECORD = {
    "id": "loom_grant",
    "name": "Loom Renewal Grant",
    "short_name": "LRG",
    "names_hi": ["करघा नवीकरण अनुदान"],
    "aliases": ["kargha anudan"],
    "level": "state",
    "state": "Odisha",
    "ministry": "Department of Handlooms (fictional test record)",
    "category": "credit_enterprise",
    "summary": "A one-time grant for handloom weavers to replace a worn-out loom.",
    "sections": {
        "eligibility": ["The weaver must be between 21 and 50 years old.",
                        "Annual family income must be Rs 90,000 or less."],
        "exclusions": ["Weavers who received the grant in the last five years."],
        "benefits": ["A grant of Rs 4,000 paid into the weaver's bank account."],
        "how_to_apply": ["Apply at the block handloom office with the form."],
        "documents": ["Weaver identity card.", "Bank passbook."],
    },
    "rules": {"age_min": 21, "age_max": 50, "annual_income_max": 90000, "gender": "any"},
    "facts": [
        {"key": "age_max", "statement": "The upper age limit is 50.", "value": "50",
         "question": "What is the upper age limit for the Loom Renewal Grant?",
         "question_hi": "करघा नवीकरण अनुदान के लिए अधिकतम आयु क्या है?", "source": 1, "status": "verified"},
        {"key": "grant_amount", "statement": "The grant is Rs 4,000.", "value": "4,000", "accept": ["4000"],
         "question": "How much is the Loom Renewal Grant?", "source": 1, "status": "verified"},
    ],
    "sources": [{"title": "Fictional test page", "url": "https://example.org/loom", "accessed": "2026-10-03"}],
    "review": {"status": "verified", "checked_on": "2026-10-03"},
}


def with_file(record):
    return {**copy.deepcopy(record), "_file": f"{record.get('id')}.yaml"}


class SchemeRecordTests(unittest.TestCase):
    def errors(self, change):
        record = with_file(RECORD)
        change(record)
        return validate_record(record)[0]

    def test_a_complete_record_has_no_problems(self):
        self.assertEqual(validate_record(with_file(RECORD)), ([], []))

    def test_a_rule_number_must_be_stated_in_the_written_rules(self):
        errors = self.errors(lambda r: r["rules"].update(age_max=45))
        self.assertEqual(len(errors), 1)
        self.assertIn("rules.age_max is 45 but that number is not in the eligibility statements", errors[0])
        self.assertIn("age_min is above", " ".join(self.errors(lambda r: r["rules"].update(age_min=50, age_max=21))))
        self.assertIn("not a known rule", " ".join(self.errors(lambda r: r["rules"].update(height=3))))

    def test_a_fact_must_be_in_the_schemes_text_and_point_at_a_source(self):
        self.assertIn("does not appear", " ".join(self.errors(lambda r: r["facts"][0].update(value="55"))))
        self.assertIn("listed sources", " ".join(self.errors(lambda r: r["facts"][0].update(source=2))))
        # 4,000 in the text matches a fact written as 4000
        self.assertEqual(self.errors(lambda r: r["facts"][1].update(value="4000")), [])

    def test_a_record_cannot_be_called_verified_while_a_fact_is_not(self):
        errors = self.errors(lambda r: r["facts"][0].update(status="unverified"))
        self.assertIn("cannot contain unverified or disputed facts", " ".join(errors))
        record = with_file(RECORD)
        record["facts"][0]["status"] = "disputed"
        record["review"] = {"status": "needs_review", "notes": "age limit differs between two pages"}
        errors, warnings = validate_record(record)
        self.assertEqual(errors, [])
        self.assertIn("marked needs_review (age limit differs between two pages)", warnings[0])

    def test_missing_basics_are_reported_in_plain_words(self):
        self.assertIn("at least one source", " ".join(self.errors(lambda r: r.update(sources=[]))))
        self.assertIn("needs an https url", " ".join(self.errors(lambda r: r["sources"][0].update(url="ftp://x"))))
        self.assertIn("accessed date", " ".join(self.errors(lambda r: r["sources"][0].update(accessed="today"))))
        self.assertIn("eligibility needs", " ".join(self.errors(lambda r: r["sections"].update(eligibility=[]))))
        self.assertIn("needs its state", " ".join(self.errors(lambda r: r.update(state=""))))
        self.assertIn("category must be", " ".join(self.errors(lambda r: r.update(category="misc"))))
        self.assertIn("must be named", " ".join(self.errors(lambda r: r.update(_file="other.yaml"))))

    def test_two_schemes_cannot_share_an_id_and_shared_names_are_flagged(self):
        twin = with_file(RECORD)
        errors, _ = validate_records([with_file(RECORD), twin])
        self.assertIn("is also used by", " ".join(errors))
        other = with_file({**copy.deepcopy(RECORD), "id": "other_grant", "name": "Other Grant"})
        errors, warnings = validate_records([with_file(RECORD), other])
        self.assertEqual(errors, [])
        self.assertTrue(any("'LRG' is also used by loom_grant" in w for w in warnings))

    def test_chunks_name_the_scheme_keep_exceptions_with_the_rules_and_label_the_rules(self):
        chunks = record_chunks(RECORD)
        self.assertEqual([c["chunk_id"] for c in chunks],
                         ["loom_grant__overview", "loom_grant__eligibility", "loom_grant__benefits",
                          "loom_grant__how_to_apply", "loom_grant__documents"])
        self.assertEqual([c["section"] for c in chunks], ["overview", "rules", "benefits", "application", "documents"])
        for chunk in chunks:
            self.assertTrue(chunk["title"].startswith("Loom Renewal Grant (LRG) — "))
            self.assertEqual(chunk["aliases"], ["LRG", "करघा नवीकरण अनुदान", "kargha anudan"])
            self.assertEqual((chunk["scheme_id"], chunk["source"]), ("loom_grant", "https://example.org/loom"))
        rules = chunks[1]["text"]
        self.assertEqual(rules, "- The weaver must be between 21 and 50 years old.\n"
                                "- Annual family income must be Rs 90,000 or less.\n"
                                "Not eligible:\n- Weavers who received the grant in the last five years.")
        self.assertIn("State scheme of Odisha", chunks[0]["text"])

    def test_long_sections_are_split_between_statements_never_inside_one(self):
        record = copy.deepcopy(RECORD)
        record["sections"]["eligibility"] = [f"Condition number {i}: " + "word " * 40 for i in range(12)]
        chunks = [c for c in record_chunks(record, max_chars=700) if c["section"] == "rules"]
        self.assertGreater(len(chunks), 3)
        self.assertEqual(chunks[1]["chunk_id"], "loom_grant__eligibility_2")
        self.assertIn(f"(part 2 of {len(chunks)})", chunks[1]["title"])
        lines = [line for chunk in chunks for line in chunk["text"].splitlines()]
        self.assertEqual([l for l in lines if l.startswith("- Condition")],
                         [f"- {s.strip()}" for s in record["sections"]["eligibility"]])
        self.assertTrue(all(len(c["text"]) <= 700 for c in chunks))

    def test_questions_come_only_from_verified_facts_and_point_at_the_chunk_with_the_answer(self):
        questions = record_questions(RECORD)
        self.assertEqual([q["query_id"] for q in questions],
                         ["loom_grant__age_max__en", "loom_grant__age_max__hi", "loom_grant__grant_amount__en"])
        self.assertEqual(questions[0]["gold_chunk_ids"], ["loom_grant__eligibility"])
        self.assertEqual(questions[2]["gold_chunk_ids"], ["loom_grant__benefits"])
        self.assertEqual((questions[2]["answer"], questions[2]["accept"]), ("4,000", ["4000"]))
        unverified = copy.deepcopy(RECORD)
        unverified["facts"][0]["status"] = "unverified"
        self.assertEqual(len(record_questions(unverified)), 1)

    def test_build_writes_a_corpus_the_search_can_load_and_leaves_out_unchecked_schemes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, output = Path(directory) / "schemes", Path(directory) / "documents"
            folder.mkdir()
            unchecked = {**copy.deepcopy(RECORD), "id": "boat_aid", "name": "Boat Repair Aid", "short_name": "BRA",
                         "names_hi": [], "aliases": [], "review": {"status": "needs_review"}}
            for record in (RECORD, unchecked):
                (folder / f"{record['id']}.yaml").write_text(yaml.safe_dump(record, allow_unicode=True), encoding="utf-8")
            self.assertEqual([r["id"] for r in load_records(folder)], ["boat_aid", "loom_grant"])
            report = build_corpus(folder, output)
            self.assertEqual((report["records"], report["used"], report["left_out_needs_review"]), (2, 1, 1))
            documents = load_corpus(output)
            self.assertEqual({d.scheme_id for d in documents}, {"loom_grant"})
            self.assertEqual([d.chunk_id for d in documents if d.is_rules], ["loom_grant__eligibility"])
            questions = [json.loads(l) for l in (output / "fact_questions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(questions), 3)
            self.assertEqual(build_corpus(folder, output, include_unverified=True)["used"], 2)
            (folder / "loom_grant.yaml").write_text(
                yaml.safe_dump({**RECORD, "rules": {"age_max": 99}}, allow_unicode=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rules.age_max is 99"):
                build_corpus(folder, output)


if __name__ == "__main__":
    unittest.main()
