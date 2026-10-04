"""The answer grader, the question builder, the draft importer, the scale converter, keep-or-undo
and the fine-tuning examples."""

import copy
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.answer_evaluation import is_refusal, numbers_in, states_value, verdict
from engine.scheme_records import load_records, validate_record
from scripts.build_eval_questions import eligibility_questions, split_of
from scripts.build_scale_corpus import background_chunks, clean, same_scheme
from scripts.import_scheme_drafts import apply
from scripts.make_scheme_training_data import NOT_HERE, age_examples, build, income_examples
from test_scheme_records import RECORD


class GraderTests(unittest.TestCase):
    def test_numbers_are_read_whole_with_indian_grouping_scaling_words_and_devanagari_digits(self):
        self.assertEqual(numbers_in("Rs 6,000 a year in 3 instalments of Rs 2,000"), {6000, 3, 2000})
        self.assertEqual(numbers_in("कवर ₹२ लाख, आयु १८ से ५०"), {200000, 18, 50})
        self.assertEqual(numbers_in("1.5 crore and Rs 1,20,000"), {15000000, 120000})

    def test_each_accepted_form_is_matched_whole(self):
        self.assertTrue(states_value("You can join between 18 and 40.", "18 to 40"))
        self.assertFalse(states_value("You can join from 18 years.", "18 to 40"))        # half a range is wrong
        self.assertFalse(states_value("Call 1800-11-222.", "1800-180-5129"))
        self.assertFalse(states_value("Report the loss within 3 weeks.", "72 hours", ["3 days"]))
        self.assertTrue(states_value("Report the loss within 72 hours.", "72 hours", ["3 days"]))
        self.assertTrue(states_value("The premium is at most 1.5% of the sum.", "1.5%", ["1.5 percent"]))
        self.assertFalse(states_value("Given every 15 days.", "twice a month", ["1st and 15th"]))

    def test_values_match_as_numbers_and_words_as_text(self):
        self.assertTrue(states_value("The maximum entry age is 40 years.", "40"))
        self.assertFalse(states_value("The fee is 400 rupees.", "40"))
        self.assertTrue(states_value("Life cover of ₹2,00,000", "Rs 2 lakh", ["200000"]))
        self.assertTrue(states_value("Registration is free of cost.", "free", ["no fee", "0"]))
        self.assertFalse(states_value("Registration costs Rs 50.", "free", ["no fee", "0"]))

    def test_refusals_in_english_and_hindi(self):
        self.assertTrue(is_refusal("I could not find this in the scheme information."))
        self.assertTrue(is_refusal("योजना की जानकारी में यह नहीं मिला।"))
        self.assertFalse(is_refusal("You can apply at any bank."))

    def test_yes_and_no_are_read_from_the_opening_or_the_first_sentence(self):
        cases = {"Yes, you can join.": "yes", "No. The limit is 40.": "no", "Not sure: it depends.": "unsure",
                 "हाँ, आप पात्र हैं।": "yes", "नहीं, आपकी उम्र 40 से अधिक है।": "no",
                 "नहीं। आपकी उम्र 40 से अधिक है।": "no", "हाँ। आप पात्र हैं।": "yes",
                 "यह उम्र 13 साल है, इसलिए आवेदन नहीं कर सकते।": "no",
                 "आपकी उम्र 26 साल है, इसलिए आप आवेदन कर सकते हैं।": "yes",
                 "You are 45, so you cannot apply.": "no", "Nobody over 40 can join.": "none",
                 "The scheme covers ages 18 to 40.": "none"}
        for text, expected in cases.items():
            self.assertEqual(verdict(text), expected, text)

    def test_repeating_the_question_or_opening_with_a_refusal_is_not_an_answer(self):
        from engine.corpus import CorpusDocument
        from scripts.evaluate_pipeline import SectionWords, grade
        docs = [CorpusDocument("a__rules", "Weavers aged 21 to 50 with a handloom.", title="Loom Grant (LG) — Rules",
                               scheme_id="a"),
                CorpusDocument("a__how_to_apply", "Submit the form at the block office.", title="Loom Grant (LG) — Apply",
                               scheme_id="a"),
                CorpusDocument("b__rules", "Small farmers with land records.", title="Seed Kit — Rules", scheme_id="b")]
        words = SectionWords(docs)
        patterns = {"a": [re.compile(r"\bLG\b")], "b": [re.compile(r"\bSeed Kit\b")]}
        compare = {"query_id": "c", "group_id": "c", "type": "compare", "language": "en", "expect_abstain": False,
                   "question": "What is the difference between LG and Seed Kit?", "gold_answer": {"schemes": ["a", "b"]}}
        self.assertEqual(grade(compare, compare["question"], "answered", patterns, words)["correct"], 0.0)
        said = "LG is for weavers aged 21 to 50; Seed Kit is for small farmers with land records."
        self.assertEqual(grade(compare, said, "answered", patterns, words)["correct"], 1.0)
        apply = {"query_id": "p", "group_id": "p", "type": "apply", "language": "en", "expect_abstain": False,
                 "question": "Where do I submit the form for LG?", "gold_groups": [["a__how_to_apply"]]}
        self.assertEqual(grade(apply, apply["question"], "answered", patterns, words)["correct"], 0.0)
        self.assertEqual(grade(apply, "Go to the block office.", "answered", patterns, words)["correct"], 1.0)
        fact = {"query_id": "f", "group_id": "f", "type": "fact", "language": "en", "expect_abstain": False,
                "question": "Under Poshan 2.0, how often is the ration given: every month or only once?",
                "gold_answer": {"value": "twice a month", "accept": ["2 times a month", "once"]}}
        digit_names = re.compile(r"Poshan 2\.0", re.IGNORECASE)
        self.assertEqual(grade(fact, "Under Poshan 2.0 it is given every week.", "answered", patterns, words,
                               digit_names)["correct"], 0.0)                     # "2.0" is a name, not 2
        self.assertEqual(grade(fact, "It is given only once.", "answered", patterns, words)["correct"], 0.0)
        self.assertEqual(grade(fact, "The information does not contain this. It is 2 times a month.", "answered",
                               patterns, words)["correct"], 0.0)                 # opens by refusing
        self.assertEqual(grade(fact, "It is given 2 times a month.", "answered", patterns, words)["correct"], 1.0)

    def test_use_case_answers_must_use_words_of_the_right_section_not_just_the_schemes_name(self):
        from engine.corpus import CorpusDocument
        from scripts.evaluate_pipeline import SectionWords
        docs = [CorpusDocument("loom__how_to_apply", "Apply at the block handloom office with the form.", scheme_id="loom"),
                CorpusDocument("loom__documents", "Weaver identity card and handloom bank passbook.", scheme_id="loom"),
                CorpusDocument("loom__benefits", "A handloom grant paid into the bank account.", scheme_id="loom")]
        words = SectionWords(docs)
        self.assertEqual(words.own(["loom__documents"]), {"weaver", "identity", "card", "passbook"})
        self.assertEqual(words.used("Handloom weavers get this from the bank.", ["loom__documents"]), 0)
        self.assertEqual(words.used("Bring your weaver identity card.", ["loom__documents"]), 3)


class QuestionBuilderTests(unittest.TestCase):
    def test_eligibility_questions_follow_the_schemes_own_limits(self):
        questions = eligibility_questions(RECORD, ["loom_grant__eligibility"])
        by_variant = {(q["variant"], q["language"]): q for q in questions}
        self.assertEqual(by_variant[("age_over", "en")]["facts_given"], {"age": 55})
        self.assertEqual(by_variant[("age_over", "en")]["gold_verdict"], "no")
        self.assertEqual(by_variant[("age_under", "en")]["facts_given"], {"age": 18})
        self.assertEqual(by_variant[("age_inside", "hi")]["gold_verdict"], "not_no")
        self.assertEqual(by_variant[("income_over", "en")]["facts_given"], {"annual_income_max": 135000})
        self.assertIn("करघा नवीकरण अनुदान", by_variant[("age_inside", "hi")]["question"])
        self.assertTrue(all(q["gold_chunk_ids"] == ["loom_grant__eligibility"] for q in questions))

    def test_splits_are_fixed_by_scheme(self):
        self.assertEqual(split_of("apy"), split_of("apy"))
        self.assertEqual({split_of(f"scheme_{i}") for i in range(50)}, {"dev", "test"})


class ImporterTests(unittest.TestCase):
    def draft(self):
        record = copy.deepcopy(RECORD)
        record.pop("review")
        for fact in record["facts"]:
            fact.pop("status")
        record["status_note"] = "Amounts checked on the state portal."
        return record

    def item(self, item, verdict="confirmed", opened=True, correct=""):
        return {"item": item, "claimed": "", "verdict": verdict, "correct_value": correct,
                "official_url": "https://example.gov.in/x", "page_opened": opened, "note": ""}

    def test_everything_confirmed_on_opened_pages_makes_a_verified_record(self):
        check = {"still_open": "yes", "other_problems": [],
                 "items": [self.item("age_max"), self.item("grant_amount"), self.item("rule:age_max"), self.item("names_hi")]}
        record = apply(self.draft(), check, {})
        self.assertEqual(record["review"]["status"], "verified")
        self.assertEqual(validate_record({**record, "_file": "loom_grant.yaml"})[0], [])

    def test_unconfirmed_facts_disputes_and_missing_checks_need_review(self):
        snippet_only = {"still_open": "yes", "other_problems": [],
                        "items": [self.item("age_max", opened=False), self.item("grant_amount")]}
        record = apply(self.draft(), snippet_only, {})
        self.assertEqual(record["review"]["status"], "needs_review")
        self.assertEqual(record["facts"][0]["status"], "unverified")
        self.assertEqual(apply(self.draft(), None, {})["review"]["status"], "needs_review")
        closed = {"still_open": "no", "other_problems": [], "items": [self.item("age_max"), self.item("grant_amount")]}
        self.assertIn("closed to new applicants", " ".join(apply(self.draft(), closed, {})["review"]["notes"]))

    def test_a_third_look_from_an_official_page_corrects_the_fact_and_its_statement(self):
        check = {"still_open": "yes", "other_problems": [],
                 "items": [self.item("age_max"), self.item("grant_amount", "wrong", True, "Rs 5,000")]}
        decision = {"grant_amount": {"decided": "checker_right", "final_value": "Rs 5,000", "page_opened": True,
                                     "official_url": "https://example.gov.in/rules",
                                     "corrected_statement": "A grant of Rs 5,000 paid into the weaver's bank account."}}
        record = apply(self.draft(), check, decision)
        self.assertEqual(record["facts"][1]["value"], "Rs 5,000")
        self.assertIn("A grant of Rs 5,000 paid into the weaver's bank account.", record["sections"]["benefits"])
        self.assertEqual(record["review"]["status"], "verified")
        self.assertIn("corrected from '4,000' to 'Rs 5,000'", " ".join(record["review"]["notes"]))

    def test_a_rule_the_checker_read_differently_is_removed_and_hindi_names_are_added(self):
        draft = self.draft()
        draft["names_hi"] = []
        check = {"still_open": "yes", "other_problems": [],
                 "items": [self.item("age_max"), self.item("grant_amount"),
                           self.item("rule:age_max", "wrong", True, "45 for general, 50 for women"),
                           self.item("names_hi", "wrong", True, "करघा नवीकरण अनुदान (official)")]}
        record = apply(draft, check, {})
        self.assertNotIn("age_max", record["rules"])
        self.assertEqual(record["names_hi"], ["करघा नवीकरण अनुदान"])
        self.assertEqual(record["review"]["status"], "needs_review")


class ScaleCorpusTests(unittest.TestCase):
    ROW = {"slug": "abc", "name": ' "Relief" for Fishers', "description": "Help for <b>fisher</b> families.",
           "state": "Goa", "department": "Fisheries Department", "ministry": "null",
           "eligibility_text": "- Must be a fisher.\n- Aged **18-60** years.<br>",
           "benefits": "₹ 1,00,000 in two parts.", "application_process": "Step 1: Visit the office.",
           "documents_required": "Aadhaar\nnull", "official_url": "https://www.myscheme.gov.in/schemes/abc"}

    def test_background_schemes_get_the_same_sections_cleaned_of_markup(self):
        self.assertEqual(clean("- Aged **18-60** years.<br>\n> null"), ["Aged 18-60 years."])
        chunks = background_chunks(self.ROW)
        self.assertEqual([c["section"] for c in chunks], ["overview", "rules", "benefits", "application", "documents"])
        self.assertEqual(chunks[1]["title"], "Relief for Fishers — Eligibility rules")
        self.assertEqual(chunks[1]["text"], "- Must be a fisher.\n- Aged 18-60 years.")
        self.assertIn("State scheme of Goa. Fisheries Department.", chunks[0]["text"])
        self.assertTrue(all(c["scheme_id"] == "ms__abc" for c in chunks))

    def test_copies_of_our_own_schemes_are_recognised(self):
        ours = {"apy": {"atal pension yojana", "apy"}}
        self.assertEqual(same_scheme({"name": "Atal Pension Yojana"}, ours), "apy")
        self.assertIsNone(same_scheme({"name": "Goa Fishers Relief"}, ours))


class KeepOrUndoTests(unittest.TestCase):
    def write_run(self, folder, correct, unsafe=0):
        folder.mkdir()
        (folder / "report.json").write_text(json.dumps({"questions_sha256": "same"}))
        rows = [{"query_id": f"q{i}", "group_id": f"g{i % 10}", "type": "fact", "correct": float(c),
                 "unsafe_yes": float(i < unsafe)} for i, c in enumerate(correct)]
        (folder / "answers_fusion.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def verdict(self, before, after, unsafe_before=0, unsafe_after=0):
        with tempfile.TemporaryDirectory() as d:
            self.write_run(Path(d) / "a", before, unsafe_before)
            self.write_run(Path(d) / "b", after, unsafe_after)
            out = subprocess.run([sys.executable, str(ROOT / "scripts/compare_runs.py"), str(Path(d) / "a"),
                                  str(Path(d) / "b")], capture_output=True, text=True, check=True).stdout
            return out.strip().splitlines()[-1]

    def test_clear_gains_are_kept_losses_and_new_unsafe_answers_undone(self):
        self.assertEqual(self.verdict([0] * 60 + [1] * 40, [1] * 100), "VERDICT: KEEP")
        self.assertEqual(self.verdict([1] * 100, [0] * 60 + [1] * 40), "VERDICT: UNDO")
        self.assertEqual(self.verdict([1] * 100, [1] * 100, 0, 3), "VERDICT: UNDO")
        self.assertEqual(self.verdict([1, 0] * 50, [0, 1] * 50), "VERDICT: NO CLEAR DIFFERENCE")

    def write_answers(self, folder, rows):
        folder.mkdir()
        (folder / "report.json").write_text(json.dumps({"questions_sha256": "same"}))
        (folder / "answers_fusion.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def same_model(self, before, after):
        with tempfile.TemporaryDirectory() as d:
            self.write_answers(Path(d) / "a", before)
            self.write_answers(Path(d) / "b", after)
            out = subprocess.run([sys.executable, str(ROOT / "scripts/compare_runs.py"), str(Path(d) / "a"),
                                  str(Path(d) / "b"), "--same-model"], capture_output=True, text=True, check=True).stdout
            return out.strip().splitlines()[-1]

    def test_search_changes_need_steady_answers_and_no_slice_or_refusal_loss(self):
        def row(i, correct, evidence="e1", answer="a", kind="fact", language="en"):
            return {"query_id": f"q{i}", "group_id": f"g{i % 20}", "type": kind, "language": language,
                    "correct": float(correct), "evidence": [evidence], "answer": answer}
        before = [row(i, i >= 40) for i in range(100)]
        better = [row(i, True, "e2", "b") if i < 40 else row(i, True) for i in range(100)]
        self.assertEqual(self.same_model(before, better), "VERDICT: KEEP")
        unsteady = [row(i, True, answer="other") for i in range(100)]            # same evidence, new answer
        self.assertTrue(self.same_model(before, unsteady).startswith("VERDICT: INVALID"))
        before_hi = [row(i, i >= 40 or i == 0, language="hi" if i == 0 else "en") for i in range(100)]
        lost_hi = [row(i, i != 0, "e2", "b", language="hi" if i == 0 else "en") for i in range(100)]
        self.assertEqual(self.same_model(before_hi, lost_hi), "VERDICT: UNDO")   # Hindi lost its one answer
        with tempfile.TemporaryDirectory() as d:          # runs on different splits are not compared
            self.write_answers(Path(d) / "a", before)
            (Path(d) / "a" / "report.json").write_text(json.dumps({"questions_sha256": "same", "split": "dev"}))
            self.write_answers(Path(d) / "b", better)
            (Path(d) / "b" / "report.json").write_text(json.dumps({"questions_sha256": "same", "split": "test"}))
            done = subprocess.run([sys.executable, str(ROOT / "scripts/compare_runs.py"), str(Path(d) / "a"),
                                   str(Path(d) / "b")], capture_output=True, text=True)
            self.assertNotEqual(done.returncode, 0)
        refused = [row(i, i >= 40) for i in range(99)] + [row(99, True, kind="absent")]
        answered = [row(i, True, "e2", "b") for i in range(99)] + [row(99, False, "e2", "b", kind="absent")]
        self.assertEqual(self.same_model(refused, answered), "VERDICT: UNDO")


class SearchRoundRuleTests(unittest.TestCase):
    """scripts/search_round.py: a search change is kept only if nothing gets worse."""

    QUESTIONS = ([{"query_id": f"q{i}", "group_id": f"s{i % 10}", "set": "questions", "type": "fact",
                   "language": "hi" if i < 5 else "en"} for i in range(60)]
                 + [{"query_id": f"n{i}", "group_id": f"n{i}", "set": "questions", "type": "absent", "language": "en"}
                    for i in range(3)])

    class Fake:
        def __init__(self, in_prompt, refused=(True, True, True), lost=0, ms=1.0):
            row = lambda ok: {"in_prompt": ok, "scheme": ok, "refused": False}  # noqa: E731
            rows = {f"q{i}": row(ok) for i, ok in enumerate(in_prompt)}
            rows.update({f"n{i}": {"refused": r} for i, r in enumerate(refused)})
            self.top_k, self.ms = 3, ms
            self.rows = {k: rows for k in (3, 2, 5)}
            self.wordings = [{"scheme": n >= lost, "refused": False} for n in range(10)]

    def judge(self, before, after, own=True):
        from scripts.search_round import judge_size
        return judge_size("size", self.QUESTIONS, before, after, own)[1]

    def test_a_change_of_prompt_size_is_judged_at_each_sides_own_size(self):
        from scripts.search_round import judge_size
        before = self.Fake([i >= 20 for i in range(60)])
        after = self.Fake([i >= 10 for i in range(60)])
        after.top_k = 4
        after.rows = {**after.rows, 4: after.rows[3]}      # the after side's own size is 4
        before.rows = {**before.rows, 3: before.rows[3]}
        report, problems = judge_size("size", self.QUESTIONS, before, after, True)
        self.assertEqual(report["in_prompt"], [40, 50])
        self.assertEqual(problems, [])

    def test_settings_are_checked(self):
        from scripts.evaluate_pipeline import merged, parse_settings
        self.assertEqual(parse_settings(["add_named_schemes=true"]), {"add_named_schemes": True})
        for bad in (["add_named_schemes"], ["add_named_scheme=true"], ["top_k=4"]):
            with self.assertRaises(SystemExit):
                parse_settings(bad)
        self.assertEqual(parse_settings(["top_k=4"], extra=("top_k",)), {"top_k": 4})
        self.assertEqual(merged({"fusion_weights": {"bm25": 0.25, "char": 0.25}}, {"fusion_weights": {"bm25": 0.5}}),
                         {"fusion_weights": {"bm25": 0.5, "char": 0.25}})

    def test_rule(self):
        base = [i >= 20 for i in range(60)]
        gain = [i >= 10 for i in range(60)]
        self.assertEqual(self.judge(self.Fake(base), self.Fake(gain)), [])
        hindi_lost = [i >= 10 and i != 30 for i in range(60)]
        hindi_lost[2] = False
        base_hi = list(base)
        base_hi[2] = True
        problems = self.judge(self.Fake(base_hi), self.Fake(hindi_lost))
        self.assertTrue(any(p.startswith("A3") and "language=hi" in p for p in problems))
        self.assertTrue(any(p.startswith("A4") for p in self.judge(self.Fake(base), self.Fake(gain, (True, True, False)))))
        self.assertEqual(self.judge(self.Fake(base), self.Fake(gain, (True, True, False)), own=False), [])
        self.assertTrue(any(p.startswith("A8") for p in self.judge(self.Fake(base), self.Fake(gain, lost=1))))
        self.assertTrue(any(p.startswith("A9") for p in self.judge(self.Fake(base), self.Fake(gain, ms=3.0))))
        self.assertTrue(any("A1" in p for p in self.judge(self.Fake(base), self.Fake(base))))


class TrainingExampleTests(unittest.TestCase):
    def test_answers_follow_the_schemes_limits_and_open_the_way_the_grader_reads_them(self):
        examples = age_examples(RECORD, random.Random(1)) + income_examples(RECORD, random.Random(1))
        self.assertTrue({"en", "hi"} <= {language for language, _, _ in examples})
        for language, question, answer in examples:
            given = int(re.search(r"\d+", question.replace(",", "")).group())
            about_income = "earn" in question or "आय" in question
            limit_ok = given <= 90000 if about_income else 21 <= given <= 50
            self.assertEqual(verdict(answer), "yes" if limit_ok else "no", answer)

    def test_only_dev_schemes_are_used_and_refusals_show_another_schemes_rules(self):
        examples = build("india_schemes", random.Random(7))
        names = {r["id"]: r["name"] for r in load_records(ROOT / "domains" / "india_schemes" / "schemes")}
        self.assertTrue(examples)
        self.assertEqual({split_of(e["scheme_id"]) for e in examples}, {"dev"})
        for e in examples:
            evidence = e["messages"][1]["content"]
            self.assertEqual([m["role"] for m in e["messages"]], ["system", "user", "assistant"])
            shown, section = evidence.split("\n")[1].split(" — ")
            name = names[e["scheme_id"]]
            asked = shown == name or shown.startswith(f"{name} (")
            self.assertTrue(section.startswith("Eligibility rules"), section)
            if e["kind"] == "not_here":
                self.assertEqual(e["messages"][-1]["content"], NOT_HERE[e["language"]])
                self.assertFalse(asked, shown)
            else:
                self.assertTrue(asked, shown)


if __name__ == "__main__":
    unittest.main()
