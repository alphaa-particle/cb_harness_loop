"""From a scheme in the corpus to the prompt the model is given, with no model.

Small cases pin down each rule of the path. The last class runs the whole
path for every one of 5,000 generated schemes (20,000 sections).
"""

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.api import create_app
from engine.corpus import CorpusPack, load_corpus
from engine.domain_pack import DomainPack
from engine.enforcement import ANSWERED, BLOCKED, NO_EVIDENCE, UNCLEAR, build_messages, format_evidence
from engine.evaluator import Evaluator
from engine.llm_assistant import OllamaAssistant, TransformersAssistant
from engine.retrieval_evaluation import evaluate_context, load_questions
from engine.retriever import Retriever
from engine.schemas import RetrievedChunk
from engine.training_data import gold_chunks, make_example
from scripts.build_scheme_corpus import build
from support import asgi_call, write_jsonl

# Two schemes whose "how to apply" sections repeat the scheme name, as real
# documents do, so a question that names the scheme matches them before the rules.
SECTIONS = [
    {"chunk_id": "loom_rules", "scheme_id": "loom", "section": "rules", "title": "Loom Grant — eligibility rules",
     "text": "Eligibility rules: age must be between 21 and 50. Annual income must be 90000 rupees or below."},
    {"chunk_id": "loom_apply", "scheme_id": "loom", "section": "application", "title": "Loom Grant — how to apply",
     "text": "How to apply for the Loom Grant. Loom Grant applicants bring the Loom Grant form to the Loom Grant office."},
    {"chunk_id": "loom_pay", "scheme_id": "loom", "section": "payment", "title": "Loom Grant — payment",
     "text": "Payment under the Loom Grant. The Loom Grant pays 4000 rupees to the Loom Grant holder."},
    {"chunk_id": "boat_rules", "scheme_id": "boat", "section": "rules", "title": "Boat Repair Aid — eligibility rules",
     "text": "Eligibility rules: the applicant must own a registered fishing boat damaged by a storm."},
    {"chunk_id": "boat_apply", "scheme_id": "boat", "section": "application", "title": "Boat Repair Aid — how to apply",
     "text": "How to apply for Boat Repair Aid. Boat Repair Aid applicants visit the Boat Repair Aid harbour desk."},
]


def stubbed(assistant, reply="Likely eligible: none. Next step: check the official office."):
    """Replace the model with a fixed reply and record the prompts it would have received."""
    return patch.object(assistant, "_generate", return_value=reply)


class EvidenceSelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.corpus = write_jsonl(self.path / "corpus.jsonl", SECTIONS)

    def retriever(self, top_k=3, method="char_tfidf"):
        return Retriever(CorpusPack(self.corpus, top_k=top_k, method=method))

    def test_rules_are_added_when_search_alone_misses_them(self):
        for method in ("char_tfidf", "word_tfidf", "hybrid"):
            retriever = self.retriever(top_k=2, method=method)
            question = "How do I apply for the Loom Grant?"
            plain = [c.chunk_id for c in retriever.retrieve(question)]
            self.assertNotIn("loom_rules", plain, method)
            evidence = retriever.retrieve_evidence(question)
            self.assertEqual([c.chunk_id for c in evidence], ["loom_rules", plain[0]], method)
            self.assertEqual(evidence[0].text, SECTIONS[0]["text"])
            self.assertEqual((evidence[0].title, evidence[0].scheme_id, evidence[0].section),
                             ("Loom Grant — eligibility rules", "loom", "rules"))

    def test_no_section_is_shown_without_its_schemes_rules_and_the_limit_holds(self):
        retriever = self.retriever(top_k=5)
        rules = {"loom": "loom_rules", "boat": "boat_rules"}
        for question in ("How to apply for the Loom Grant or Boat Repair Aid?", "payment", "storm damaged boat"):
            for limit in (1, 2, 3, 4, 5):
                ids = [c.chunk_id for c in retriever.retrieve_evidence(question, top_k=limit)]
                self.assertLessEqual(len(ids), limit)
                self.assertEqual(len(ids), len(set(ids)))
                for position, chunk in enumerate(retriever.retrieve_evidence(question, top_k=limit)):
                    self.assertIn(rules[chunk.scheme_id], ids[:position + 1], (question, limit, ids))
        self.assertEqual([c.chunk_id for c in retriever.retrieve_evidence("Loom Grant form", top_k=1)], ["loom_rules"])

    def test_nothing_is_added_when_search_finds_nothing(self):
        retriever = self.retriever()
        for question in ("", "   ", "qqqq zzzz"):
            self.assertEqual(retriever.retrieve_evidence(question), [])
        strict = Retriever(CorpusPack(self.corpus, min_score=0.99))
        self.assertEqual(strict.retrieve_evidence("How do I apply for the Loom Grant?"), [])

    def test_unlabelled_corpora_behave_exactly_as_before(self):
        demo = Retriever(DomainPack("welfare_demo"))
        stress = Retriever(CorpusPack(ROOT / "tests/fixtures/retrieval/corpus.jsonl", method="hybrid"))
        questions = ["I am 30 and earn 9000. Which welfare schemes am I likely eligible for?",
                     "What income limit applies to the Nila Campus Books Grant?", "kidney dialysis in Maharashtra"]
        for retriever in (demo, stress):
            for question in questions:
                self.assertEqual(retriever.retrieve_evidence(question), retriever.retrieve(question))
        # The demo documents still hash to the value recorded before section labels existed.
        self.assertEqual(demo.corpus_fingerprint,
                         "34d5908415690e8b1090ffcaeaa2279aa046de5a7d211e91b9250bc940ee3d73")

    def test_markdown_sections_can_name_their_scheme_and_kind(self):
        (self.path / "md").mkdir()
        (self.path / "md" / "schemes.md").write_text(
            "Scheme ID: loom\nSection: rules\nAge must be between 21 and 50.\n---\n"
            "Scheme ID: loom\nChunk ID: loom_apply\nSection: application\nLoom Grant form at the Loom Grant office.\n---\n"
            "Scheme ID: legacy_scheme\nA section with one ID line keeps that ID.", encoding="utf-8")
        docs = load_corpus(self.path / "md")
        self.assertEqual([(d.chunk_id, d.scheme_key, d.section) for d in docs],
                         [("loom", "loom", "rules"), ("loom_apply", "loom", "application"),
                          ("legacy_scheme", "legacy_scheme", "")])
        retriever = Retriever(CorpusPack(self.path / "md", top_k=2))
        self.assertEqual([c.chunk_id for c in retriever.retrieve_evidence("Loom Grant office form", top_k=2)],
                         ["loom", "loom_apply"])
        with self.assertRaisesRegex(ValueError, "section must be a string"):
            load_corpus(write_jsonl(self.path / "bad.jsonl", [{"chunk_id": "x", "text": "t", "section": 3}]))


class PromptAndGuardTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.pack = DomainPack("welfare_demo", corpus_path=write_jsonl(Path(temp.name) / "corpus.jsonl", SECTIONS))
        # On five sections even an unrelated question shares letters with something
        # (about 0.2); real questions about these schemes score 0.4 and above.
        self.pack.eval_config["retrieval"].update(top_k=2, min_score=0.3)
        self.retriever = Retriever(self.pack)
        self.assistants = [TransformersAssistant(self.pack, self.retriever, _model=object(), _tokenizer=object()),
                           OllamaAssistant(self.pack, self.retriever)]

    def test_prompt_holds_the_instructions_the_rules_word_for_word_and_the_question(self):
        question = "How do I apply for the Loom Grant?"
        for assistant in self.assistants:
            with stubbed(assistant) as model:
                answer = assistant.answer(question)
            system, user = model.call_args.args[0]
            self.assertEqual(system, {"role": "system", "content": self.pack.prompts["system"]})
            self.assertEqual(user["role"], "user")
            self.assertTrue(user["content"].startswith(
                "Retrieved context:\nLoom Grant — eligibility rules\n" + SECTIONS[0]["text"]))
            self.assertIn(f"User question:\n{question}", user["content"])
            self.assertEqual(answer.mode, ANSWERED)
            self.assertEqual([c.chunk_id for c in answer.retrieved_chunks], ["loom_rules", "loom_apply"])
            self.assertEqual([system, user], build_messages(self.pack, question, answer.retrieved_chunks))

    def test_a_title_already_in_the_text_is_not_repeated(self):
        chunks = [RetrievedChunk("a", "Loom Grant rules: age 21 to 50.", 1.0, title="Loom Grant"),
                  RetrievedChunk("b", "Storm damage only.", 1.0, title="Boat Repair Aid"),
                  RetrievedChunk("c", "Untitled section.", 1.0)]
        self.assertEqual(format_evidence(chunks),
                         "Loom Grant rules: age 21 to 50.\n\nBoat Repair Aid\nStorm damage only.\n\nUntitled section.")

    def test_the_model_is_never_asked_without_evidence(self):
        for assistant in self.assistants:
            with stubbed(assistant) as model:
                unclear = assistant.answer("xkcdqrst bzzzrk plmnvx")
                unmatched = assistant.answer("What is the capital of France?")
                empty = assistant.answer("   ")
            model.assert_not_called()
            self.assertEqual((unclear.mode, unclear.answer_text), (UNCLEAR, self.pack.prompts["unclear"]))
            self.assertEqual((empty.mode, empty.retrieved_chunks), (UNCLEAR, []))
            self.assertEqual((unmatched.mode, unmatched.answer_text, unmatched.retrieved_chunks),
                             (NO_EVIDENCE, self.pack.prompts["no_evidence"], []))

    def test_a_forbidden_claim_is_replaced_by_the_rules_themselves(self):
        evaluator = Evaluator(self.pack)
        claims = list(self.pack.forbidden_claims())
        question = "How do I apply for the Loom Grant?"
        for assistant in self.assistants:
            for reply in ("Great news, your approval is guaranteed.", "You will receive money. Approval is certain!"):
                with stubbed(assistant, reply):
                    answer = assistant.answer(question)
                self.assertEqual(answer.mode, BLOCKED)
                self.assertNotIn(reply, answer.answer_text)
                self.assertTrue(answer.answer_text.startswith(self.pack.prompts["blocked"]))
                self.assertIn(SECTIONS[0]["text"], answer.answer_text)
                self.assertEqual(evaluator.forbidden_claim_violations(answer.answer_text, claims), [])
            careful = "Approval is not guaranteed. Next step: confirm your age with the office."
            with stubbed(assistant, careful):
                answer = assistant.answer_with_context(question, self.retriever.get_chunks_by_ids(["loom_rules"]))
            self.assertEqual((answer.mode, answer.answer_text), (ANSWERED, careful))

    def test_training_examples_use_the_runtime_prompt_builder(self):
        record = {"case_id": "c1", "split": "train", "condition": "clean",
                  "question": "Can I get the Loom Grant at 30?",
                  "ground_truth": {"gold_chunk_ids": ["loom_rules"], "likely_eligible": [],
                                   "unknown_due_to_missing_info": ["loom"], "ineligible": ["boat"],
                                   "must_ask_about": ["income", "age"],
                                   "ineligibility_reasons": {"boat": "no boat was mentioned"}}}
        example = make_example(record, self.pack, self.retriever, {"loom": "Loom Grant", "boat": "Boat Repair Aid"})
        self.assertEqual(example["messages"][:2],
                         build_messages(self.pack, record["question"], gold_chunks(record, self.retriever)))
        self.assertEqual(example["messages"][2], {"role": "assistant", "content": (
            "Likely eligible: none based on the provided information. Needs confirmation: Loom Grant. "
            "Not likely eligible: Boat Repair Aid. Reason: Boat Repair Aid: no boat was mentioned. "
            "Missing details: please share your age, monthly income. " + self.pack.prompts["closing"])})
        # The grader accepts the wording the training target uses to ask for each field.
        labels = self.pack.profile_schema["field_labels"]
        asking = "Missing details: please share your " + ", ".join(labels.values()) + "."
        self.assertEqual(Evaluator(DomainPack("welfare_demo")).followup(asking, list(labels)), 1)

    def test_demo_training_targets_state_the_reason_from_the_domain_rules(self):
        pack = DomainPack("welfare_demo")
        saved = json.loads((ROOT / "data/outputs/audit_qwen_500_cases.jsonl").read_text().splitlines()[0])
        self.assertNotIn("ineligibility_reasons", saved["ground_truth"])  # recorded before reasons were kept
        fresh = pack.build_ground_truth(saved["profile"], saved["visible"])
        self.assertEqual({k: v for k, v in fresh.items() if k != "ineligibility_reasons"}, saved["ground_truth"])
        self.assertEqual(set(fresh["ineligibility_reasons"]), set(fresh["ineligible"]))

    def test_context_endpoint_shows_the_prompt_without_loading_a_model(self):
        async def exercise():
            app = create_app(corpus_path=self.pack.corpus_path, backend="transformers")
            async with app.router.lifespan_context(app):
                _, found = await asgi_call(app, "/context", {"question": "How do I apply for the Loom Grant?", "top_k": 2})
                _, missing = await asgi_call(app, "/context", {"question": "qqqq zzzz"})
                self.assertIsNone(app.state.assistant)
            self.assertEqual([e["chunk_id"] for e in found["evidence"]], ["loom_rules", "loom_apply"])
            self.assertIn(SECTIONS[0]["text"], found["messages"][1]["content"])
            self.assertEqual((missing["mode"], missing["evidence"], missing["messages"]), (NO_EVIDENCE, [], []))
        asyncio.run(exercise())


class FiveThousandSchemeTests(unittest.TestCase):
    """Scheme -> search -> rules-first evidence -> prompt -> (stub model) -> guard -> answer."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        path = Path(cls.temp.name)
        summary = build(path, 5000)
        assert (summary["schemes"], summary["sections"]) == (5000, 20000)
        cls.pack = DomainPack("welfare_demo", corpus_path=path / "corpus.jsonl")
        # Chosen on the dev questions only: scripts/calibrate_retrieval.py picks 0.30.
        cls.pack.eval_config["retrieval"].update(method="hybrid", top_k=3, min_score=0.30)
        cls.retriever = Retriever(cls.pack)
        cls.questions = path / "questions.jsonl"
        cls.cases = load_questions(cls.questions)
        cls.text = dict(zip(cls.retriever.chunk_ids, cls.retriever.chunks))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def of(self, condition, split=None):
        return [c for c in self.cases if c["condition"] == condition and split in (None, c["split"])]

    def test_every_scheme_asked_for_by_name_gets_its_own_rules_in_the_prompt(self):
        assistant = TransformersAssistant(self.pack, self.retriever, _model=object(), _tokenizer=object())
        named = self.of("named")
        self.assertEqual(len({c["group_id"] for c in named}), 5000)
        with stubbed(assistant) as model:
            for case in named:
                answer = assistant.answer(case["question"])
                rules_id = case["required_chunk_ids"][0]
                prompt = model.call_args.args[0][1]["content"]
                self.assertEqual(answer.mode, ANSWERED, case["question"])
                self.assertEqual(answer.retrieved_chunks[0].chunk_id, rules_id, case["question"])
                self.assertIn(self.text[rules_id], prompt, case["question"])
                self.assertLessEqual(len(answer.retrieved_chunks), 3)
        self.assertEqual(model.call_count, 5000)

    def test_held_out_harder_questions_reach_the_right_rules(self):
        test = [c for c in self.cases if c["split"] == "test"]
        enforced = evaluate_context(self.pack, self.retriever, test)["by_condition"]
        plain = evaluate_context(self.pack, self.retriever, test, enforce=False)["by_condition"]
        self.assertEqual(enforced["named"]["right_text_in_prompt_rate"], 1)
        self.assertEqual(enforced["documents"]["right_text_in_prompt_rate"], 1)
        # Measured 500/500 and 463/500; the floors are regression guards, not the claim.
        # Every described miss is one wording ("getting hurt at work" for accident
        # insurance) that shares no words with the scheme it means.
        self.assertGreaterEqual(enforced["typo"]["right_text_in_prompt_rate"], 0.98)
        self.assertGreaterEqual(enforced["described"]["right_text_in_prompt_rate"], 0.90)
        # Why rules-first exists: search alone ranks a scheme's other sections above its rules.
        for condition in ("named", "documents"):
            self.assertLess(plain[condition]["right_text_in_prompt_rate"], 0.05, condition)
        for condition in ("typo", "described"):
            self.assertLess(plain[condition]["right_text_in_prompt_rate"],
                            enforced[condition]["right_text_in_prompt_rate"], condition)

    def test_no_scheme_section_is_ever_sent_without_that_schemes_rules(self):
        sample = [c for kind in ("documents", "typo", "described", "absent", "off_topic")
                  for c in self.of(kind)[:150]]
        for case in sample:
            evidence = self.retriever.retrieve_evidence(case["question"])
            self.assertLessEqual(len(evidence), 3)
            ids = [c.chunk_id for c in evidence]
            for position, chunk in enumerate(evidence):
                self.assertIn(f"{chunk.scheme_id}_rules", ids[:position + 1], case["question"])

    def test_off_topic_questions_never_reach_the_model(self):
        assistant = OllamaAssistant(self.pack, self.retriever)
        off_topic = self.of("off_topic", "test")
        self.assertEqual(len(off_topic), 20)
        with stubbed(assistant) as model:
            modes = {assistant.answer(case["question"]).mode for case in off_topic}
        model.assert_not_called()
        self.assertEqual(modes, {NO_EVIDENCE})

    def test_a_scheme_missing_from_the_corpus_is_not_detected_but_its_stand_in_shows_its_state(self):
        # Known limit: word matching cannot tell that a named scheme is absent, so
        # the nearest scheme is sent instead. Its rules, which name the state it
        # is for, are always what the model sees first.
        absent = self.of("absent", "test")
        self.assertEqual(len(absent), 270)
        for case in absent:
            evidence = self.retriever.retrieve_evidence(case["question"])
            self.assertEqual(evidence[0].section, "rules")
            self.assertIn("The applicant must live in ", evidence[0].text)


if __name__ == "__main__":
    unittest.main()
