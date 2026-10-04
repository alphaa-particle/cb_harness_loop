"""Corpus replacement, metric correctness and HTTP/training integration tests."""

import asyncio
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.api import create_app
from engine.corpus import CorpusPack, load_corpus
from engine.domain_pack import DomainPack
from engine.evaluator import Evaluator
from engine.retrieval_evaluation import evaluate_retrieval, load_questions, ranked_metrics
from engine.retriever import Retriever
from engine.schemas import RetrievedChunk
from engine.training_data import gold_chunks, make_example
from scripts.calibrate_retrieval import calibrate
from support import asgi_call as request


class CorpusPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def write_jsonl(self, name, records):
        path = self.path / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        return path

    def sample_corpus(self):
        return self.write_jsonl("corpus.jsonl", [
            {"chunk_id": "new_rules", "scheme_id": "new_scheme", "title": "New Education Support", "aliases": ["NEWEDU"],
             "text": "New Education Support. College textbook reimbursement. Family income must be below 17000."},
            {"chunk_id": "medical_rules", "text": "Kidney dialysis treatment. A medical referral is required."},
        ])

    def sample_record(self, split="train"):
        return {"case_id": f"example_{split}", "split": split, "condition": "clean",
                "question": "What is NEWEDU?", "visible": {},
                "ground_truth": {"gold_chunk_ids": ["new_rules"], "likely_eligible": ["new_scheme"],
                                 "unknown_due_to_missing_info": [], "ineligible": [], "must_ask_about": []}}

    def test_jsonl_alias_search_preserves_original_gold_text(self):
        source = self.sample_corpus()
        for method in ("char_tfidf", "word_tfidf", "hybrid"):
            retriever = Retriever(CorpusPack(source, method=method))
            match = retriever.retrieve("NEWEDU", top_k=1)[0]
            self.assertEqual(match.chunk_id, "new_rules")
            original = load_corpus(source)[0].text
            self.assertEqual(match.text, original)
            record = self.sample_record()
            record["question"] = "kidney dialysis"  # Gold evidence must not come from a search prediction.
            self.assertEqual([c.text for c in gold_chunks(record, retriever)], [original])

    def test_swapping_corpus_changes_evidence_without_editing_domain_code(self):
        source = self.sample_corpus()
        with patch.dict(os.environ, {"SYNALIGN_CORPUS_PATH": str(source)}):
            first = Retriever(DomainPack("welfare_demo"))
            self.assertEqual(first.retrieve("NEWEDU", top_k=1)[0].chunk_id, "new_rules")
            self.write_jsonl("corpus.jsonl", [{"chunk_id": "updated", "text": "Updated grant pays for electric bicycles."}])
            second = Retriever(DomainPack("welfare_demo"))
            self.assertNotEqual(first.corpus_fingerprint, second.corpus_fingerprint)
            self.assertEqual(second.chunk_ids, ["updated"])
            self.assertEqual(second.get_chunks_by_ids(["new_rules"]), [])

    def test_top_k_configuration_reaches_the_assistant_and_search_api(self):
        source = self.sample_corpus()
        with patch.dict(os.environ, {"SYNALIGN_RETRIEVAL_TOP_K": "5"}):
            retriever = Retriever(DomainPack("welfare_demo", corpus_path=source))
            self.assertEqual(retriever.top_k, 5)
        from engine.api import SearchRequest
        self.assertIsNone(SearchRequest(question="books").top_k)

    def test_model_assistants_route_replacement_evidence_without_loading_a_model(self):
        from engine.llm_assistant import TransformersAssistant, OllamaAssistant, _looks_like_gibberish
        source = self.write_jsonl("multilingual.jsonl", [
            {"chunk_id": "new_rules", "title": "Textbook Grant", "aliases": ["NEWEDU", "किताब सहायता"],
             "text": "The fictional textbook grant requires college enrollment and income below 17000."}])
        pack = DomainPack("welfare_demo", corpus_path=source)
        retriever = Retriever(pack)
        transformer = TransformersAssistant(pack, retriever, _model=object(), _tokenizer=object())
        ollama = OllamaAssistant(pack, retriever)
        for assistant in (transformer, ollama):
            with patch.object(assistant, "_generate", return_value="stub generated answer") as generate:
                for question in ("NEWEDU", "किताब सहायता"):
                    answer = assistant.answer(question)
                    self.assertEqual([c.chunk_id for c in answer.retrieved_chunks], ["new_rules"])
                    messages = generate.call_args.args[0]
                    self.assertIn(retriever.chunks[0], messages[1]["content"])
                    self.assertIn(question, messages[1]["content"])
                    self.assertNotIn("PMSYM", str(messages))
        self.assertFalse(_looks_like_gibberish("मुझे किताब सहायता चाहिए"))
        self.assertTrue(_looks_like_gibberish("!!!"))
        self.assertTrue(_looks_like_gibberish("xkcdqrst bzzzrk plmnvx"))
        # Typos and vowel-poor words in an otherwise readable question are not gibberish.
        self.assertFalse(_looks_like_gibberish("My mohthly income is 12200. Which welfare schems am I likely eligibe for?"))

    def test_beir_folder_does_not_index_queries_as_evidence(self):
        self.write_jsonl("corpus.jsonl", [{"_id": "paper", "title": "Research", "text": "Original source text"}])
        self.write_jsonl("queries.jsonl", [{"_id": "query", "text": "Do not index me"}])
        docs = load_corpus(self.path)
        self.assertEqual([d.chunk_id for d in docs], ["paper"])

    def test_malformed_corpora_fail_instead_of_silently_skipping_rows(self):
        rows = [
            [{"chunk_id": "x", "_id": "y", "text": "source"}],
            [{"chunk_id": "x", "text": " "}],
            [{"text": "source"}],
            [{"chunk_id": "x", "text": "source", "aliases": "not-a-list"}],
            [{"chunk_id": "x", "text": "one"}, {"chunk_id": "x", "text": "two"}],
        ]
        for records in rows:
            with self.subTest(records=records), self.assertRaises(ValueError):
                load_corpus(self.write_jsonl("bad.jsonl", records))

    def test_nested_markdown_fallback_ids_are_distinct_and_stable(self):
        for name in ("first", "second"):
            (self.path / name).mkdir()
            (self.path / name / "rules.md").write_text(f"Rules for {name}")
        self.assertEqual([d.chunk_id for d in load_corpus(self.path)], ["first::rules_0", "second::rules_0"])

    def test_ranked_metrics_against_hand_calculated_values(self):
        result = ranked_metrics(["irrelevant", "b", "a"], {"a": 3, "b": 1}, 3)
        self.assertEqual(result["hit_rate"], 1)
        self.assertEqual(result["recall"], 1)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertEqual(result["mrr"], 0.5)
        self.assertAlmostEqual(result["ndcg"], (1 / math.log2(3) + 7 / 2) / (7 + 1 / math.log2(3)))
        self.assertEqual(ranked_metrics(["irrelevant"], {"a": 1}, 1)["recall"], 0)
        with self.assertRaises(ValueError):
            ranked_metrics(["a", "a"], {"a": 1}, 3)

    def test_no_answer_cases_are_not_counted_as_successful_recall(self):
        labels = self.write_jsonl("queries.jsonl", [
            {"query_id": "positive", "question": "books", "gold_chunk_ids": ["a"]},
            {"query_id": "negative", "question": "weather", "gold_chunk_ids": [], "expect_abstain": True},
        ])
        retriever = SimpleNamespace(chunk_ids=["a"], method="test", min_score=0, corpus_fingerprint="fixture",
                                    retrieve=lambda query, top_k: [RetrievedChunk("a", "books", 0.7)])
        report = evaluate_retrieval(retriever, load_questions(labels), [1, 3])
        self.assertEqual(report["summary"]["answerable"], 1)
        self.assertEqual(report["summary"]["unanswerable"], 1)
        self.assertEqual(report["summary"]["abstention_accuracy"], 0)
        self.assertEqual(report["summary"]["recall@3"], 1)
        self.assertEqual(report["summary"]["false_positive_rate"], 1)

    def test_incomplete_labels_unknown_ids_and_split_leakage_fail(self):
        incomplete = self.write_jsonl("incomplete.jsonl", [{"question": "books"}])
        with self.assertRaises(ValueError):
            load_questions(incomplete)
        leaked = self.write_jsonl("leaked.jsonl", [
            {"query_id": "a", "group_id": "same", "split": "dev", "question": "books", "gold_chunk_ids": ["x"]},
            {"query_id": "b", "group_id": "same", "split": "test", "question": "textbooks", "gold_chunk_ids": ["x"]},
        ])
        with self.assertRaisesRegex(ValueError, "leaks across splits"):
            load_questions(leaked, "test")
        unknown = self.write_jsonl("unknown.jsonl", [{"question": "books", "gold_chunk_ids": ["missing"]}])
        retriever = Retriever(CorpusPack(self.sample_corpus()))
        with self.assertRaisesRegex(ValueError, "missing corpus IDs"):
            evaluate_retrieval(retriever, load_questions(unknown))

    def test_calibration_cannot_use_test_cases_or_drop_relevant_evidence_unnoticed(self):
        cases = [{"split": "dev", "question": "positive", "expect_abstain": False, "relevance": {"good": 1}},
                 {"split": "dev", "question": "negative", "expect_abstain": True, "relevance": {}}]
        def search(question, top_k=None):
            return ([RetrievedChunk("bad", "", 0.9), RetrievedChunk("good", "", 0.2)]
                    if question == "positive" else [RetrievedChunk("bad", "", 0.5)])
        retriever = SimpleNamespace(chunk_ids=["bad", "good"], method="fixture", corpus_fingerprint="fixture",
                                    top_k=3, min_score=0, retrieve=search, retrieve_evidence=search)
        result = calibrate(retriever, cases)
        self.assertLessEqual(result["selected"]["min_score"], 0.2)
        self.assertEqual(result["selected"]["positive_hit_and_accept_rate"], 1)
        cases[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "dev cases"):
            calibrate(retriever, cases)

    def test_gold_sft_uses_replacement_corpus_and_preserves_file_schema(self):
        pack = DomainPack("welfare_demo", corpus_path=self.sample_corpus())
        retriever = Retriever(pack)
        example = make_example(self.sample_record(), pack, retriever, {"new_scheme": "New Education Support"})
        self.assertEqual(set(example), {"messages", "case_id", "split", "condition"})
        self.assertIn("17000", example["messages"][1]["content"])
        self.assertNotIn("PMSYM", json.dumps(example))
        self.assertIn("New Education Support", example["messages"][2]["content"])
        bad = self.sample_record()
        bad["ground_truth"]["gold_chunk_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "Gold evidence missing"):
            make_example(bad, pack, retriever)
        incomplete = self.sample_record()
        del incomplete["ground_truth"]["ineligible"]
        with self.assertRaisesRegex(ValueError, "Reviewed ground_truth"):
            make_example(incomplete, pack, retriever)

    def test_demo_audit_rejects_unrelated_corpus_before_writing(self):
        from engine import audit
        with patch.dict(os.environ, {"SYNALIGN_CORPUS_PATH": str(self.sample_corpus())}), \
                patch.object(audit, "OUTPUT_DIR", self.path / "audit_outputs"):
            with self.assertRaisesRegex(ValueError, "absent from the selected corpus"):
                audit.run_audit(n_users=2, backend="naive", label="invalid")
            self.assertFalse((self.path / "audit_outputs").exists())

    def test_sft_cli_keeps_splits_and_does_not_overwrite_on_bad_evidence(self):
        source = self.sample_corpus()
        records = [self.sample_record(split) for split in ("train", "dev", "test")]
        audit = self.write_jsonl("audit.jsonl", records)
        output = self.path / "training"
        command = [sys.executable, "-B", str(ROOT / "scripts/make_training_data.py"), "--input", str(audit),
                   "--corpus", str(source), "--output-dir", str(output)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        files = {"train": "sft_train.jsonl", "dev": "sft_dev.jsonl", "test": "sft_test_reference.jsonl"}
        original = {name: (output / name).read_bytes() for name in files.values()}
        for split, name in files.items():
            self.assertEqual(json.loads(original[name])["split"], split)
        records[-1]["ground_truth"]["gold_chunk_ids"] = ["missing"]
        self.write_jsonl("audit.jsonl", records)
        failed = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(original, {name: (output / name).read_bytes() for name in files.values()})

    def test_followup_scorer_credits_requests_and_field_identifiers(self):
        evaluator = Evaluator(DomainPack("welfare_demo"))
        self.assertEqual(evaluator.followup("Please share your age, income, worker_type and epfo.",
                                            ["age", "income", "worker_type", "epfo"]), 1)
        self.assertEqual(evaluator.followup("Age and income affect eligibility.", ["age", "income"]), 0)
        self.assertEqual(evaluator.followup("What is your income?", ["income"]), 1)
        self.assertEqual(evaluator.followup("Please do not share your income.", ["income"]), 0)
        self.assertEqual(evaluator.followup("Missing details: please share your worker_type and epfo.",
                                            ["worker_type", "epfo"]), 1)

    def test_api_search_works_without_model_and_after_corpus_replacement(self):
        source = self.sample_corpus()

        async def exercise():
            app = create_app(corpus_path=source, backend="transformers")
            async with app.router.lifespan_context(app):
                status, first = await request(app, "/search", {"question": "NEWEDU", "top_k": 1})
                self.assertEqual(status, 200)
                self.assertEqual(first["evidence"][0]["chunk_id"], "new_rules")
                self.assertIsNone(app.state.assistant)
                status, _ = await request(app, "/search", {"question": "anything", "top_k": 0})
                self.assertEqual(status, 422)
            self.write_jsonl("corpus.jsonl", [{"chunk_id": "replacement", "text": "Bicycle repair grant rules."}])
            replacement = create_app(corpus_path=source, backend="naive")
            async with replacement.router.lifespan_context(replacement):
                status, second = await request(replacement, "/search", {"question": "bicycle", "top_k": 1})
                self.assertEqual(status, 200)
                self.assertEqual(second["evidence"][0]["chunk_id"], "replacement")
                self.assertNotEqual(first["corpus_sha256"], second["corpus_sha256"])
                status, answer = await request(replacement, "/ask", {"question": "bicycle repair"})
                self.assertEqual(status, 200)
                self.assertEqual(answer["evidence"][0]["chunk_id"], "replacement")
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
