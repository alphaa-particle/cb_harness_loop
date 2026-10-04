"""Retrieval correctness and compatibility checks; no model weights required."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.assistant import NaiveBaselineAssistant
from engine.corpus import CorpusPack
from engine.domain_pack import DomainPack
from engine.evaluator import Evaluator
from engine.perturbations import make_test_cases
from engine.retrieval_evaluation import recall_at_k, mrr
from engine.retriever import Retriever
from engine.schemas import RetrievedChunk
from engine.splits import assign_splits
from engine.synthesis import make_synthetic_users


class LegacyRetriever(Retriever):
    """The previous query algorithm, retained only as a regression oracle."""

    def retrieve(self, query, top_k=None):
        qv = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self._search_matrix.T).flatten()
        order = scores.argsort()[::-1][:top_k or self.top_k]
        return [RetrievedChunk(self.chunk_ids[i], self.chunks[i], float(scores[i]))
                for i in order]


class RetrieverTests(unittest.TestCase):
    def pack_for(self, sections, top_k=3):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        (path / "schemes.md").write_text("\n---\n".join(sections), encoding="utf-8")
        return CorpusPack(path, top_k=top_k)

    def test_relevant_rules_and_original_ids_are_returned(self):
        sections = [
            "Scheme ID: education\nScholarship for students. Income must be below 12000.",
            "Scheme ID: housing\nHousing roof repair. Applicant must own the damaged home.",
            "Scheme ID: pension\nPension for workers. Age must be between 18 and 40.",
        ]
        retriever = Retriever(self.pack_for(sections))
        result = retriever.retrieve("scholarship income students", top_k=1)
        self.assertEqual(result[0].chunk_id, "education")
        self.assertEqual(result[0].text, sections[0])
        gold = retriever.get_chunks_by_ids(["pension", "absent", "education", "pension"])
        self.assertEqual([c.chunk_id for c in gold], ["pension", "education", "pension"])
        self.assertEqual([c.text for c in gold], [sections[2], sections[0], sections[2]])
        self.assertTrue(all(c.score == 1.0 for c in gold))

    def test_empty_and_unmatched_queries_do_not_return_arbitrary_schemes(self):
        retriever = Retriever(self.pack_for(["cobalt cobalt", "zzzz zzzz"]))
        for question in ("", "  ", "🛰️", "qqqqqq"):
            with self.subTest(question=question):
                self.assertEqual(retriever.retrieve(question), [])
        self.assertEqual(len(retriever.retrieve("cobalt", top_k=100)), 1)
        self.assertEqual(retriever.retrieve("cobalt", top_k=0), [])
        with self.assertRaises(ValueError):
            retriever.retrieve("cobalt", top_k=-1)

    def test_equal_scores_at_top_k_boundary_are_deterministic(self):
        retriever = Retriever(self.pack_for(["worker pension assistance"] * 20))
        for k in (1, 3, 19, 20, 100):
            expected = [f"schemes_{i}" for i in range(19, max(-1, 19 - k), -1)]
            first = retriever.retrieve("worker pension", top_k=k)
            second = retriever.retrieve("worker pension", top_k=k)
            self.assertEqual([c.chunk_id for c in first], expected)
            self.assertEqual(first, second)

    def test_ambiguous_gold_ids_and_empty_corpora_fail_at_startup(self):
        duplicate = self.pack_for(["Scheme ID: same\nEducation", "Scheme ID: same\nHousing"])
        with self.assertRaisesRegex(ValueError, "Duplicate chunk ID"):
            Retriever(duplicate)
        with self.assertRaisesRegex(ValueError, "No document sections"):
            Retriever(self.pack_for([]))
        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            Retriever(self.pack_for(["worker pension"], top_k=0))

    def test_saved_500_queries_keep_rankings_scores_and_gold_text(self):
        pack = DomainPack("welfare_demo")
        retriever = Retriever(pack)
        legacy = LegacyRetriever(pack)
        path = Path(__file__).resolve().parents[1] / "data/outputs/audit_qwen_500_cases.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.assertEqual(len(records), 500)
        original_text = dict(zip(legacy.chunk_ids, legacy.chunks))
        for record in records:
            with self.subTest(case=record["case_id"]):
                expected = legacy.retrieve(record["question"])
                actual = retriever.retrieve(record["question"])
                self.assertEqual([c.chunk_id for c in actual], [c.chunk_id for c in expected])
                self.assertEqual([c.text for c in actual], [c.text for c in expected])
                np.testing.assert_allclose([c.score for c in actual],
                                           [c.score for c in expected], atol=1e-12, rtol=1e-12)
                gold_ids = record["ground_truth"]["gold_chunk_ids"]
                gold = retriever.get_chunks_by_ids(gold_ids)
                self.assertEqual([c.chunk_id for c in gold], gold_ids)
                self.assertEqual([c.text for c in gold], [original_text[cid] for cid in gold_ids])

    def test_baseline_audit_is_unchanged(self):
        pack = DomainPack("welfare_demo")
        old = LegacyRetriever(pack)
        new = Retriever(pack)
        assistants = [NaiveBaselineAssistant(pack, r) for r in (old, new)]
        evaluator = Evaluator(pack)
        users = make_synthetic_users(pack.profile_schema, 30)
        cases = make_test_cases(pack, users, assign_splits([u["user_id"] for u in users]))
        self.assertEqual(len(cases), 150)
        for case in cases:
            records = []
            for assistant in assistants:
                answer = assistant.answer(case.question)
                gold_ids = case.ground_truth["gold_chunk_ids"]
                result = evaluator.evaluate(case, answer, recall_at_k(answer.retrieved_chunks, gold_ids),
                                            mrr(answer.retrieved_chunks, gold_ids))
                records.append({**case.to_dict(), "answer": answer.answer_text,
                                "retrieved_chunk_ids": [c.chunk_id for c in answer.retrieved_chunks],
                                "evaluation": result.to_dict()})
            self.assertEqual(records[0], records[1], case.case_id)


if __name__ == "__main__":
    unittest.main()
