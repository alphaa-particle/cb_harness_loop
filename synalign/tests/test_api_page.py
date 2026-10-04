"""The web page, feedback saving, model status and the rules-only answer, through the API."""

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.api import create_app
from engine.domain_pack import DomainPack
from engine.enforcement import NO_EVIDENCE, RULES_ONLY
from engine.llm_assistant import RulesOnlyAssistant
from engine.retriever import Retriever
from support import asgi_call


class PageAndFeedbackTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.feedback = Path(temp.name) / "feedback.jsonl"

    def run_app(self, backend, steps):
        async def go():
            app = create_app(backend=backend, feedback_file=self.feedback)
            async with app.router.lifespan_context(app):
                return [await asgi_call(app, *step) for step in steps]
        return asyncio.run(go())

    def test_the_page_is_served_and_status_says_whether_an_ai_is_on(self):
        (status, page), (_, home) = self.run_app("rules_only", [("/ui", None, True), ("/",)])
        self.assertEqual(status, 200)
        self.assertIn("<title>Scheme Helper</title>", page)
        self.assertIn("textContent", page)                 # answers are inserted as text, never as HTML
        self.assertNotIn("innerHTML", page)
        self.assertEqual((home["assistant_backend"], home["model_ready"]), ("rules_only", False))
        self.assertIn("/feedback", home["routes"])
        _, (_, llama) = self.run_app("llama_cpp", [("/ui", None, True), ("/",)])
        self.assertEqual(llama["model"], "Qwen3.5-0.8B-Q8_0")   # reported even when its server is not running

    def test_feedback_is_appended_and_bad_input_is_refused(self):
        good = {"question": "Can I join APY?", "answer": "Yes.", "mode": "answered", "rating": "down",
                "note": "wrong age", "evidence_ids": ["apy__eligibility"]}
        results = self.run_app("rules_only", [("/feedback", good), ("/feedback", {**good, "rating": "meh"}),
                                              ("/feedback", {**good, "rating": "up", "note": ""})])
        self.assertEqual([r[0] for r in results], [200, 422, 200])
        saved = [json.loads(line) for line in self.feedback.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([s["rating"] for s in saved], ["down", "up"])
        self.assertEqual(saved[0]["note"], "wrong age")
        self.assertEqual(saved[0]["domain"], "welfare_demo")
        self.assertIn("corpus_sha256", saved[0])

    def test_rules_only_answers_show_the_rules_and_never_call_a_model(self):
        (_, answer), (_, nothing) = self.run_app("rules_only", [
            ("/ask", {"question": "I am 30 and earn 9000. Am I eligible for PMSYM?"}), ("/ask", {"question": "   "})])
        self.assertEqual(answer["mode"], RULES_ONLY)
        self.assertIn("Monthly income must be 15000 or below.", answer["answer"])
        self.assertTrue(answer["answer"].startswith(DomainPack("welfare_demo").prompts["rules_only"]))
        self.assertEqual(nothing["mode"], "empty")
        pack = DomainPack("welfare_demo")
        pack.eval_config["retrieval"]["min_score"] = 0.5
        self.assertEqual(RulesOnlyAssistant(pack, Retriever(pack)).answer("What is the capital of France?").mode,
                         NO_EVIDENCE)


if __name__ == "__main__":
    unittest.main()
