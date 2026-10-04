"""Word ranking (BM25), meaning search, their merge, the vector cache and the llama.cpp clients.

No model is needed: meaning vectors come from a small stand-in embedder, and the
llama.cpp HTTP calls go to a stand-in server running in this process.
"""

import hashlib
import http.server
import json
import math
from pathlib import Path
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.corpus import CorpusPack
from engine.domain_pack import DomainPack
from engine.enforcement import ANSWERED, MODEL_ERROR
from engine.llama_cpp import LlamaEmbedder, LlamaServer, chat, is_ready
from engine.llm_assistant import LlamaCppAssistant
from engine.retriever import BM25, Retriever, tokens
from support import write_jsonl

SECTIONS = [
    {"chunk_id": "apy_rules", "scheme_id": "apy", "section": "rules", "title": "Atal Pension Yojana — Eligibility rules",
     "aliases": ["APY", "अटल पेंशन योजना"], "text": "- Indian citizens aged 18 to 40.\n- Must have a savings bank account."},
    {"chunk_id": "apy_benefits", "scheme_id": "apy", "section": "benefits", "title": "Atal Pension Yojana — Benefits",
     "aliases": ["APY"], "text": "- A guaranteed monthly pension of Rs 1,000 to Rs 5,000 after the age of 60."},
    {"chunk_id": "pmay_rules", "scheme_id": "pmay", "section": "rules", "title": "Pradhan Mantri Awas Yojana — Eligibility rules",
     "aliases": ["PMAY", "आवास योजना"], "text": "- Families without a pucca house anywhere in India."},
    {"chunk_id": "ssy_rules", "scheme_id": "ssy", "section": "rules", "title": "Sukanya Samriddhi Yojana — Eligibility rules",
     "aliases": ["SSY"], "text": "- A savings account opened for a girl child below 10 years of age."},
]
# The stand-in embedder's notion of meaning: words that mean the same thing share a vector slot.
CONCEPTS = {"pension": "pension", "पेंशन": "pension", "old": "pension", "budhapa": "pension", "बुढ़ापे": "pension",
            "house": "house", "pucca": "house", "home": "house", "awas": "house", "makaan": "house",
            "girl": "girl", "daughter": "girl", "beti": "girl", "बेटी": "girl"}


class FakeEmbedder:
    """Deterministic vectors: one slot per concept, the rest hashed. Counts its calls."""

    model_id = "fake-embedder-v1"

    def __init__(self):
        self.embedded = 0

    def _vector(self, text):
        v = np.zeros(64, dtype=np.float32)
        for word in tokens(text):
            concept = CONCEPTS.get(word)
            slot = ["pension", "house", "girl"].index(concept) if concept else 3 + int(hashlib.md5(word.encode()).hexdigest(), 16) % 61
            v[slot] += 3.0 if concept else 0.2
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text):
        return self._vector(text)


class SearchMethodTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.corpus = write_jsonl(self.path / "corpus.jsonl", SECTIONS)

    def retriever(self, method, top_k=3, min_score=0.0, embedder=None, cache=None, glossary=None):
        pack = CorpusPack(self.corpus, top_k=top_k, method=method, min_score=min_score)
        if glossary:
            pack.glossary = glossary
        return Retriever(pack, embedder=embedder or FakeEmbedder(), cache_dir=cache)

    def test_hindi_words_stay_whole(self):
        self.assertEqual(tokens("बुढ़ापे में पेंशन, APY-योजना!"), ["बुढ़ापे", "में", "पेंशन", "apy", "योजना"])

    def test_bm25_matches_the_formula_worked_by_hand(self):
        texts = ["pension pension scheme", "house scheme", "girl savings account scheme"]
        scores = BM25(texts).scores("pension scheme")
        n, lengths = 3, [3, 2, 4]
        avg = sum(lengths) / n
        def term(tf, df, dl):
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            return idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avg))
        expected = [term(2, 1, 3) + term(1, 3, 3), term(1, 3, 2), term(1, 3, 4)]
        np.testing.assert_allclose(scores, expected, rtol=1e-9)

    def test_meaning_search_finds_a_scheme_that_shares_no_words_with_the_question(self):
        question = "beti ke liye kuch hai"     # romanised Hindi: no word in common with the scheme text
        self.assertNotIn("ssy_rules", [c.chunk_id for c in self.retriever("bm25").retrieve(question)])
        self.assertEqual(self.retriever("dense").retrieve(question)[0].chunk_id, "ssy_rules")
        self.assertEqual(self.retriever("fusion").retrieve(question)[0].chunk_id, "ssy_rules")
        hindi = self.retriever("fusion").retrieve("बुढ़ापे में पेंशन")
        self.assertEqual(hindi[0].scheme_id, "apy")

    def test_fusion_keeps_exact_name_matches_and_reports_meaning_similarity(self):
        found = self.retriever("fusion").retrieve("Who can join APY?")
        self.assertEqual(found[0].scheme_id, "apy")
        meaning = FakeEmbedder()
        expected = float(meaning.embed_documents([Retriever(CorpusPack(self.corpus, method="char_tfidf")).documents[0].search_text])[0]
                          @ meaning.embed_query("Who can join APY?"))
        apy_rules = next(c for c in found if c.chunk_id == "apy_rules")
        self.assertAlmostEqual(apy_rules.score, expected, places=5)

    def test_a_question_far_from_every_scheme_gets_nothing_once_a_threshold_is_set(self):
        self.assertTrue(self.retriever("fusion").retrieve("cricket score today"))
        self.assertEqual(self.retriever("fusion", min_score=0.5).retrieve("cricket score today"), [])
        self.assertTrue(self.retriever("fusion", min_score=0.5).retrieve("old age pension"))
        self.assertEqual(self.retriever("dense", min_score=0.99).retrieve("old age pension"), [])

    def test_rules_still_come_first_with_meaning_search(self):
        evidence = self.retriever("fusion", top_k=2).retrieve_evidence("how much monthly pension after 60")
        self.assertEqual([c.chunk_id for c in evidence], ["apy_rules", "apy_benefits"])

    def test_glossary_words_help_word_matching_only(self):
        plain = self.retriever("bm25").retrieve("makaan chahiye")
        helped = self.retriever("bm25", glossary={"makaan": "house pucca"}).retrieve("makaan chahiye")
        self.assertEqual(plain, [])
        self.assertEqual(helped[0].chunk_id, "pmay_rules")
        pack = DomainPack("welfare_demo")
        self.assertEqual(pack.glossary, {})        # a pack without glossary.yaml is unchanged

    def test_vectors_are_cached_and_only_changed_sections_are_embedded_again(self):
        cache = self.path / "index"
        first = FakeEmbedder()
        self.retriever("dense", embedder=first, cache=cache)
        self.assertEqual(first.embedded, 4)
        again = FakeEmbedder()
        self.retriever("dense", embedder=again, cache=cache)
        self.assertEqual(again.embedded, 0)
        edited = [dict(s) for s in SECTIONS]
        edited[2]["text"] = "- Families without a pucca house, with income below Rs 3 lakh."
        write_jsonl(self.corpus, edited)
        after_edit = FakeEmbedder()
        self.retriever("dense", embedder=after_edit, cache=cache)
        self.assertEqual(after_edit.embedded, 1)
        other_corpus = self.path / "other.jsonl"
        write_jsonl(other_corpus, [{"chunk_id": "x", "text": "An unrelated scheme for fishers."}])
        Retriever(CorpusPack(other_corpus, method="dense"), embedder=FakeEmbedder(), cache_dir=cache)
        back_again = FakeEmbedder()
        self.retriever("dense", embedder=back_again, cache=cache)
        self.assertEqual(back_again.embedded, 0)    # building another corpus did not evict these vectors
        other_model = FakeEmbedder()
        other_model.model_id = "another-model"
        self.retriever("dense", embedder=other_model, cache=cache)
        self.assertEqual(other_model.embedded, 4)   # vectors from different models are never mixed

    def test_unknown_methods_and_old_methods(self):
        with self.assertRaisesRegex(ValueError, "Unknown retrieval method"):
            Retriever(CorpusPack(self.corpus, method="magic"))
        old = Retriever(CorpusPack(self.corpus, method="char_tfidf"))
        self.assertIsNone(old.dense)                 # word methods never touch an embedder


def scheme(sid, name, sections, aliases=()):
    return [{"chunk_id": f"{sid}__{key}", "scheme_id": sid, "section": label, "title": f"{name} — {key}",
             "aliases": list(aliases), "text": text} for key, label, text in sections]


EVIDENCE_CORPUS = (
    scheme("loom", "Loom Renewal Grant (LRG)", [
        ("overview", "overview", "A grant for handloom weavers to replace an old loom."),
        ("eligibility", "rules", "- Weavers aged 21 to 50.\n- Family income below Rs 90,000 a year."),
        ("eligibility_2", "rules", "Not eligible:\n- Weavers who got the grant in the last five years."),
        ("benefits", "benefits", "- Rs 4,000 paid into the weaver's bank account."),
        ("how_to_apply", "application", "- Apply at the block handloom office with the form."),
        ("documents", "documents", "- Weaver identity card.\n- Bank passbook.")])
    + scheme("seed", "Seed Kit Scheme", [
        ("overview", "overview", "Free seed kits for small farmers before the sowing season."),
        ("eligibility", "rules", "- Small and marginal farmers with land records."),
        ("benefits", "benefits", "- One seed kit per family each season."),
        ("how_to_apply", "application", "- Register at the krishi office.")], aliases=["Seed Kit"])
    + scheme("seed_plus", "Seed Kit Plus Scheme", [
        ("overview", "overview", "Seed Kit Plus gives seed kits and fertiliser to tenant farmers."),
        ("eligibility", "rules", "- Tenant farmers with a lease paper.")])
    + scheme("dp_goa", "Disability Pension", [("eligibility", "rules", "- Construction workers in Goa with a disability.")])
    + scheme("dp_bihar", "Disability Pension", [("eligibility", "rules", "- Construction workers in Bihar with a disability.")])
)


class EvidenceSettingsTests(unittest.TestCase):
    """rules_count_once, named_schemes and overview_last (all off by default)."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.corpus = write_jsonl(Path(temp.name) / "corpus.jsonl", EVIDENCE_CORPUS)

    def evidence(self, question, **settings):
        pack = CorpusPack(self.corpus, top_k=3, method="bm25")
        pack.eval_config["retrieval"].update(settings)
        return [c.chunk_id for c in Retriever(pack).retrieve_evidence(question)]

    def test_rules_cut_into_parts_take_one_slot(self):
        today = self.evidence("What benefits does the Loom Renewal Grant give?")
        self.assertEqual(today[:2], ["loom__eligibility", "loom__eligibility_2"])
        self.assertEqual(len(today), 3)                       # the two rule parts left one slot
        once = self.evidence("What benefits does the Loom Renewal Grant give?", rules_count_once=True)
        self.assertEqual(once[:2], ["loom__eligibility", "loom__eligibility_2"])
        self.assertEqual(len(once), 4)                        # rules are one slot: two more sections fit
        self.assertIn("loom__benefits", once)

    def test_schemes_the_question_names_get_the_slots_and_comparisons_get_both(self):
        both = self.evidence("What is the difference between the Loom Renewal Grant and the Seed Kit Scheme?",
                             rules_count_once=True, named_schemes=True)
        self.assertIn("loom__eligibility", both)
        self.assertIn("seed__eligibility", both)
        one = self.evidence("How do I apply for LRG?", rules_count_once=True, named_schemes=True)
        self.assertEqual({c.split("__")[0] for c in one}, {"loom"})

    def test_a_name_shared_by_several_schemes_names_none_and_a_short_name_never_pushes_out_the_top_one(self):
        pack = CorpusPack(self.corpus, top_k=3, method="bm25")
        pack.eval_config["retrieval"]["named_schemes"] = True
        names = dict(Retriever(pack)._names)
        self.assertNotIn("disability pension", names)         # two schemes have it
        self.assertEqual(names["seed kit"], "seed")
        # "Seed Kit Plus" contains the name "Seed Kit"; search ranks Seed Kit Plus first, so it stays
        found = self.evidence("Who can get Seed Kit Plus as a tenant farmer with a lease paper?",
                              rules_count_once=True, named_schemes=True)
        self.assertIn("seed_plus__eligibility", found)

    def test_for_a_named_scheme_the_overview_only_fills_free_slots(self):
        settings = {"rules_count_once": True, "named_schemes": True}
        question = "How do I apply for the Loom Renewal Grant, the grant for an old handloom?"
        self.assertIn("loom__overview", self.evidence(question, **settings))
        last = self.evidence(question, **settings, overview_last=True)
        self.assertNotIn("loom__overview", last)
        self.assertIn("loom__how_to_apply", last)
        # a scheme with fewer specific sections than slots still gets its overview
        self.assertIn("seed_plus__overview", self.evidence("Tell me about Seed Kit Plus Scheme", **settings,
                                                           overview_last=True))

    def test_a_named_scheme_that_is_missing_replaces_only_the_last_slot(self):
        question = "How is the Loom Renewal Grant for handloom weavers with an old loom different from the Seed Kit Scheme?"
        today = self.evidence(question)
        self.assertNotIn("seed", {c.split("__")[0] for c in today})          # the second scheme was left out
        patched = self.evidence(question, add_named_schemes=True)
        self.assertEqual(patched[:-1], today[:-1])                           # nothing else moves
        self.assertEqual(patched[-1], "seed__eligibility")
        for unchanged in ("How do I apply for LRG?", "Which papers does a weaver need?"):
            self.assertEqual(self.evidence(unchanged, add_named_schemes=True), self.evidence(unchanged))

    def test_a_name_that_continues_into_another_schemes_longer_name_does_not_count(self):
        pack = CorpusPack(self.corpus, top_k=3, method="bm25")
        pack.eval_config["retrieval"]["add_named_schemes"] = True
        retriever = Retriever(pack)
        order = retriever._rank("x", 30)[1]
        self.assertEqual(retriever._named("Can a tenant get Seed Kit Plus?", order), [])      # "kit plus"
        self.assertEqual(retriever._named("Can I get a Seed Kit this season?", order), ["seed"])


def evidence_v2(corpus, question, **settings):
    pack = CorpusPack(corpus, top_k=3, method="bm25")
    pack.eval_config["retrieval"].update(add_named_schemes=True, name_rules=2, **settings)
    return Retriever(pack)


class NameRulesTwoTests(unittest.TestCase):
    """Round 5b's corrections to name matching, one test per defect the code review found."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        extra = (scheme("pens", "Old Age Pension Scheme (OAPS)", [
                     ("overview", "overview", "A monthly pension for people over 60. Seed Kit holders may also apply."),
                     ("eligibility", "rules", "- Aged 60 or more.")], aliases=["Vridha Pension (old name)"])
                 + scheme("grant", "Medical Relief Grant (Medical Treatment)", [
                     ("eligibility", "rules", "- Families needing medical treatment.")])
                 + scheme("tools", "Tool Kit Grant", [("overview", "overview", "Free tool kits for carpenters."),
                                                       ("benefits", "benefits", "- A tool kit worth Rs 5,000.")]))
        self.corpus = write_jsonl(Path(temp.name) / "corpus.jsonl", EVIDENCE_CORPUS + extra)

    def names(self, question):
        return evidence_v2(self.corpus, question)._named_v2(question)

    def test_bracketed_descriptions_and_alias_notes_are_read_like_titles(self):
        self.assertEqual(self.names("Can I get help for medical treatment?"), [])          # a description
        self.assertEqual(self.names("Is Vridha Pension still given?"), ["pens"])           # "(old name)" dropped
        self.assertEqual(self.names("How do I apply for OAPS?"), ["pens"])                # a bracketed short form

    def test_joining_words_do_not_block_a_name_and_a_hindi_full_stop_does_not_stick_to_it(self):
        self.assertEqual(self.names("Is the Seed Kit Scheme for small farmers?"), ["seed"])
        self.assertEqual(self.names("मुझे Seed Kit चाहिए।"), ["seed"])
        self.assertEqual(self.names("Can a tenant get Seed Kit Plus?"), [])                # still part of a longer name

    def test_the_least_useful_slot_is_given_up_and_never_a_named_schemes_rules(self):
        retriever = evidence_v2(self.corpus, "")
        loom = [retriever._chunk_index[c] for c in ("loom__eligibility", "loom__overview", "loom__documents")]
        order = np.array(loom)
        # a named scheme comes in: the overview goes, the documents section the question asked about stays
        out = retriever._add_missing("Which documents does the Loom Renewal Grant need, unlike the Seed Kit Scheme?",
                                     order, list(loom), 3)
        self.assertEqual([retriever.documents[i].chunk_id for i in out],
                         ["loom__eligibility", "seed__eligibility", "loom__documents"])

    def test_a_named_scheme_without_rules_brings_its_best_section(self):
        retriever = evidence_v2(self.corpus, "")
        loom = [retriever._chunk_index[c] for c in ("loom__eligibility", "loom__overview", "loom__documents")]
        out = retriever._add_missing("Compare the Loom Renewal Grant with the Tool Kit Grant",
                                     np.array(loom), list(loom), 3)
        self.assertIn(retriever.documents[out[1]].scheme_id, {"tools"})


class StandInServer(http.server.BaseHTTPRequestHandler):
    """Answers like llama-server: /health, /props, /v1/embeddings, /v1/chat/completions."""

    replies = []
    requests = []

    def log_message(self, *args):
        pass

    def _send(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"status": "ok"} if self.path == "/health" else {"model_path": "/m/Qwen3-Embedding-0.6B-Q8_0.gguf"})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StandInServer.requests.append((self.path, payload))
        if self.path == "/v1/embeddings":
            data = [{"index": i, "embedding": [float(len(t)), 1.0, 0.0]} for i, t in enumerate(payload["input"])]
            self._send({"data": list(reversed(data))})          # out of order on purpose
        else:
            self._send(StandInServer.replies.pop(0))


class LlamaClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A plain TCP server: HTTPServer looks up this machine's host name, which can take half a minute.
        cls.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), StandInServer)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        StandInServer.requests.clear()
        StandInServer.replies.clear()

    def test_embedder_prefixes_only_questions_keeps_order_and_normalises(self):
        embedder = LlamaEmbedder(self.url, instruction="Find the scheme", batch_size=2)
        self.assertTrue(is_ready(self.url))
        self.assertEqual(embedder.model_id, "Qwen3-Embedding-0.6B-Q8_0.gguf")
        vectors = embedder.embed_documents(["a", "bbb", "cc"])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, rtol=1e-6)
        np.testing.assert_allclose(vectors[:, 0] / vectors[:, 1], [1, 3, 2], rtol=1e-6)   # input order kept
        self.assertEqual([p["input"] for _, p in StandInServer.requests], [["a", "bbb"], ["cc"]])
        embedder.embed_query("pension?")
        self.assertEqual(StandInServer.requests[-1][1]["input"], ["Instruct: Find the scheme\nQuery:pension?"])

    def test_chat_is_greedy_seeded_uncached_and_thinking_off(self):
        StandInServer.replies.append({"choices": [{"message": {"content": "Yes."}}]})
        self.assertEqual(chat([{"role": "user", "content": "hi"}], url=self.url, max_tokens=50), "Yes.")
        sent = StandInServer.requests[-1][1]
        self.assertEqual((sent["temperature"], sent["seed"], sent["cache_prompt"], sent["max_tokens"]), (0.0, 42, False, 50))
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})

    def test_the_llama_assistant_answers_through_the_server_and_falls_back_when_it_fails(self):
        pack = DomainPack("welfare_demo")
        retriever = Retriever(pack)
        assistant = LlamaCppAssistant(pack, retriever, url=self.url)
        StandInServer.replies.append({"choices": [{"message": {"content": "<think></think>Needs confirmation: PMSYM."}}]})
        answer = assistant.answer("I am 30 and earn 9000. Am I eligible for PMSYM?")
        self.assertEqual((answer.mode, answer.answer_text), (ANSWERED, "Needs confirmation: PMSYM."))
        self.assertIn("Monthly income must be 15000 or below.", StandInServer.requests[-1][1]["messages"][1]["content"])
        StandInServer.replies.append({"choices": [{"message": {"content": ""}}]})
        self.assertEqual(assistant.answer("Am I eligible for PMSYM?").mode, MODEL_ERROR)
        down = LlamaCppAssistant(pack, retriever, url="http://127.0.0.1:9")    # nothing listens there
        failed = down.answer("Am I eligible for PMSYM?")
        self.assertEqual(failed.mode, MODEL_ERROR)
        self.assertIn("Monthly income must be 15000 or below.", failed.answer_text)

    def test_a_server_that_was_already_running_is_left_running(self):
        background = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with tempfile.TemporaryDirectory() as d:
                server = LlamaServer("unused.gguf", self.server.server_address[1], [], "decoder")
                server.log_path = Path(d) / "decoder.log"
                server.pid_file.write_text(str(background.pid))     # as left by serve_models.py --detach
                server.start()                                      # already answering: nothing is started
                server.stop()
                self.assertIsNone(background.poll())
                self.assertTrue(server.pid_file.is_file())
                server.stop_detached()                              # serve_models.py --stop
                self.assertEqual(background.wait(timeout=10), -15)
                self.assertFalse(server.pid_file.is_file())
        finally:
            background.kill()


if __name__ == "__main__":
    unittest.main()
