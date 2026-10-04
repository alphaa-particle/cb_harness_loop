"""Users drawn from the Jansankhya population file, followed through the whole pipeline."""

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.audit import run_audit
from engine.assistant import NaiveBaselineAssistant
from engine.diagnosis import attribute_failures, failure_type_summary
from engine.domain_pack import DomainPack
from engine.enforcement import ANSWERED
from engine.evaluator import Evaluator
from engine.llm_assistant import TransformersAssistant
from engine.retriever import Retriever
from engine.synthesis import make_synthetic_users
from engine.training_data import make_example, make_preference_pair, scheme_names

POPULATION_FILE = ROOT.parent / "jansankhya" / "synthetic_population_imputed.csv"


class PopulationUserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = DomainPack("welfare_demo")
        cls.population = cls.pack.load_population()
        cls.by_id = cls.population.set_index("worker_id")

    def users(self, n=200, seed=42):
        return make_synthetic_users(self.pack.profile_schema, n, seed, population=self.population)

    def test_the_file_is_the_one_the_schema_points_to_and_is_complete(self):
        self.assertTrue(POPULATION_FILE.is_file())
        self.assertEqual(len(self.population), 10000)
        self.assertTrue(self.population["worker_id"].is_unique)
        needed = {"worker_id", *(source["column"] for source in self.pack.profile_schema["population"]["fields"].values())}
        self.assertLessEqual(needed, set(self.population.columns))
        self.assertFalse((self.population[sorted(needed)].astype(str) == "").any().any())

    def test_words_like_none_are_kept_as_values_not_read_as_blanks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "people.csv"
            path.write_text("worker_id,education\n0,None\n1,NA\n", encoding="utf-8")
            schema = {**self.pack.profile_schema, "population": {"file": str(path)}}
            with patch.object(self.pack, "profile_schema", schema):
                self.assertEqual(list(self.pack.load_population()["education"]), ["None", "NA"])

    def test_each_user_is_one_distinct_worker_with_fields_taken_from_that_row(self):
        users = self.users()
        self.assertEqual(len({u["worker_id"] for u in users}), len(users))
        for user in users:
            row = self.by_id.loc[user["worker_id"]]
            self.assertEqual(user["age"], int(row["age"]))
            self.assertEqual(user["state"], row["state"])
            self.assertEqual(user["income"], max(int(round(row["annual_wage_inr"] / 12 / 100) * 100), 100))
            self.assertEqual(user["worker_type"],
                             "salaried" if row["contract_status"] == "written" else "unorganised_worker")
            self.assertIn(user["epfo"], (True, False, None))   # not in the file: still sampled
            json.dumps(user)                                    # plain values, ready for the audit file

    def test_the_same_seed_gives_the_same_users_and_another_seed_does_not(self):
        self.assertEqual(self.users(), self.users())
        self.assertNotEqual([u["worker_id"] for u in self.users()], [u["worker_id"] for u in self.users(seed=7)])

    def test_bad_requests_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "population has 10000"):
            self.users(n=10001)
        broken = self.population.head(5).assign(contract_status="handshake")
        with self.assertRaisesRegex(ValueError, "No mapping for contract_status='handshake'"):
            make_synthetic_users(self.pack.profile_schema, 3, population=broken)
        with self.assertRaisesRegex(ValueError, "Unknown user source"):
            run_audit(n_users=2, backend="naive", user_source="census")

    def test_invented_users_are_exactly_the_ones_the_saved_runs_used(self):
        saved = {}
        for line in (ROOT / "data/outputs/audit_qwen_500_cases.jsonl").read_text().splitlines():
            record = json.loads(line)
            saved[record["user_id"]] = record["profile"]
        self.assertEqual(len(saved), 100)
        self.assertEqual(make_synthetic_users(self.pack.profile_schema, 100), [saved[i] for i in range(100)])


class PopulationEndToEndTests(unittest.TestCase):
    """Population file -> users -> questions -> answer key -> answers -> grades -> diagnosis -> training data."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.pack = DomainPack("welfare_demo")
        cls.retriever = Retriever(cls.pack)
        cls.population = cls.pack.load_population().set_index("worker_id")
        run_audit(n_users=120, backend="naive", label="population", output_dir=Path(cls.temp.name),
                  user_source="population")
        lines = (Path(cls.temp.name) / "audit_population.jsonl").read_text(encoding="utf-8").splitlines()
        cls.records = [json.loads(line) for line in lines]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_every_user_asks_five_ways_and_stays_in_one_split(self):
        self.assertEqual(len(self.records), 600)
        self.assertEqual(Counter(r["condition"] for r in self.records),
                         dict.fromkeys(["clean", "vague", "missing_info", "typo_heavy", "misleading"], 120))
        splits = {}
        for record in self.records:
            self.assertEqual(splits.setdefault(record["user_id"], record["split"]), record["split"])
        self.assertEqual(Counter(splits.values()), {"train": 72, "dev": 24, "test": 24})

    def test_questions_carry_the_workers_own_facts(self):
        for record in (r for r in self.records if r["condition"] == "clean"):
            row = self.population.loc[record["profile"]["worker_id"]]
            self.assertIn(f"I am {int(row['age'])} years old.", record["question"])
            self.assertIn(f"My monthly income is {record['profile']['income']}.", record["question"])
            self.assertEqual(record["visible"]["age"], int(row["age"]))

    def test_the_answer_key_follows_the_scheme_rules_for_the_workers_facts(self):
        seen = Counter()
        for record in (r for r in self.records if r["condition"] == "clean"):
            profile, truth = record["profile"], record["ground_truth"]
            ruled_out = (not 18 <= profile["age"] <= 40 or profile["income"] > 15000
                         or profile["worker_type"] != "unorganised_worker" or profile["epfo"] is True)
            if ruled_out:
                expected = "ineligible"
            elif profile["epfo"] is None:
                expected = "unknown_due_to_missing_info"
            else:
                expected = "likely_eligible"
            self.assertIn("scheme_pmsym", truth[expected], record["case_id"])
            seen[expected] += 1
        self.assertEqual(set(seen), {"ineligible", "unknown_due_to_missing_info", "likely_eligible"})

    def test_every_answer_is_graded_on_the_right_evidence(self):
        for record in self.records:
            self.assertEqual(record["retrieved_chunk_ids"] and set(record["retrieved_chunk_ids"]),
                             {"scheme_pmsym", "scheme_eshram"})
            self.assertEqual(record["evaluation"]["retrieval_recall"], 1.0)
            self.assertEqual(record["evaluation"]["gate_violations"], [])
        passed = sum(r["evaluation"]["passed"] for r in self.records)
        self.assertTrue(0 < passed < len(self.records))

    def test_failures_can_be_diagnosed(self):
        failures = failure_type_summary(self.records)
        self.assertIn("missing_followup_question", set(failures["failure_type"]))
        attribution = attribute_failures(self.records, self.retriever,
                                         NaiveBaselineAssistant(self.pack, self.retriever), Evaluator(self.pack))
        self.assertEqual(set(attribution["verdict"]), {"generation_failure"})

    def test_a_model_assistant_gets_the_scheme_rules_for_these_questions(self):
        assistant = TransformersAssistant(self.pack, self.retriever, _model=object(), _tokenizer=object())
        with patch.object(assistant, "_generate", return_value="Needs confirmation: PMSYM.") as model:
            for record in self.records:
                answer = assistant.answer(record["question"])
                prompt = model.call_args.args[0][1]["content"]
                self.assertEqual(answer.mode, ANSWERED)
                self.assertIn("Monthly income must be 15000 or below.", prompt)
                self.assertIn(record["question"], prompt)

    def test_training_data_is_built_for_every_record_and_keeps_the_split(self):
        names = scheme_names(self.pack, self.retriever)
        examples = [make_example(record, self.pack, self.retriever, names) for record in self.records]
        self.assertEqual(Counter(e["split"] for e in examples), {"train": 360, "dev": 120, "test": 120})
        for record, example in zip(self.records, examples):
            self.assertIn(record["question"], example["messages"][1]["content"])
            target = example["messages"][2]["content"]
            for scheme in record["ground_truth"]["ineligible"]:
                self.assertIn(record["ground_truth"]["ineligibility_reasons"][scheme], target)
        pairs = [make_preference_pair(r, e) for r, e in zip(self.records, examples) if r["split"] == "train"]
        self.assertTrue(any(pairs))


if __name__ == "__main__":
    unittest.main()
