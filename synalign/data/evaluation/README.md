# Saved measurements

## india_schemes/: the real pipeline

Every folder was written by `scripts/evaluate_pipeline.py` on the same frozen question file
(`domains/india_schemes/evaluation/questions.jsonl`, SHA-256 `b77812e2d673…`). Each holds `report.md`
(the tables), `report.json` (the numbers and settings) and `answers_<condition>.jsonl` (every answer
with its grade and the sections used).

| Folder | Split | What it is |
|---|---|---|
| `dev_search` | dev | search alone, 54 schemes; used to choose the fusion weights and the not-found floor |
| `dev_search_scale` | dev | search alone, 54 schemes among ~4,600 background schemes |
| `dev_round0` | dev | all six conditions with the starting instructions: the SynAlign baseline |
| `dev_round1` … `dev_round3` | dev | one change each, all UNDO by `compare_runs.py` (see the top-level README) |
| `dev_scale_k3`, `dev_scale_k5` | dev | fusion answers at full scale with 3 and 5 sections; 5 made no difference |
| `test_final` | test | all six conditions on the held-out questions: the reported results |
| `test_scale` | test | old search, meaning and fusion at full scale on the held-out questions |
| `logs/` | | console output of each run |

| `search_r0_*`, `usecase_search_r0_*` | dev | search alone before the search cycle (original and use-case questions) |
| `search_rounds/` | dev | one record per SynAlign search round: every number, fixed and broken question ids, the answer-check outcome |
| `e2e_r*_{q,u}_{54,scale}` | dev | each round's end-to-end answer check (q = original questions, u = use cases) |
| `design/` | dev | the diagnosis, the first design round's proposals and review, and prototype measurements |
| `exam_search_{before,after}_*`, `exam_r4_*` | test | the search cycle's exam: search for every set; answers for the original questions |
| `exam_before_questions_*` | test | copies of `test_final` / `test_scale`, used as the exam's "before" |
| `exam_after_questions_*` | test | answers with rounds 4 + 5b (5b was then undone) |

All answer folders were graded again with the final grader (`--regrade`), so they compare with each other.

## Older measurements

These JSON files are measurements, not application input or evidence documents.
See [the accuracy report](../../docs/RETRIEVAL_ACCURACY_REPORT.md) for methodology,
limitations, interpretation and reproduction commands.

- `research_manifest.json`: environment, dataset source/checksum and fixture provenance.
- `scifact_*.json`: independent 5,183-document / 300-query SciFact rankings and metrics.
- `welfare_stress_*.json`: fictional 5,000-section stress-test results, including failures.
- `welfare_dev_calibration.json`: threshold selection using only labelled development cases.
- `capacity_5000_schemes_20000_chunks*.json`: sequential capacity with top-3 and top-5 evidence.
- `capacity_rules_20000_*.json`: word/hybrid diagnostics of the same multi-section failure.
- `demo_system.json`: fresh naive audit summary and explicitly labelled saved-answer re-scoring,
  as measured before the grader accepted "work type".
- `saved_runs_regraded.json`: the same re-scoring with the current grader, for all four saved model runs.
- `context_5000_schemes_dev_calibration.json`: score-threshold choice on the development questions
  of the generated 5,000-scheme corpus (`scripts/build_scheme_corpus.py`).
- `context_5000_schemes_test.json`: whether the labelled text reaches the model's prompt for the
  held-out questions of that corpus, with rules first and with plain search, and every failure.

Stored temporary source paths identify this run; reproduce the generated corpus
using the checked-in scripts rather than relying on those directories persisting.
No SciFact source text or model weights are included here. Fixtures are authored
fictional examples and are not expert-reviewed welfare facts.
