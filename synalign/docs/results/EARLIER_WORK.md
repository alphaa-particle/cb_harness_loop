# Earlier work: the two-scheme demo and the invented 5,000-scheme corpus

These results come from before the real Indian schemes pipeline. They are kept because they explain
why the pipeline is built the way it is: rules first, a meaning search, and measuring every change.
The current results are in the [top-level README](../../README.md).

## Results

### A. Does the right scheme's text reach the model? (5,000 schemes, no model needed)

The corpus is 5,000 invented schemes with four sections each, 20,000 sections in total. Search method, sections per prompt and score threshold were chosen on a separate set of development questions. These are the held-out test questions.

| Question type | Questions | Right text in prompt, rules first | Plain search |
|---|---:|---:|---:|
| Asks for a scheme's rules by name | 4,500 | 4,500 (100%) | 0 |
| Asks how to apply (needs rules and application section) | 500 | 500 (100%) | 0 |
| Named, with keyboard typos | 500 | 500 (100%) | 16 |
| Describes the person and need, no scheme name | 500 | 463 (92.6%) | 328 |
| **All answerable** | **6,000** | **5,963 (99.4%)** | **344 (5.7%)** |

| Question the documents cannot answer | Questions | Model not called |
|---|---:|---:|
| Nothing to do with schemes | 20 | 20 |
| Names a scheme in a state the corpus does not cover | 270 | 0 |

- Search takes about 1.5 ms per question and the index builds in about 3 seconds, on one machine with no database.
- A prompt averages 3 sections and about 1,600 characters.
- **All 37 misses on described questions are one wording.** "Getting hurt at work" shares no words with "insurance cover for accidents at work", so a sibling scheme is retrieved. Search here matches words, not meaning.
- **A scheme that is not in the documents is not detected.** The nearest scheme is sent instead. Its rules, which name the state it applies to, are always first in the prompt, but whether the model then says "that is for a different state" is not tested here.
- The older set of 47 hand-written test questions gives the same result as before this change: 38 of 41 answerable, 3 of 6 unanswerable refused.

These schemes are invented and the questions follow templates. The numbers show the mechanism works at this size. They are not a measure of real user questions on real schemes.

Raw results: [context_5000_schemes_test.json](../../data/evaluation/context_5000_schemes_test.json). Method and older retrieval measurements: [RETRIEVAL_ACCURACY_REPORT.md](../../docs/RETRIEVAL_ACCURACY_REPORT.md).

### B. Model runs on the two-scheme demo

All runs below were made **before** rules-first enforcement existed. Each is 500 questions (100 users × 5 styles) unless noted.

| Run | What changed | Passed, as first graded | Passed, graded by current code | Automatic fails |
|---|---|---:|---:|---:|
| Naive baseline | no AI, fixed template (1,500 questions) | 843 (56.2%) | 843 (56.2%) | 0 |
| Qwen3 0.6B | plain model with documents | 51 (10.2%) | answers not saved | 179 |
| Qwen3.5 0.8B | bigger model | 136 (27.2%) | 143 (28.6%) | 304 |
| Gemma 1B | different model | 111 (22.2%) | answers not saved | 4 |
| + router and rules | eligibility decided by hand-written code | 141 (28.2%) | 241 (48.2%) | 146 |
| + rule triggering fixed | the hand-written rules actually fire | 269 (53.8%) | 487 (97.4%) | 0 |
| + LoRA fine-tune | model trained on ideal answers | 269 (53.8%) | 335 (67.0%) | 0 |

What these runs showed:

1. **A bigger model was less safe.** Qwen3.5 passed more often than Qwen3 but wrongly confirmed eligibility 304 times against 179.
2. **Hand-written rules beat every model.** In the 97.4% run, code decided eligibility and wrote the answer from a template; the model was skipped for those questions. That code covered only the two demo schemes and has been removed from the tree. It is in git history: `git show 712acf1:synalign/engine/qwen_assistant_rule_backup.py`.
3. **A readability check was refusing readable questions.** It treated long words with few vowels as keyboard mashing, and "monthly" is one, so a question with one more typo got "I could not understand that". All 13 remaining failures of the 97.4% run were this. It is fixed in [llm_assistant.py](../../engine/llm_assistant.py); the saved answers predate the fix.
4. **The grader was wrong three times, and each fix moved the results.** It ignored requests that did not end in a question mark ("Please share your age"), it did not recognise `worker_type` written with an underscore, and it did not recognise "work type", which is the exact phrase the test questions themselves use. The third fix, made in this change, moves the fine-tuned model from 278 to 335 passes. All three are now in [evaluator.py](../../engine/evaluator.py) and [aliases.yaml](../../domains/welfare_demo/aliases.yaml).

The middle column is a re-grading of saved answers, not a new model run. Reproduce it with `python scripts/regrade_audits.py data/outputs/<file>.jsonl`; the output is in [saved_runs_regraded.json](../../data/evaluation/saved_runs_regraded.json).

### What was not known then, and what answered it

- **Whether a model given the right evidence answers correctly at 5,000 schemes.** Answered by the real
  pipeline: with the right sections in front of it, Qwen3.5-0.8B answers 81.5% of the held-out questions
  correctly, and the full pipeline 78% of questions naming one of 4,600 schemes (see the top-level README).
- **Real questions on real schemes.** The corpus above is invented; the real pipeline uses 54 researched
  schemes and about 4,600 real background schemes.
- **Meaning, not just words.** The 37 misses above were paraphrases; the real pipeline adds a meaning search.

## Names that changed in the first cleanup

Older reports in `docs/` use the old names.

Older reports in `docs/` use the old names.

| Old | Now |
|---|---|
| `scripts/run_audit_qwen.py`, `scripts/run_qwen_local_progress.py` | `scripts/run_audit.py --backend transformers` |
| `scripts/smoke_test_qwen.py` | `scripts/smoke_test.py` |
| `scripts/make_sft_from_audit.py`, old `scripts/make_training_data.py` | `scripts/make_training_data.py` |
| `scripts/evaluate_demo_system.py`, `scripts/rescore_followup_fix.py` | `scripts/regrade_audits.py` |
| `scripts/benchmark_retrieval.py` | `scripts/build_scheme_corpus.py` + `scripts/evaluate_context.py` |
| `engine/retrieval_metrics.py` | merged into `engine/retrieval_evaluation.py` |
| `tests/test_raw_qwen_behavior.py`, `synalign_poc.py`, `CODEBASE_ANALYSIS.md` | removed; see git history |
| `engine/qwen_assistant.py` | `engine/llm_assistant.py` |
| `engine/qwen_assistant_rule_backup.py` | removed; see git history |
| `requirements-qwen.txt` | `requirements-model.txt` |
| backend name `qwen` | `transformers` |
