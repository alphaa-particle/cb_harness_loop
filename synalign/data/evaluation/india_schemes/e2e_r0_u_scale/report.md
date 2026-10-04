# Pipeline evaluation: dev questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/usecase_questions.jsonl` (sha256 `7bfbe6d4e6d4`). Corpus: `data/external/scale_corpus/corpus.jsonl`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | How to apply | Documents | Benefits | Who is excluded | Eligibility and how to apply | Compare two | Which schemes | English | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 77.0% (68–84) | 91.7% (78–100) | 88.2% (76–97) | 50.0% (33–64) | 100.0% (100–100) | 83.3% (61–100) | 30.0% (0–70) | 25.0% (0–75) | 77.0% (68–84) | – | 66.77 |

## Answers

Correct: see the grading rules at the top of scripts/evaluate_pipeline.py.

| Condition | All | How to apply | Documents | Benefits | Who is excluded | Eligibility and how to apply | Compare two | Which schemes | Unsafe yes | Hindi answer to Hindi question | s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 79.8% (74–85) | 97.2% (92–100) | 88.2% (74–100) | 36.1% (19–53) | 97.2% (92–100) | 94.4% (83–100) | 90.0% (70–100) | 37.5% (0–75) | – | – | 1.968 |

## Paired differences (same questions, percentage points)

| Comparison | Questions | Difference | 95% interval |
|---|---|---|---|
