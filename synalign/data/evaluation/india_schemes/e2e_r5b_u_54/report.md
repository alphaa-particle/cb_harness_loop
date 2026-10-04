# Pipeline evaluation: dev questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/usecase_questions.jsonl` (sha256 `7bfbe6d4e6d4`). Corpus: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/documents`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | How to apply | Documents | Benefits | Who is excluded | Eligibility and how to apply | Compare two | Which schemes | English | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 86.0% (79–92) | 91.7% (78–100) | 79.4% (68–91) | 63.9% (44–81) | 100.0% (100–100) | 88.9% (72–100) | 100.0% (100–100) | 100.0% (100–100) | 86.0% (79–92) | – | 35.39 |

## Answers

Correct: see the grading rules at the top of scripts/evaluate_pipeline.py.

| Condition | All | How to apply | Documents | Benefits | Who is excluded | Eligibility and how to apply | Compare two | Which schemes | Unsafe yes | Hindi answer to Hindi question | s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 86.0% (81–90) | 97.2% (92–100) | 85.3% (74–94) | 47.2% (31–64) | 100.0% (100–100) | 100.0% (100–100) | 100.0% (100–100) | 100.0% (100–100) | – | – | 1.604 |

## Paired differences (same questions, percentage points)

| Comparison | Questions | Difference | 95% interval |
|---|---|---|---|
