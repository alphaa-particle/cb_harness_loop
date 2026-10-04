# Pipeline evaluation: dev questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/questions.jsonl` (sha256 `b77812e2d673`). Corpus: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/documents`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | Fact | Typo | Situation | Eligibility | English | Hindi | Hindi in Latin letters | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 92.8% (89–96) | 92.0% (86–97) | 96.3% (91–100) | 80.6% (67–94) | 100.0% (100–100) | 97.4% (94–100) | 88.0% (79–95) | 88.3% (81–95) | 75.0% (60–100) | 33.11 |

## Answers

Correct: see the grading rules at the top of scripts/evaluate_pipeline.py.

| Condition | All | Fact | Typo | Situation | Eligibility | Not found (scheme absent) | Not found (off topic) | Unsafe yes | Hindi answer to Hindi question | s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 81.7% (77–85) | 84.0% (80–88) | 94.4% (89–100) | 55.6% (39–75) | 75.0% (60–90) | 100.0% (100–100) | 100.0% (100–100) | 9.6% (2–18) | 97.8% (95–100) | 1.208 |

## Paired differences (same questions, percentage points)

| Comparison | Questions | Difference | 95% interval |
|---|---|---|---|
