# Pipeline evaluation: test questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/questions.jsonl` (sha256 `b77812e2d673`). Corpus: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/documents`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | Fact | Typo | Situation | Eligibility | English | Hindi | Hindi in Latin letters | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 91.0% (87–94) | 87.5% (81–93) | 96.2% (92–100) | 84.7% (78–92) | 100.0% (100–100) | 97.2% (94–100) | 84.8% (78–91) | 84.6% (77–92) | 54.5% (43–68) | 41.93 |

## Answers

Correct: see the grading rules at the top of scripts/evaluate_pipeline.py.

| Condition | All | Fact | Typo | Situation | Eligibility | Not found (scheme absent) | Not found (off topic) | Unsafe yes | Hindi answer to Hindi question | s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 76.4% (72–80) | 75.8% (70–81) | 90.8% (84–96) | 51.4% (39–64) | 75.0% (69–81) | 80.0% (67–90) | 92.9% (79–100) | 10.5% (5–16) | 100.0% (100–100) | 1.274 |

## Paired differences (same questions, percentage points)

| Comparison | Questions | Difference | 95% interval |
|---|---|---|---|
