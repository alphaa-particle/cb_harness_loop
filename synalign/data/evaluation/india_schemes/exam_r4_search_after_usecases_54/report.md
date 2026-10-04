# Pipeline evaluation: test questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/usecase_questions.jsonl` (sha256 `7bfbe6d4e6d4`). Corpus: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/documents`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | How to apply | Documents | Benefits | Who is excluded | Eligibility and how to apply | Compare two | Which schemes | English | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 82.0% (78–86) | 90.3% (82–97) | 85.7% (79–93) | 36.1% (24–49) | 100.0% (100–100) | 94.4% (86–100) | 96.4% (89–100) | 100.0% (100–100) | 82.0% (78–86) | – | 34.92 |
