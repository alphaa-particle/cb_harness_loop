# Pipeline evaluation: test questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/usecase_questions.jsonl` (sha256 `7bfbe6d4e6d4`). Corpus: `data/external/scale_corpus/corpus.jsonl`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | How to apply | Documents | Benefits | Who is excluded | Eligibility and how to apply | Compare two | Which schemes | English | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 67.5% (61–74) | 80.6% (69–90) | 85.7% (77–94) | 31.9% (19–44) | 95.8% (90–100) | 86.1% (75–97) | 7.1% (0–18) | 25.0% (0–50) | 67.5% (61–74) | – | 35.76 |
