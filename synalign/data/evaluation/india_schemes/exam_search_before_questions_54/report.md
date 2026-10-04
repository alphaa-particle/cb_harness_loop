# Pipeline evaluation: test questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/questions.jsonl` (sha256 `b77812e2d673`). Corpus: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/documents`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | Fact | Typo | Situation | Eligibility | English | Hindi | Hindi in Latin letters | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 90.6% (86–94) | 87.5% (81–93) | 96.2% (92–100) | 84.7% (78–92) | 97.6% (92–100) | 97.2% (94–100) | 83.4% (76–90) | 84.6% (77–92) | 54.5% (43–68) | 32.52 |
