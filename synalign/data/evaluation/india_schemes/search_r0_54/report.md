# Pipeline evaluation: dev questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/questions.jsonl` (sha256 `b77812e2d673`). Corpus: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/documents`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: the section holding the answer is among those given to the model.

| Method | All | Fact | Typo | Situation | Eligibility | English | Hindi | Hindi in Latin letters | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 92.8% (89–96) | 92.0% (86–97) | 96.3% (91–100) | 80.6% (67–94) | 100.0% (100–100) | 97.4% (94–100) | 88.0% (79–95) | 88.3% (81–95) | 75.0% (60–100) | 51.94 |
