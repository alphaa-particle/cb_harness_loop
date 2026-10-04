# Pipeline evaluation: test questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/questions.jsonl` (sha256 `b77812e2d673`). Corpus: `data/external/scale_corpus/corpus.jsonl`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | Fact | Typo | Situation | Eligibility | English | Hindi | Hindi in Latin letters | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 87.4% (83–91) | 88.0% (82–93) | 92.4% (86–97) | 56.9% (44–69) | 97.6% (92–100) | 92.5% (88–96) | 82.0% (75–89) | 82.5% (74–89) | 22.7% (10–38) | 37.73 |
