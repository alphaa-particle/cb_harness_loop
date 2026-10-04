# Pipeline evaluation: dev questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/usecase_questions.jsonl` (sha256 `7bfbe6d4e6d4`). Corpus: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/documents`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | How to apply | Documents | Benefits | Who is excluded | Eligibility and how to apply | Compare two | Which schemes | English | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| char_tfidf | 55.6% (49–62) | 58.3% (42–75) | 41.2% (24–59) | 19.4% (3–39) | 94.4% (86–100) | 72.2% (50–89) | 20.0% (0–60) | 100.0% (100–100) | 55.6% (49–62) | – | 0.33 |
| bm25 | 64.6% (57–72) | 69.4% (53–83) | 47.1% (32–59) | 38.9% (22–56) | 97.2% (92–100) | 77.8% (56–94) | 40.0% (10–70) | 87.5% (62–100) | 64.6% (57–72) | – | 0.09 |
| dense | 82.0% (73–90) | 88.9% (75–100) | 79.4% (68–91) | 69.4% (50–86) | 100.0% (100–100) | 83.3% (67–100) | 30.0% (0–70) | 100.0% (100–100) | 82.0% (73–90) | – | 34.78 |
| fusion | 82.0% (74–89) | 91.7% (78–100) | 79.4% (68–91) | 63.9% (44–81) | 100.0% (100–100) | 88.9% (72–100) | 30.0% (0–70) | 100.0% (100–100) | 82.0% (74–89) | – | 32.55 |
| fusion_no_glossary | 82.0% (74–89) | 91.7% (78–100) | 79.4% (68–91) | 63.9% (44–81) | 100.0% (100–100) | 88.9% (72–100) | 30.0% (0–70) | 100.0% (100–100) | 82.0% (74–89) | – | 29.92 |
