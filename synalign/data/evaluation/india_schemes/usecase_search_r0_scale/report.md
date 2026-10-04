# Pipeline evaluation: dev questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/usecase_questions.jsonl` (sha256 `7bfbe6d4e6d4`). Corpus: `data/external/scale_corpus/corpus.jsonl`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | How to apply | Documents | Benefits | Who is excluded | Eligibility and how to apply | Compare two | Which schemes | English | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| char_tfidf | 48.9% (43–54) | 55.6% (42–69) | 44.1% (26–62) | 8.3% (0–17) | 91.7% (83–100) | 66.7% (44–89) | 20.0% (0–40) | 25.0% (0–75) | 48.9% (43–54) | – | 2.14 |
| bm25 | 60.7% (54–66) | 69.4% (56–83) | 50.0% (35–65) | 22.2% (8–36) | 97.2% (92–100) | 94.4% (83–100) | 20.0% (0–40) | 50.0% (0–100) | 60.7% (54–66) | – | 0.72 |
| dense | 72.5% (63–81) | 86.1% (72–97) | 70.6% (59–82) | 63.9% (44–81) | 97.2% (92–100) | 72.2% (50–89) | 20.0% (0–40) | 12.5% (0–38) | 72.5% (63–81) | – | 32.19 |
| fusion | 77.0% (68–84) | 91.7% (78–100) | 88.2% (76–97) | 50.0% (33–64) | 100.0% (100–100) | 83.3% (61–100) | 30.0% (0–70) | 25.0% (0–75) | 77.0% (68–84) | – | 37.56 |
| fusion_no_glossary | 76.4% (68–84) | 91.7% (78–100) | 88.2% (76–97) | 47.2% (31–64) | 100.0% (100–100) | 83.3% (61–100) | 30.0% (0–70) | 25.0% (0–75) | 76.4% (68–84) | – | 35.52 |
