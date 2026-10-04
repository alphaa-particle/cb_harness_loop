# Pipeline evaluation: dev questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/questions.jsonl` (sha256 `b77812e2d673`). Corpus: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/documents`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | Fact | Typo | Situation | Eligibility | English | Hindi | Hindi in Latin letters | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|
| char_tfidf | 75.0% (61–86) | 71.6% (55–86) | 83.3% (67–94) | 55.6% (36–75) | 90.4% (77–100) | 84.9% (70–93) | 63.0% (42–80) | 68.3% (51–83) | 8.3% (0–27) | 0.35 |
| bm25 | 74.3% (66–82) | 78.4% (68–90) | 94.4% (89–100) | 47.2% (31–64) | 59.6% (37–83) | 81.6% (75–88) | 55.4% (35–73) | 85.0% (75–95) | 16.7% (0–42) | 0.09 |
| dense | 91.5% (85–96) | 90.7% (83–97) | 96.3% (91–100) | 83.3% (69–94) | 94.2% (83–100) | 97.4% (94–100) | 88.0% (72–98) | 81.7% (70–91) | 75.0% (60–100) | 31.52 |
| fusion | 92.8% (89–96) | 92.0% (86–97) | 96.3% (91–100) | 80.6% (67–94) | 100.0% (100–100) | 97.4% (94–100) | 88.0% (79–95) | 88.3% (81–95) | 75.0% (60–100) | 32.05 |
| fusion_no_glossary | 90.1% (85–95) | 88.9% (80–96) | 96.3% (91–100) | 80.6% (67–92) | 94.2% (83–100) | 97.4% (94–100) | 79.3% (62–92) | 88.3% (80–96) | 75.0% (60–100) | 31.89 |

## Answers

Correct: see the grading rules at the top of scripts/evaluate_pipeline.py.

| Condition | All | Fact | Typo | Situation | Eligibility | Not found (scheme absent) | Not found (off topic) | Unsafe yes | Hindi answer to Hindi question | s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 73.1% (68–78) | 77.8% (68–85) | 83.3% (76–91) | 50.0% (31–69) | 57.7% (52–65) | 100.0% (100–100) | 100.0% (100–100) | 28.8% (26–31) | 98.8% (96–100) | 1.052 |

## Paired differences (same questions, percentage points)

| Comparison | Questions | Difference | 95% interval |
|---|---|---|---|
