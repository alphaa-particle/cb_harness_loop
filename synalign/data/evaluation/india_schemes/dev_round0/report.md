# Pipeline evaluation: dev questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/questions.jsonl` (sha256 `b77812e2d673`). Corpus: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/documents`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | Fact | Typo | Situation | Eligibility | English | Hindi | Hindi in Latin letters | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|
| char_tfidf | 75.0% (61–86) | 71.6% (55–86) | 83.3% (67–94) | 55.6% (36–75) | 90.4% (77–100) | 84.9% (70–93) | 63.0% (42–80) | 68.3% (51–83) | 8.3% (0–27) | 0.36 |
| bm25 | 74.3% (66–82) | 78.4% (68–90) | 94.4% (89–100) | 47.2% (31–64) | 59.6% (37–83) | 81.6% (75–88) | 55.4% (35–73) | 85.0% (75–95) | 16.7% (0–42) | 0.09 |
| dense | 91.5% (85–96) | 90.7% (83–97) | 96.3% (91–100) | 83.3% (69–94) | 94.2% (83–100) | 97.4% (94–100) | 88.0% (72–98) | 81.7% (70–91) | 75.0% (60–100) | 3067.98 |
| fusion | 92.8% (89–96) | 92.0% (86–97) | 96.3% (91–100) | 80.6% (67–94) | 100.0% (100–100) | 97.4% (94–100) | 88.0% (79–95) | 88.3% (81–95) | 75.0% (60–100) | 1181.05 |
| fusion_no_glossary | 90.1% (85–95) | 88.9% (80–96) | 96.3% (91–100) | 80.6% (67–92) | 94.2% (83–100) | 97.4% (94–100) | 79.3% (62–92) | 88.3% (80–96) | 75.0% (60–100) | 45.53 |

## Answers

Correct: see the grading rules at the top of scripts/evaluate_pipeline.py.

| Condition | All | Fact | Typo | Situation | Eligibility | Not found (scheme absent) | Not found (off topic) | Unsafe yes | Hindi answer to Hindi question | s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| closed_book | 13.6% (10–17) | 8.6% (4–14) | 11.1% (4–21) | 0.0% (0–0) | 42.3% (38–48) | 16.7% (0–50) | 0.0% (0–0) | 57.7% (52–62) | 100.0% (100–100) | 1.152 |
| rules_only | 78.2% (73–83) | 95.7% (91–99) | 100.0% (100–100) | 80.6% (67–94) | 0.0% (0–0) | 50.0% (50–50) | 100.0% (100–100) | 0.0% (0–0) | 0.0% (0–0) | 0.034 |
| old_search | 64.2% (53–72) | 63.6% (48–76) | 85.2% (68–96) | 27.8% (17–39) | 69.2% (58–81) | 50.0% (50–50) | 83.3% (50–100) | 9.6% (4–15) | 100.0% (100–100) | 1.162 |
| meaning | 80.4% (76–84) | 82.7% (76–88) | 96.3% (91–100) | 50.0% (36–67) | 73.1% (56–88) | 100.0% (100–100) | 100.0% (100–100) | 7.7% (2–14) | 97.8% (95–100) | 1.142 |
| fusion | 82.3% (78–86) | 84.0% (79–89) | 96.3% (91–100) | 55.6% (39–75) | 76.9% (63–90) | 100.0% (100–100) | 100.0% (100–100) | 7.7% (2–14) | 97.8% (95–100) | 1.276 |
| oracle | 84.2% (79–89) | 90.1% (86–95) | 98.2% (94–100) | 63.9% (47–81) | 65.4% (50–81) | – | – | 3.9% (0–10) | 100.0% (100–100) | 1.125 |

## Paired differences (same questions, percentage points)

| Comparison | Questions | Difference | 95% interval |
|---|---|---|---|
| fusion minus closed_book | 316 | +68.7 | +63.4 to +73.6 |
| fusion minus old_search | 316 | +18.0 | +9.8 to +28.5 |
| oracle minus fusion | 304 | +2.6 | -0.4 to +5.4 |
| old_search minus closed_book | 316 | +50.6 | +40.9 to +59.0 |
| fusion minus meaning | 316 | +1.9 | -1.7 to +5.3 |
