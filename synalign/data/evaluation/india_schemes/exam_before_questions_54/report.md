# Pipeline evaluation: test questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/questions.jsonl` (sha256 `b77812e2d673`). Corpus: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/documents`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | Fact | Typo | Situation | Eligibility | English | Hindi | Hindi in Latin letters | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|
| char_tfidf | 72.5% (66–78) | 68.5% (59–77) | 84.0% (75–92) | 63.9% (53–74) | 78.2% (65–91) | 84.4% (77–90) | 53.9% (44–63) | 70.6% (60–80) | 0.0% (0–0) | 0.42 |
| bm25 | 70.0% (64–75) | 68.7% (61–76) | 86.3% (79–92) | 61.1% (49–72) | 62.1% (50–77) | 79.4% (73–85) | 52.5% (43–62) | 72.7% (64–81) | 9.1% (2–19) | 0.09 |
| dense | 92.8% (89–96) | 91.9% (87–96) | 95.4% (91–99) | 86.1% (78–94) | 96.8% (90–100) | 96.7% (94–99) | 90.8% (84–96) | 86.0% (80–92) | 54.5% (43–68) | 40.93 |
| fusion | 90.6% (86–94) | 87.5% (81–93) | 96.2% (92–100) | 84.7% (78–92) | 97.6% (92–100) | 97.2% (94–100) | 83.4% (76–90) | 84.6% (77–92) | 54.5% (43–68) | 52.64 |
| fusion_no_glossary | 90.3% (87–93) | 87.5% (82–92) | 96.2% (92–100) | 81.9% (75–90) | 97.6% (92–100) | 97.2% (94–100) | 81.6% (74–88) | 86.0% (80–92) | 54.5% (43–68) | 44.92 |

## Answers

Correct: see the grading rules at the top of scripts/evaluate_pipeline.py.

| Condition | All | Fact | Typo | Situation | Eligibility | Not found (scheme absent) | Not found (off topic) | Unsafe yes | Hindi answer to Hindi question | s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| closed_book | 13.9% (11–16) | 8.6% (6–12) | 12.2% (8–17) | 0.0% (0–0) | 45.2% (41–49) | 0.0% (0–0) | 0.0% (0–0) | 54.0% (51–58) | 100.0% (100–100) | 1.194 |
| rules_only | 74.1% (69–79) | 89.6% (83–95) | 97.0% (93–100) | 87.5% (81–94) | 0.0% (0–0) | 36.7% (27–47) | 92.9% (79–100) | 0.0% (0–0) | 0.0% (0–0) | 0.031 |
| old_search | 61.7% (56–67) | 58.8% (51–66) | 80.2% (70–88) | 38.9% (26–51) | 68.5% (62–75) | 40.0% (30–50) | 71.4% (43–93) | 11.3% (7–16) | 99.5% (98–100) | 1.721 |
| meaning | 77.5% (74–81) | 77.3% (73–82) | 90.8% (84–96) | 55.6% (44–67) | 71.8% (66–77) | 86.7% (77–97) | 100.0% (100–100) | 10.5% (5–16) | 99.5% (98–100) | 2.385 |
| fusion | 76.4% (72–80) | 75.8% (70–81) | 90.8% (84–96) | 51.4% (39–64) | 75.0% (69–81) | 80.0% (67–90) | 92.9% (79–100) | 10.5% (5–16) | 100.0% (100–100) | 1.332 |
| oracle | 82.4% (79–85) | 86.8% (83–90) | 95.4% (92–98) | 56.9% (44–69) | 69.3% (63–75) | – | – | 8.9% (4–16) | 99.1% (98–100) | 1.092 |

## Paired differences (same questions, percentage points)

| Comparison | Questions | Difference | 95% interval |
|---|---|---|---|
| fusion minus closed_book | 764 | +62.6 | +58.1 to +67.1 |
| fusion minus old_search | 764 | +14.8 | +11.1 to +18.7 |
| oracle minus fusion | 720 | +6.4 | +3.7 to +9.3 |
| old_search minus closed_book | 764 | +47.8 | +41.5 to +53.3 |
| fusion minus meaning | 764 | -1.1 | -2.9 to +0.7 |
