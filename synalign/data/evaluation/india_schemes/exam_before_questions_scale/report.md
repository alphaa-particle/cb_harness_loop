# Pipeline evaluation: test questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/questions.jsonl` (sha256 `b77812e2d673`). Corpus: `data/external/scale_corpus/corpus.jsonl`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | Fact | Typo | Situation | Eligibility | English | Hindi | Hindi in Latin letters | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|
| char_tfidf | 62.6% (55–69) | 60.6% (51–69) | 71.8% (59–83) | 29.2% (17–42) | 79.0% (69–91) | 67.8% (58–76) | 52.1% (43–61) | 65.7% (56–75) | 0.0% (0–0) | 1.87 |
| bm25 | 66.4% (61–71) | 61.6% (54–69) | 80.9% (75–86) | 40.3% (31–50) | 81.5% (72–91) | 81.9% (77–86) | 48.9% (40–58) | 53.8% (43–64) | 4.5% (0–12) | 0.6 |
| dense | 85.1% (80–89) | 87.5% (82–93) | 92.4% (87–97) | 47.2% (33–60) | 91.9% (84–98) | 90.3% (86–94) | 79.3% (72–87) | 81.1% (74–87) | 22.7% (10–38) | 43.53 |
| fusion | 87.4% (83–91) | 88.0% (82–93) | 92.4% (86–97) | 56.9% (44–69) | 97.6% (92–100) | 92.5% (88–96) | 82.0% (75–89) | 82.5% (74–89) | 22.7% (10–38) | 48.19 |
| fusion_no_glossary | 87.5% (83–91) | 88.5% (83–94) | 92.4% (86–97) | 55.6% (43–67) | 97.6% (92–100) | 92.5% (88–96) | 82.0% (74–90) | 83.2% (77–89) | 22.7% (10–38) | 43.47 |

## Answers

Correct: see the grading rules at the top of scripts/evaluate_pipeline.py.

| Condition | All | Fact | Typo | Situation | Eligibility | Not found (scheme absent) | Not found (off topic) | Unsafe yes | Hindi answer to Hindi question | s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| old_search | 54.3% (48–60) | 53.7% (46–60) | 73.3% (62–83) | 19.4% (11–29) | 68.5% (60–77) | 3.3% (0–10) | 57.1% (29–86) | 10.5% (6–15) | 99.5% (98–100) | 1.22 |
| meaning | 68.7% (64–73) | 74.8% (70–80) | 87.8% (80–95) | 23.6% (14–35) | 70.2% (64–76) | 3.3% (0–10) | 78.6% (57–100) | 11.3% (6–18) | 100.0% (100–100) | 1.363 |
| fusion | 70.8% (66–75) | 75.6% (70–81) | 88.5% (81–95) | 33.3% (22–44) | 74.2% (68–80) | 3.3% (0–10) | 78.6% (57–100) | 10.5% (5–16) | 100.0% (100–100) | 1.239 |

## Paired differences (same questions, percentage points)

| Comparison | Questions | Difference | 95% interval |
|---|---|---|---|
| fusion minus old_search | 764 | +16.5 | +13.0 to +20.3 |
| fusion minus meaning | 764 | +2.1 | +0.0 to +4.2 |
