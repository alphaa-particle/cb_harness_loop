# Pipeline evaluation: dev questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/questions.jsonl` (sha256 `b77812e2d673`). Corpus: `data/external/scale_corpus/corpus.jsonl`. Sections in the prompt: 5. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: every section the answer needs is among those given to the model.

| Method | All | Fact | Typo | Situation | Eligibility | English | Hindi | Hindi in Latin letters | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|
| char_tfidf | 76.3% (68–82) | 79.6% (71–87) | 90.7% (81–100) | 19.4% (6–36) | 90.4% (77–100) | 82.9% (73–89) | 57.6% (37–76) | 88.3% (79–96) | 8.3% (0–27) | 2.17 |
| bm25 | 79.0% (72–85) | 83.3% (75–91) | 96.3% (91–100) | 30.6% (17–47) | 80.8% (57–100) | 88.2% (82–94) | 62.0% (41–79) | 81.7% (73–89) | 8.3% (0–27) | 0.63 |
| dense | 88.5% (82–93) | 95.1% (89–100) | 98.2% (94–100) | 41.7% (22–61) | 90.4% (77–100) | 91.5% (88–94) | 85.9% (74–95) | 85.0% (74–92) | 41.7% (13–80) | 50.31 |
| fusion | 90.8% (85–94) | 95.1% (88–100) | 98.2% (94–100) | 52.8% (36–69) | 96.2% (88–100) | 92.8% (90–96) | 89.1% (75–98) | 88.3% (81–94) | 41.7% (13–80) | 51.9 |
| fusion_no_glossary | 90.1% (85–94) | 94.4% (88–100) | 98.2% (94–100) | 50.0% (31–67) | 96.2% (88–100) | 92.8% (90–96) | 87.0% (73–95) | 88.3% (81–94) | 41.7% (13–80) | 51.27 |

## Answers

Correct: see the grading rules at the top of scripts/evaluate_pipeline.py.

| Condition | All | Fact | Typo | Situation | Eligibility | Not found (scheme absent) | Not found (off topic) | Unsafe yes | Hindi answer to Hindi question | s per answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fusion | 73.7% (66–79) | 80.9% (73–87) | 92.6% (87–98) | 19.4% (6–36) | 76.9% (62–91) | 0.0% (0–0) | 83.3% (50–100) | 7.7% (2–16) | 95.6% (92–99) | 1.748 |

## Paired differences (same questions, percentage points)

| Comparison | Questions | Difference | 95% interval |
|---|---|---|---|
