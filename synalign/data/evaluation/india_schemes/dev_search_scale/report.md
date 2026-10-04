# Pipeline evaluation: dev questions

Questions: `/Users/vaibhav_joshi/Desktop/cb_harness_loop/synalign/domains/india_schemes/evaluation/questions.jsonl` (sha256 `b77812e2d673`). Corpus: `data/external/scale_corpus/corpus.jsonl`. Sections in the prompt: 3. Meaning-similarity floor: 0.5.

Each cell: rate, then the 95% interval from resampling whole schemes.

## Search alone (no model)

Right text in the prompt: the section holding the answer is among those given to the model.

| Method | All | Fact | Typo | Situation | Eligibility | English | Hindi | Hindi in Latin letters | Not-found questions refused | ms per question |
|---|---|---|---|---|---|---|---|---|---|---|
| char_tfidf | 65.1% (54–74) | 66.0% (53–79) | 70.4% (56–82) | 16.7% (6–33) | 90.4% (77–100) | 69.7% (57–79) | 53.3% (34–70) | 71.7% (56–85) | 8.3% (0–27) | 6.77 |
| bm25 | 70.1% (62–77) | 71.6% (62–81) | 90.7% (85–96) | 25.0% (11–42) | 75.0% (54–94) | 82.9% (76–89) | 51.1% (33–69) | 66.7% (50–79) | 8.3% (0–27) | 2.76 |
| dense | 83.2% (73–90) | 86.4% (74–96) | 96.3% (91–100) | 38.9% (19–58) | 90.4% (77–100) | 90.8% (87–94) | 77.2% (58–91) | 73.3% (56–85) | 41.7% (13–80) | 88.44 |
| fusion | 85.9% (78–91) | 90.7% (82–98) | 94.4% (88–100) | 44.4% (28–61) | 90.4% (77–100) | 89.5% (85–93) | 81.5% (62–95) | 83.3% (73–92) | 41.7% (13–80) | 69.48 |
| fusion_no_glossary | 85.2% (77–91) | 90.7% (83–98) | 94.4% (88–100) | 38.9% (22–56) | 90.4% (77–100) | 89.5% (85–93) | 79.3% (59–94) | 83.3% (74–91) | 41.7% (13–80) | 68.59 |
