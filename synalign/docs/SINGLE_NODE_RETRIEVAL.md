# Single-node retrieval and corpus replacement

One local corpus is loaded into one in-memory index when the process starts. To
use other scheme documents, point to another corpus and restart. There is no
database and no cloud service. Word-based search needs only `requirements.txt`;
meaning search also needs the local embedding server (`scripts/serve_models.py`).

## Search methods

| Method | How it ranks | Needs the embedder |
|---|---|---|
| `char_tfidf` | runs of 3–5 letters (survives typos); the original default | no |
| `word_tfidf`, `hybrid` | words and word pairs; `hybrid` averages the two TF-IDF scores | no |
| `bm25` | whole words, Lucene BM25 (k1 1.2, b 0.75); Hindi words are kept whole | no |
| `dense` | meaning: Qwen3-Embedding-0.6B vectors, cosine similarity | yes |
| `fusion` | `bm25`, `char_tfidf` and `dense` merged by weighted reciprocal rank | yes |

For `fusion`, each method ranks every section; a section's merged score is the
sum over methods of `weight / (rrf_k + rank)`, counting the top `fusion_depth`
ranks of each. The india_schemes settings (`bm25` 0.25, `char` 0.25, `dense`
1.0, `rrf_k` 20, depth 50) were chosen on dev questions only. Glossary words
(`glossary.yaml`: Hindi and Hinglish words with their English twins) are added
to the question for the word-based methods only; the meaning model sees the
question as asked.

`min_score` for `dense` and `fusion` is a floor on meaning similarity: when no
section reaches it, nothing is retrieved and the model is not asked. For the
word-based methods it is a floor on TF-IDF cosine; BM25 scores are unbounded,
so it does not apply to `bm25` alone.

### Which sections go into the prompt

After ranking, `retrieve_evidence()` fills the prompt's `top_k` slots, each found scheme's
rules first. Settings refine this; every one went through a SynAlign round, recorded in
[SYNALIGN_RETRIEVAL_ROUNDS.md](results/SYNALIGN_RETRIEVAL_ROUNDS.md):

| Setting | Effect | Status |
|---|---|---|
| `add_named_schemes` | a scheme the question names but the prompt lacks is brought in, in place of the least useful slot; every other prompt is unchanged | on for india_schemes |
| `name_rules` | 1: the first name rules; 2: corrections from a code review (bracketed descriptions are not names, short forms used elsewhere still count, joining words do not block a name, the slot given up is never a named scheme's rules) | 1 for india_schemes: 2 passed the practice checks but broke exam answers at 4,600 schemes |
| `rules_count_once` | a scheme's rules take one slot however many parts they were cut into | tried, undone |
| `named_schemes` | schemes the question names get all the slots | tried, undone |
| `overview_last` | for a named scheme, its overview only fills slots left free | tried, not adopted |

Name matching uses the corpus's own titles and aliases, so it needs nothing per
scheme. A name shared by several schemes (such as a generic "Disability Pension")
identifies none of them, and the scheme search ranks first always keeps its place.

Meaning vectors for sections are cached in `models/index/<model>.npz`, keyed by
the model and the exact text. A changed section is embedded again; unchanged
ones are reused, and the cache only grows, so several corpora can share it.
Questions get the instruction prefix the embedding model's card asks for;
sections get none.

## Insert a real corpus

```bash
export SYNALIGN_CORPUS_PATH=/absolute/path/corpus.jsonl
python scripts/search_corpus.py --query "What are the eligibility rules for the scheme?" --context
python scripts/run_api.py
```

Without other settings the india_schemes search settings apply (`fusion`, 3
sections, floor 0.50). They were chosen for that corpus: re-check them on your
own labelled dev questions (`scripts/evaluate_pipeline.py` with `--corpus`, or
the older `evaluate_context.py` below) before relying on them.
`SYNALIGN_RETRIEVAL_METHOD`, `SYNALIGN_RETRIEVAL_TOP_K` and
`SYNALIGN_RETRIEVAL_MIN_SCORE` override them.

The API listens on `127.0.0.1:8000` with one worker. `GET /` reports the corpus
hash, section count and search settings. `POST /search` returns the most similar
sections with their IDs, original text, scores and the corpus fingerprint.
`POST /context` returns the evidence that would be sent to the model, each
scheme's rule section first, and the full prompt; no answer model is used.
`POST /ask` answers through the configured assistant with that same evidence.
If the answer model is unavailable, `/search` and `/context` still work.

A model-backed assistant is never asked without evidence: when nothing reaches
the floor, `/ask` returns a fixed "not found" reply.

An index is a snapshot. Editing a file does not update a running process; restart
it and check that `corpus_sha256` changed.

## Corpus contract

Preferred format: UTF-8 JSONL, one complete source section per line:

```json
{"chunk_id":"grant_x_rules_v1","scheme_id":"grant_x","section":"rules","title":"Example Textbook Grant — Eligibility","aliases":["ETG","local name"],"text":"Full source rules, including exclusions and exceptions, go here.","source":"official source URL or reference"}
{"chunk_id":"grant_x_apply_v1","scheme_id":"grant_x","section":"application","title":"Example Textbook Grant — Application","text":"Full source application instructions go here."}
```

Only `chunk_id` and nonempty `text` are required. Optional `title`, `scheme_id`,
`section` and `source` must be strings; `aliases` must be a list of strings.

Sections that share a `scheme_id` are one scheme. `"section": "rules"` marks the
section holding that scheme's eligibility rules, and is what rules-first selection
acts on: whenever any section of a scheme is retrieved, its rule section is placed
in the prompt first, within the same evidence limit. A scheme may have more than
one rule section. Any other `section` value is only a label. A corpus with no
`rules` label behaves exactly as plain search. BEIR `_id`
works as an alternative to `chunk_id`. Conflicting IDs, duplicate IDs, empty
source text and malformed records stop loading with an error. Keep IDs stable
across indexing, labels and training; change versioned IDs when their meaning
changes. Titles and aliases enrich search while returned/gold text stays exactly
the source `text`. `scheme_id` groups sections; it is not an eligibility engine
or a jurisdiction filter. Put jurisdiction, scheme identity and qualifications
in the evidence text; put the source reference in the text too if the answer
must cite it. Each returned section carries its `title`, `scheme_id` and `section`.
In the prompt a section is headed by its title unless the text already contains it.

Existing Markdown also works: pass a `.md` file or a directory recursively
containing `.md` files. Separate sections with `---`; give each an explicit,
unique `Chunk ID:` or `Scheme ID:` line. Positional fallback IDs remain supported,
but explicit IDs are safer when documents are edited. To split one scheme over
several Markdown sections, give each extra section both a `Scheme ID:` line (the
scheme it belongs to) and its own `Chunk ID:` line, and add `Section: rules` to
the rule section. These lines are part of the section text and are indexed with it. A directory containing
`corpus.jsonl` uses that file instead of Markdown, so accompanying benchmark
queries are never indexed as evidence. Arbitrary JSONL filenames in a directory
are not merged; pass the corpus file explicitly.

PDF/HTML extraction is outside this loader: supply their extracted, reviewed
text in one of these formats. Keep a rule and its exceptions together. Include
common names and the languages your users actually use. Adding a translated
name does not provide cross-language understanding of the full rules.

## Choose evidence count and validate relevance

Capacity depends on section count and length. Five thousand schemes with four
sections each means 20,000 indexed sections. In the synthetic test, plain top-3
search found the scheme but missed its rules, because the scheme's other sections
repeat its name and outrank the rule section. Labelling the rule section fixes
this at top-3 without a longer prompt. More evidence also increases model context
length, so select top-k on development queries and check completeness, not just
whether any scheme name appeared.

To check the prompt itself rather than the ranking, use `evaluate_context.py`. It
builds each question's prompt as the assistant does and reports whether the
labelled source text is in it, and whether unanswerable questions would skip the
model, next to the same figures for plain search:

```bash
python scripts/evaluate_context.py --corpus "$SYNALIGN_CORPUS_PATH" \
  --questions reviewed_queries.jsonl --split test --method hybrid --top-k 3 \
  --min-score 0.30 --output /tmp/real_test_context.json
```

Supply reviewed questions with independently chosen correct source IDs:

```json
{"query_id":"dev_01","group_id":"dev_intent_01","split":"dev","question":"Who qualifies for the textbook grant?","gold_chunk_ids":["grant_x_rules_v1"]}
{"query_id":"test_01","group_id":"test_intent_01","split":"test","question":"Am I eligible, and how do I apply?","gold_chunk_ids":["grant_x_rules_v1","grant_x_apply_v1"],"required_chunk_ids":["grant_x_rules_v1","grant_x_apply_v1"]}
{"query_id":"test_02","group_id":"test_intent_02","split":"test","question":"Does this corpus describe an overseas travel subsidy?","gold_chunk_ids":[],"expect_abstain":true}
```

Use actual representative user questions and expert labels, including spelling
errors, local languages, ambiguous schemes, exclusions and missing evidence.
Keep variants of one user/intent in a single `group_id` and one split; leakage
across splits and unknown gold IDs are rejected. Existing audit records with
`ground_truth.gold_chunk_ids` also work. Optional `relevance` maps IDs to graded
relevance, and `condition` enables slice reports.

```bash
python scripts/evaluate_retrieval.py --corpus "$SYNALIGN_CORPUS_PATH" \
  --questions reviewed_queries.jsonl --split dev --method hybrid \
  --cutoffs 1 3 5 10 --output /tmp/real_dev_retrieval.json
python scripts/evaluate_retrieval.py --corpus "$SYNALIGN_CORPUS_PATH" \
  --questions reviewed_queries.jsonl --split test --method hybrid \
  --cutoffs 1 3 5 10 --output /tmp/real_test_retrieval.json
```

Reports include hit rate, recall, precision, MRR, nDCG, completeness of required
sections, per-condition results, query-cluster bootstrap intervals, timing,
fingerprints and per-query evidence. Unanswerable cases are scored separately
for abstention; they do not inflate recall. Freeze configuration on dev before
looking at test results.

`SYNALIGN_RETRIEVAL_MIN_SCORE` filters weak matches. It is a similarity threshold,
not a probability that an answer is correct. Calibration uses only dev cases and
reports two choices. `selected` maximizes balanced relevant-hit/abstention
accuracy, which will give up answerable questions to refuse unanswerable ones.
`selected_keeping_answerable` loses no answerable dev question and then refuses
as many unanswerable ones as it can; prefer it unless refusing matters more than
answering. Both follow the rules-first evidence at the `--top-k` you pass:

```bash
python scripts/calibrate_retrieval.py --corpus "$SYNALIGN_CORPUS_PATH" \
  --questions reviewed_queries.jsonl --method hybrid --output /tmp/calibration.json
```

Dev must contain both answerable and unanswerable cases. The script records its
selection and all candidates; it does not modify runtime settings. Apply the
selected threshold explicitly via `--min-score` during evaluation and
`SYNALIGN_RETRIEVAL_MIN_SCORE` at runtime. Then verify all-required-evidence
recall at your actual top-k. Do not copy the fictional test's 0.16 threshold to
an untested corpus: it still failed three of six held-out unanswerable cases.

## Gold chunks and fine-tuning compatibility

These retrieval interfaces and training file structures remain compatible:

- `retrieve(question, top_k=None) -> list[RetrievedChunk]`
- `get_chunks_by_ids(chunk_ids) -> list[RetrievedChunk]`
- `RetrievedChunk(chunk_id, text, score)`
- SFT files `sft_train.jsonl`, `sft_dev.jsonl`, `sft_test_reference.jsonl`, each
  retaining `messages`, `case_id`, `split`, `condition`.

Gold context is looked up directly by labelled ID, not by a retrieval prediction.
The training-data builder reads that original evidence from the selected corpus and
writes the prompt with the same function the assistant uses, so the instructions
and layout match what the model is given at run time. Training/merge scripts remain
unchanged. The builder validates all records before opening outputs, so missing
evidence or incomplete eligibility labels cannot silently replace existing files.

```bash
python scripts/make_training_data.py --corpus "$SYNALIGN_CORPUS_PATH" \
  --input reviewed_audit.jsonl --output-dir /absolute/path/new_training
```

Each input row still requires `case_id`, `split`, `condition`, `question` and
`ground_truth` with `gold_chunk_ids`, `likely_eligible`,
`unknown_due_to_missing_info`, `ineligible`, and `must_ask_about`. Supply reviewed
`ineligibility_reasons` when specific reasons should appear in the target answer.
Entity decisions refer to scheme IDs; evidence references refer to chunk IDs.
The script uses available corpus titles for names. Review generated targets
before training, as in the existing process.

Inserting documents makes them searchable and usable as model context. It does
not manufacture verified eligibility decisions or domain-specific synthetic
profiles. The demo ground-truth generator still describes only its two original
schemes; use reviewed audit records for the real corpus or provide its matching
domain pack. The demo audit rejects absent gold chunks. Retaining an old chunk
ID with changed rules still requires re-reviewing its labels.

## Implementation and checks

The word-based indexes are sparse, normalised matrices; a query is one sparse
product followed by partial top-k selection with deterministic ties. Meaning
search is one dense product against the cached section vectors. On an Apple M5,
fusion over the 286 india_schemes sections takes about 50 ms per question and
over 27,593 sections about 70 ms, nearly all of it embedding the question.

```bash
python -m unittest discover -s tests
python scripts/evaluate_pipeline.py --split dev --search-only
```

The tests cover BM25 against a hand-worked example, meaning search with a
stand-in embedder, the merge, the floor, the vector cache, old-query
equivalence for the word methods, replacement corpora, source-text fidelity,
split leakage, no-answer scoring, the API, rules-first selection, prompt
contents, the answer check, and the full path from question to prompt for
5,000 generated schemes.
