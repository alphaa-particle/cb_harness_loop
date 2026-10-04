# Retrieval research and accuracy report

Measured 2026-10-02. The implementation supports replacing a local text corpus
without changing retrieval, API or SFT code. One-node capacity is demonstrated
for 5,000 synthetic schemes / 20,000 sections. Accuracy is conditional on the
corpus, query language and evidence count; these tests do not establish perfect
retrieval or verified answer accuracy for 5,000 real welfare schemes.

> **Later the same day:** rule sections can now be labelled and are placed first
> in the model's prompt, which removes the "top-3 misses the rules" failure
> described below without a longer prompt. The grader also gained one more
> accepted wording ("work type"), which changes the re-graded figures in the last
> table: the LoRA run is 70/100 on the test split, not 59/100. Current figures are
> in the [top-level README](../../README.md); raw results are
> `context_5000_schemes_test.json` and `saved_runs_regraded.json`. Everything
> below is the earlier measurement, left as written.

[Corpus replacement instructions](SINGLE_NODE_RETRIEVAL.md) ·
[Raw results](../data/evaluation/) ·
[Environment and dataset manifest](../data/evaluation/research_manifest.json)

## Research basis and experiment design

A useful IR evaluation needs documents, representative information needs and
independent relevance judgments. A test made only from scheme names can check
lookup and capacity, but cannot establish natural-language relevance. This
motivated separate compatibility, capacity, independent benchmark and difficult
welfare-style tests. See the [Stanford IR evaluation chapter](https://nlp.stanford.edu/IR-book/html/htmledition/information-retrieval-system-evaluation-1.html)
and its [discussion of relevance judgments](https://nlp.stanford.edu/IR-book/html/htmledition/assessing-relevance-1.html).

BEIR evaluates retrieval across heterogeneous datasets and identifies BM25 as a
strong baseline. Its findings argue for testing transfer across domains rather
than assuming success from a small demo. This change compares three inexpensive
lexical methods already implementable with the installed scikit-learn stack;
it does not establish superiority over BM25, dense retrieval or reranking, which
were not run. See the [BEIR paper](https://arxiv.org/abs/2104.08663) and
[official dataset repository](https://github.com/beir-cellar/beir).

The three fixed methods are `char_tfidf` (existing character 3–5-gram cosine),
`word_tfidf` (word 1–2-gram cosine with sublinear term frequency), and `hybrid`
(equal-weight average of those scores). No supervised training, query expansion,
reranker, label-derived document augmentation or test-set weight fitting was
used in the independent comparison. The existing demo default remains unchanged;
hybrid is configurable. L2-normalized TF-IDF permits cosine similarity through
a dot product, avoiding repeated document normalization. This is documented by
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

The evaluator reports hit@k (any relevant section), recall@k (fraction of labelled
relevant sections), precision@k, MRR@k, graded nDCG@k, and all-required-evidence
completion where specified. Unanswerable cases are separate. Confidence intervals
resample query groups 1,000 times with seed 42 so variants remain together.
Hand-calculated graded ranking examples and rejection of leaking splits are
covered by automated tests. These ranked metrics follow standard IR practice;
see [Stanford's ranked evaluation discussion](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html).

All timings are local ARM64, Python 3.12.13, sequential queries with the index
already built. They exclude HTTP transport, concurrent traffic, model inference
and full-process memory. They are measurements, not service-level guarantees.

## Independent relevance benchmark: SciFact

Source: BEIR's SciFact release, 5,183 real scientific abstracts and 300 labelled
test queries. Documents and relevance judgments come from the benchmark, not
from the retriever or the coding agent. The [SciFact authors' repository](https://github.com/allenai/scifact)
describes its scientific claim-verification setting. It is not welfare data.

The downloaded official archive matched published MD5
`5f7d1de60b170fc8027bb7898e2efca1`; its SHA-256 and runtime package versions are in
the manifest. The external corpus is not copied into this repository.

| Fixed method | Hit@3 | Recall@3 | Hit@10 | nDCG@10 | Retrieval p95 | Build |
|---|---:|---:|---:|---:|---:|---:|
| Character TF-IDF | 63.33% | 61.37% | 78.00% | 0.6066 | 0.635 ms | 2.832 s |
| Word TF-IDF | 65.00% | 63.00% | 78.00% | 0.6209 | 0.244 ms | 0.937 s |
| Equal-weight hybrid | 68.67% | 66.74% | 82.33% | 0.6582 | 0.827 ms | 3.568 s |

Hybrid found at least one labelled relevant abstract in the first three results
for 206/300 queries; 94 queries missed. Its hit@3 bootstrap interval is
63.66–73.67%, versus 58.00–68.34% for character TF-IDF. This is a descriptive
comparison, not a paired significance claim. The improvement on SciFact does
not prove that hybrid will improve the future welfare corpus.

Full rankings and per-query scores:
[character](../data/evaluation/scifact_char_tfidf.json),
[word](../data/evaluation/scifact_word_tfidf.json),
[hybrid](../data/evaluation/scifact_hybrid.json).

## Difficult fictional welfare queries at 5,000 sections

The checked-in fixtures contain 15 authored source sections. The generator adds
4,985 fictional distractor sections about similar benefits in other states and
shuffles them with a fixed seed. These are engineering fixtures authored by the
coding agent, not expert-labelled real schemes or a representative traffic
sample. Their purpose is to reveal failure modes while a real corpus is absent.

There are 18 development questions (15 answerable, three unanswerable) and 47
held-out test questions (41 answerable, six unanswerable). Query families have
disjoint group IDs across splits. Test slices include names/aliases, paraphrases,
typos, misleading similar options, multi-section evidence, Hindi/transliteration,
and absent evidence. Dev chooses only the optional score threshold; test labels
are not used to choose that threshold.

| Method/configuration | Hit@1, answerable | Hit@3, answerable | Complete evidence@3 | Correct abstentions | p95 |
|---|---:|---:|---:|---:|---:|
| Character, threshold 0 | 34/41 | 38/41 (92.68%) | 2/2 | 1/6 | 0.473 ms |
| Hybrid, threshold 0 | 35/41 | 38/41 (92.68%) | 2/2 | 1/6 | 0.627 ms |
| Hybrid, dev-selected threshold 0.16 | 35/41 | 38/41 (92.68%) | 2/2 | 3/6 | 0.641 ms |

The hit@3 cluster-bootstrap interval is wide: 81.07–100.00%. The 36 English
answerable cases found relevant evidence in the top three, while cross-language
cases succeeded on only 2/5. These small, authored samples cannot justify a
production claim of 92.68% accuracy. Two multi-evidence questions are also much
too few to establish rule completeness generally.

The threshold scored perfectly on the tiny dev set but only rejected three of
six unanswerable test cases. Specific failures after calibration:

- Hindi questions asking about Punjab college books, Bihar pensions and Delhi
  roof repairs retrieved no English source sections.
- A Kerala mortgage-interest question retrieved a Delhi roof-repair scheme.
- A Gujarat dental-cleaning question retrieved a Maharashtra dialysis scheme.
- A Canadian postgraduate-tuition question retrieved Punjab textbook evidence.

These expose cross-language and qualification/jurisdiction failures. Lexical
similarity is not an eligibility decision. A global threshold cannot reliably
separate related-but-inapplicable evidence. Real-corpus evaluation should include
explicit jurisdiction, version/date, exclusions and negative-answer cases;
consider structured filters or a semantic verifier if the measured failures
warrant them. Such additions are not claimed to have been tested here.

Raw results: [character](../data/evaluation/welfare_stress_char_tfidf.json),
[hybrid](../data/evaluation/welfare_stress_hybrid.json),
[dev calibration](../data/evaluation/welfare_dev_calibration.json),
[held-out calibrated hybrid](../data/evaluation/welfare_stress_hybrid_calibrated.json).

## Capacity and the difference between finding a scheme and finding its rules

The earlier one-section-per-scheme test had 5,000 synthetic schemes and 200 named
queries. It measured approximately 0.55 ms median / 0.59 ms p95, a 1.07 s build,
and 121.9 MiB for sparse matrices. That easy test found every requested section;
it did not test competition between sections of the same scheme.

A new test uses the same 5,000 schemes, each with rules plus application,
disbursement and review sections: 20,000 sections in total. Every named query
asks for eligibility rules. The 200 selected queries use seed 42.

| Evidence limit | Correct rule found | Median / p95 | Build | Sparse matrices |
|---|---:|---:|---:|---:|
| Top 3 | 0/200 | 1.373 / 1.455 ms | 3.995 s | 430.08 MiB |
| Top 5 | 200/200 | 1.382 / 1.471 ms | 3.860 s | 430.08 MiB |

The top-3 results belonged to the requested scheme but its name-heavy ancillary
sections outranked the rule section. Word and hybrid scoring exhibited the same
rank-four rule failure on this fixture. Merely changing the scoring method did
not fix it. Selecting five sections recovered the rules here. This was a
follow-up diagnostic on the observed failure, not a separately held-out accuracy
claim. A different corpus may need another evidence budget or better ranking.

The previous full cosine/sort query path took 58.47 ms median / 59.35 ms p95 at
20,000 sections and top-5, about 40 times slower at p95. It also succeeded only
when the evidence count included the rule. All 20,000 original chunk IDs/texts
were checked through direct gold lookup. Sparse storage excludes vocabulary,
source strings, Python/runtime overhead and any model weights.

This supports a one-node MVP for the measured corpus sizes. It does not identify
a universal maximum number of schemes or a concurrent-request capacity.
[Top-3 artifact](../data/evaluation/capacity_5000_schemes_20000_chunks.json) ·
[Top-5 artifact](../data/evaluation/capacity_5000_schemes_20000_chunks_k5.json) ·
[Word diagnostic](../data/evaluation/capacity_rules_20000_word_tfidf.json) ·
[Hybrid diagnostic](../data/evaluation/capacity_rules_20000_hybrid.json).

## Existing system and fine-tuning pipeline

Twenty-two automated tests pass. They cover 500 saved query rankings/scores
against the original algorithm; 150 generated baseline cases with unchanged
training targets; source/gold identity; malformed corpora; metric correctness;
split leakage; threshold calibration; API corpus replacement; Qwen/Ollama prompt
routing; configurable top-k; and new-corpus SFT generation. Routing tests stub
model generation and do not measure model answer correctness. A Unicode routing
bug was also fixed: the English-vowel gibberish test no longer rejects every
non-Latin query. This does not make the lexical retriever cross-lingual.

The API was repaired so it builds one index in its lifespan and lazily loads the
assistant. `/search` works without model dependencies. Both the API and SFT
builder consume a replacement corpus. Rebuilding the existing 500-record SFT
input into a temporary directory retained 300 train / 100 dev / 100 test
examples. Existing stored training files and model training scripts were not
rewritten. The builder now obtains gold context from labelled source IDs;
missing or incomplete labels fail before opening outputs.

A fresh non-LLM baseline audit ran 1,500 cases from 300 synthetic demo users,
with user-level splits. It retrieved all expected demo evidence, but its held-out
behavioral pass rate was **169/300 (56.33%)**. On test queries with missing
information it passed only 3/60; vague questions passed 0/60. The two-document
demo is too small to establish scalable retrieval accuracy.

The behavioral scorer was corrected to recognize explicit requests such as
"Please share ...", section-labelled requests, and bare field IDs such as
`worker_type`. Replaying saved answers with the corrected scorer produced:

| Stored run | Original test pass | Rescored test pass | New model inference? |
|---|---:|---:|---|
| Qwen rule-trigger-fix, 500 total cases | 58/100 | 96/100 | No |
| Qwen LoRA SFT, 500 total cases | 58/100 | 59/100 | No |

These are scorer corrections on existing answers, not improvements from training
or from the new retriever. The lexical groundedness check measures word overlap;
it does not prove entailment, numeric correctness or correct eligibility. Scores
must not be described as verified factual accuracy. Saved run names alone also
do not establish how much their original answers depended on model generation
versus rule routing. Source hashes and each changed evaluation are preserved in
[demo_system.json](../data/evaluation/demo_system.json).

No local model weights, `torch` or `transformers` were available, so no fresh
Qwen/LoRA answer-quality evaluation was run. Existing router thresholds also
remain distinct from retrieval filtering. End-to-end model accuracy on the real
corpus needs a fresh run with that backend and independent answer review.

## Reproduce and replace the corpus

From `synalign/`, install the existing requirements in a Python 3.12 environment.
Use the package versions in the manifest if exact software reproduction matters.
Unset corpus/domain/retrieval overrides for the default demo checks.

```bash
python -m unittest discover -s tests -v
python scripts/build_retrieval_stress.py --output /tmp/welfare_stress --chunks 5000
python scripts/evaluate_retrieval.py --corpus /tmp/welfare_stress/corpus.jsonl \
  --questions /tmp/welfare_stress/questions.jsonl --split test --method hybrid \
  --output /tmp/stress_hybrid.json
python scripts/calibrate_retrieval.py --corpus /tmp/welfare_stress/corpus.jsonl \
  --questions /tmp/welfare_stress/questions.jsonl --method hybrid --output /tmp/calibration.json
python scripts/evaluate_retrieval.py --corpus /tmp/welfare_stress/corpus.jsonl \
  --questions /tmp/welfare_stress/questions.jsonl --split test --method hybrid \
  --min-score 0.16 --output /tmp/stress_calibrated.json
python scripts/run_audit.py
python scripts/regrade_audits.py data/outputs/audit_qwen_rule_trigger_fix_500.jsonl \
  data/outputs/audit_qwen_lora_sft_500.jsonl --output /tmp/regraded.json
python scripts/make_training_data.py --input data/outputs/audit_qwen_router_rule_fix_500.jsonl \
  --output-dir /tmp/demo_sft_validation
```

For SciFact, download `scifact.zip` through the dataset link in the manifest or
BEIR repository, verify its checksum, and extract it into `/tmp/scifact_data/`.
The selected directory must contain `corpus.jsonl`, `queries.jsonl`, and `qrels/`.
Run once per method (`char_tfidf`, `word_tfidf`, `hybrid`):

```bash
python scripts/evaluate_retrieval.py --beir /tmp/scifact_data/scifact \
  --split test --method hybrid --output /tmp/scifact_hybrid.json
```

The capacity figures and the word/hybrid multi-section diagnostics were produced by
`scripts/benchmark_retrieval.py`, which the later cleanup replaced: the same
measurement at the same size (5,000 schemes, 20,000 sections) is now
`scripts/build_scheme_corpus.py` followed by `scripts/evaluate_context.py`, which
reports build time, latency and plain-versus-rules-first results. Its corpus is
generated differently, so its numbers are not the ones in the tables above. The
original diagnostics used the same generated source as the capacity
benchmark, not another corpus.

For the real corpus, export stable-ID JSONL/Markdown, point
`SYNALIGN_CORPUS_PATH` at it, and restart. Use independent dev/test query labels
to select method, evidence count and any threshold. Adding reviewed audit labels
lets the existing gold-context/SFT workflow operate on that corpus. Documents
alone cannot replace expert eligibility labels or the demo's domain-specific
ground-truth generator. No retraining is required merely to index new schemes.
