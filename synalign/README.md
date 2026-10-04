# SYNALIGN: how to run it

What the chatbot does and what was measured is in the [top-level README](../README.md). This page is the commands. Everything runs from this `synalign/` folder.

---

## 1. Set up

Python 3.10 or newer, and [Homebrew](https://brew.sh) for llama.cpp.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install llama.cpp
```

The two model files go in `models/` (about 1.5 GB together; not in git):

```bash
mkdir -p models
curl -L -o models/Qwen3-Embedding-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf
curl -L -o models/Qwen3.5-0.8B-Q8_0.gguf https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf
```

The files used for every result here have these SHA-256 sums:

| File | SHA-256 |
|---|---|
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439` |
| `Qwen3.5-0.8B-Q8_0.gguf` | `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f` |

Check that everything works (no model needed; stand-ins replace both models):

```bash
python -m unittest discover -s tests
```

| Test file | What it covers |
|---|---|
| `test_scheme_records.py` | the scheme-file checker, cutting schemes into sections, fact questions |
| `test_search_methods.py` | BM25 worked by hand, meaning search, the merge, the not-found cut-off, the vector cache, the llama.cpp client, server start and stop |
| `test_evaluation_tools.py` | the answer grader, the question builder, the draft importer, the scale corpus, keep-or-undo, the fine-tuning examples |
| `test_api_page.py` | the web page and the API routes |
| `test_enforcement.py` | rules first, the prompt, the answer check, and the path for 5,000 generated schemes |
| `test_retriever.py`, `test_corpus_pipeline.py` | search results for saved questions; loading and swapping corpora |
| `test_population.py` | the older audit with users from `../jansankhya/` |

---

## 2. Run the chatbot

```bash
python scripts/run_api.py
```

Open http://127.0.0.1:8000/ui. This starts both model servers, serves the page, and stops the servers again on Ctrl+C. Servers that were already running (for example from `serve_models.py --detach`) are left running.

| Option | Effect |
|---|---|
| `--backend rules_only` | no answer model: the matching rules are shown as they are |
| `--backend llama_cpp` | the default: Qwen3.5-0.8B writes the answer |
| `--domain welfare_demo --backend naive` | the older two-scheme demo |
| `--port 8080` | another port |

The page shows the answer, a plain label for how it was produced, and the scheme sections it used, each with a link to the official page. Thumbs up or down (with an optional note) is saved to `data/feedback/feedback.jsonl` on this machine only; that folder is not in git.

The same server answers these routes:

| Route | Body | Returns | Answer model? |
|---|---|---|---|
| `GET /` | | domain, backend, whether the model is ready, search settings, corpus fingerprint | no |
| `POST /search` | `{"question": "...", "top_k": 3}` | the most similar sections | no |
| `POST /context` | `{"question": "..."}` | the sections the model would get (rules first) and the full prompt | no |
| `POST /ask` | `{"question": "..."}` | the answer, its `mode`, and the sections used | yes |
| `POST /feedback` | `{"question", "answer", "mode", "rating": "up" or "down", "note", "evidence_ids"}` | saves one line | no |

`mode` tells the person how the answer was produced:

| Mode | Meaning |
|---|---|
| `answered` | the model's answer, passed by the check |
| `no_evidence` | nothing was close enough in meaning; the model was not asked |
| `unclear` | the input was not readable as a question |
| `blocked` | the model's answer made a forbidden promise; the rules text is shown instead |
| `rules_only` | no answer model in use; the matching rules are shown |
| `model_error` | the model failed or said nothing; the rules text is shown instead |

To keep the models loaded between runs:

```bash
python scripts/serve_models.py --detach      # start both in the background
python scripts/serve_models.py --stop        # stop them
```

To see what search would give the model for one question, without the answer model:

```bash
python scripts/search_corpus.py --query "budhape ki pension" --context
```

---

## 3. Correct or add a scheme

Scheme files are in `domains/india_schemes/schemes/`, one YAML file each. The full editing guide is [domains/india_schemes/README.md](domains/india_schemes/README.md). In short: edit the file, then

```bash
python scripts/build_corpus.py --check       # report problems only
python scripts/build_corpus.py               # rebuild the sections
python scripts/build_eval_questions.py       # only if facts used by questions changed
```

The checker refuses a rule or fact number that the scheme's own statements do not contain, a missing source, an unknown category, and similar. The next start of the chatbot re-embeds only the sections that changed.

Rebuilding the questions changes their SHA-256. Results measured on the old file are then no longer comparable question by question, and `compare_runs.py` will refuse to compare across the change.

---

## 4. Measure the pipeline

Start the models once, then run the dev questions (for tuning) or the test questions (for reporting):

```bash
python scripts/serve_models.py --detach
python scripts/evaluate_pipeline.py --split dev --label my_run
```

It writes `data/evaluation/india_schemes/<label>/`:

| File | Contents |
|---|---|
| `report.md` | the tables: search alone, answers by condition, paired differences, each with 95% intervals |
| `report.json` | the same numbers, plus the question file's SHA-256 and every setting |
| `answers_<condition>.jsonl` | every question, the answer, the sections used, and the grade |

The conditions, all on the same questions:

| Condition | What the model gets |
|---|---|
| `closed_book` | the question only: what the model "knows" |
| `rules_only` | no model: the matching rules shown as they are |
| `old_search` | sections from the original letter-run search |
| `meaning` | sections from the meaning search alone |
| `fusion` | the full pipeline |
| `oracle` | the correct sections handed over: the best any search could do |

Useful options:

| Option | Effect |
|---|---|
| `--questions usecases` | the use-case questions instead (how to apply, documents, benefits, who is excluded, eligibility-and-apply, comparing two schemes, which schemes fit a need) |
| `--set named_schemes=false` | try other search settings without editing `eval_config.yaml` |
| `--runs fusion oracle` | only these conditions |
| `--search-only` | only the search table; no answer model needed |
| `--top-k 5`, `--min-score 0.45` | try other search settings |
| `--corpus <file>` | search another corpus (see section 5) |
| `--regrade` | grade the saved answers again with the current grader, without asking the model |

A full dev run takes about 30 minutes on an Apple M5; a full test run about 90.

How an answer is graded ([engine/answer_evaluation.py](engine/answer_evaluation.py), `grade()` in `scripts/evaluate_pipeline.py`):

| Question type | Correct when |
|---|---|
| fact, typo | the answer states the expected value in one accepted form, matched whole (every number in it, the same time unit; "₹2 lakh" = 200000); numbers inside scheme names and forms the question itself contains do not count; an answer that opens by refusing does not count |
| situation | the answer names the right scheme |
| eligibility | the answer opens with the right "yes" or "no" (a "yes" to someone outside a limit is counted as **unsafe**) |
| absent, off-topic | the answer says it could not find it |
| apply, documents, benefits, exclusions | not a refusal, and uses at least 2 words of the right section that no other section of that scheme uses (words of the question and of the scheme's names never count) |
| eligibility-and-apply | the same, with at least one such word from each of the two sections |
| compare | not a refusal, and for each scheme a word of its text that the other's text lacks (a weak check: it cannot tell whether the comparison is true) |
| which schemes | names at least one scheme of that kind |

The use-case questions are built with `python scripts/build_eval_questions.py --usecases` from the structure of the scheme files: a few fixed phrasings per kind, never per scheme.

---

## 5. Test search at 5,000 schemes

This needs a public copy of the myScheme catalogue at `data/external/myscheme_copy/Schemes.csv` (not in git; see the note on terms in the top-level README).

```bash
python scripts/build_scale_corpus.py
python scripts/evaluate_pipeline.py --split dev --label dev_scale \
    --corpus data/external/scale_corpus/corpus.jsonl --runs old_search meaning fusion
```

The first run turns about 27,600 sections into meaning vectors (about 22 minutes); they are cached in `models/index/` and reused.

---

## 6. Keep or undo a change

A change to **search** goes through one round:

```bash
python scripts/serve_models.py --only embedder --detach
python scripts/search_round.py --label my_change --set overview_last=false --note "why"
```

It measures the practice questions of both question sets, at 54 schemes and (if the background copy is present) at about 4,600, before and after. It keeps the change only if nothing gets worse: no question type or language, not the "not found" refusals, not with 2 or 5 sections instead of 3, and not on 206 differently worded questions. The full rule is at the top of [scripts/search_round.py](scripts/search_round.py); the record goes to `data/evaluation/india_schemes/search_rounds/`. A kept search change then needs the answer check:

```bash
python scripts/evaluate_pipeline.py --split dev --runs fusion --search-methods fusion --label after --set <the change>
python scripts/compare_runs.py data/evaluation/india_schemes/<before> data/evaluation/india_schemes/after --same-model
```

`--same-model` also requires identical answers wherever the evidence did not change, and no question type or language losing a correct answer. Any other change (instructions, the model) is compared directly:

```bash
python scripts/compare_runs.py data/evaluation/india_schemes/dev_round0 data/evaluation/india_schemes/my_run
```

It compares the `fusion` condition (or `--run <condition>`) question by question and prints the difference with a 95% interval, unsafe "yes" answers before and after, results by question type, and a verdict:

- **KEEP**: the whole interval is above zero and unsafe "yes" answers did not increase.
- **UNDO**: the interval is below zero, or unsafe "yes" answers increased.
- **NO CLEAR DIFFERENCE**: anything else. Keep the simpler version.

Tune on `--split dev` only. Run `--split test` once, for the final numbers.

---

## 7. Improve the answer model (SynAlign)

Instruction changes were tried and undone (see the top-level README). The next step is fine-tuning on the skills the model gets wrong.

**1. Make the examples** (no model needed):

```bash
python scripts/make_scheme_training_data.py
```

This writes `data/training/india_schemes/sft_train.jsonl` (210 examples) and `sft_dev.jsonl` (42 examples). They contain yes/no eligibility answers for ages and incomes on both sides of each limit, and "not found" answers. Both are in English and Hindi, in the chatbot's own prompt layout, from dev schemes only. Read a sample before training.

**2. Train** (needs the full-precision model and `pip install -r requirements-model.txt`; Qwen3.5 needs a `transformers` version that knows its architecture):

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
python scripts/train_qwen_lora_sft.py --model models/Qwen3.5-0.8B \
    --train data/training/india_schemes/sft_train.jsonl --dev data/training/india_schemes/sft_dev.jsonl \
    --output models/qwen3.5-0.8b-schemes-lora
python scripts/merge_qwen_lora.py --base models/Qwen3.5-0.8B \
    --adapter models/qwen3.5-0.8b-schemes-lora --output models/qwen3.5-0.8b-schemes-merged
```

**3. Convert for llama.cpp** (scripts from the [llama.cpp repository](https://github.com/ggml-org/llama.cpp)):

```bash
python llama.cpp/convert_hf_to_gguf.py models/qwen3.5-0.8b-schemes-merged \
    --outtype q8_0 --outfile models/Qwen3.5-0.8B-schemes-Q8_0.gguf
```

**4. Measure, then keep or undo.** Point the answer server at the new file, run dev, compare:

```bash
python scripts/serve_models.py --stop
cp models/Qwen3.5-0.8B-Q8_0.gguf models/Qwen3.5-0.8B-Q8_0.original.gguf
cp models/Qwen3.5-0.8B-schemes-Q8_0.gguf models/Qwen3.5-0.8B-Q8_0.gguf
python scripts/serve_models.py --detach
python scripts/evaluate_pipeline.py --split dev --label dev_finetuned --runs fusion oracle
python scripts/compare_runs.py data/evaluation/india_schemes/dev_round0 data/evaluation/india_schemes/dev_finetuned
```

Only on KEEP, run `--split test` once and report it against `test_final`. On UNDO, copy the original file back.

The dev questions are about the same 18 schemes the examples were made from. A gain on dev can therefore partly be memorised facts; only the test split, on schemes never trained on, shows whether the skill itself improved.

---

## 8. The older audit (two-scheme demo)

The original SYNALIGN audit invents users, has each ask five ways (clean, typos, missing facts, vague, misleading), and grades coverage, follow-up questions, groundedness and next steps, with automatic fails for false confirmations and forbidden promises. It runs on the two-scheme `welfare_demo` domain:

```bash
python scripts/run_audit.py                                  # no model, about two seconds
python scripts/run_diagnosis.py                              # why each failure happened: search or writing
python scripts/run_audit.py --user-source population         # users from ../jansankhya/
python scripts/run_audit.py --backend llama_cpp --users 20   # with the local model (serve_models.py running)
python scripts/make_training_data.py --input data/outputs/audit_baseline.jsonl --output-dir data/training_new
```

Its results and history are in [docs/results/EARLIER_WORK.md](docs/results/EARLIER_WORK.md).

---

## 9. Use other scheme documents

Any corpus in this one-line-per-section format can be searched and answered from:

```json
{"chunk_id": "loom__rules", "scheme_id": "loom", "section": "rules", "title": "Loom Grant — Eligibility rules", "text": "- Weavers aged 21 to 50. ...", "source": "https://..."}
```

`chunk_id` and `text` are required. Sections with the same `scheme_id` are one scheme, and `"section": "rules"` marks what must always reach the model. Details: [docs/SINGLE_NODE_RETRIEVAL.md](docs/SINGLE_NODE_RETRIEVAL.md).

```bash
SYNALIGN_CORPUS_PATH=/absolute/path/corpus.jsonl python scripts/run_api.py
```

To use a new subject entirely, copy `domains/india_schemes/`, replace its files, and pass `--domain <name>`.

---

## Settings

| Setting | Where | Meaning |
|---|---|---|
| `retrieval.method` | `domains/<domain>/eval_config.yaml` | `fusion` (default for india_schemes), `dense`, `bm25`, `char_tfidf`, `word_tfidf`, `hybrid` |
| `retrieval.top_k` | same | sections in the model's prompt (3) |
| `retrieval.min_score` | same | meaning similarity below which nothing is retrieved and the model is not asked (0.50) |
| `retrieval.fusion_weights`, `rrf_k`, `fusion_depth` | same | how the three searches are merged; chosen on dev |
| `retrieval.add_named_schemes`, `name_rules` | same | a scheme the question names but the prompt lacks is brought in (on for india_schemes, adopted in SynAlign round 4; `name_rules: 2` was tried in round 5b and undone) |
| `retrieval.rules_count_once`, `named_schemes`, `overview_last` | same | tried and undone; kept, off, so the rounds can be reproduced (see [docs/results/SYNALIGN_RETRIEVAL_ROUNDS.md](docs/results/SYNALIGN_RETRIEVAL_ROUNDS.md)) |
| `SYNALIGN_DOMAIN` | environment | domain pack (`run_api.py` default: `india_schemes`) |
| `SYNALIGN_ASSISTANT_BACKEND` | environment | `llama_cpp`, `rules_only`, `naive`, `transformers`, `ollama` |
| `SYNALIGN_CORPUS_PATH` | environment | search this corpus instead of the domain's own |
| `SYNALIGN_RETRIEVAL_METHOD`, `_TOP_K`, `_MIN_SCORE` | environment | override the domain's search settings |
| `SYNALIGN_EMBEDDER_URL`, `SYNALIGN_DECODER_URL` | environment | model servers elsewhere (default ports 8091 and 8092) |

---

## All scripts

| Script | Purpose | Model? |
|---|---|---|
| `run_api.py` | the chatbot: web page and API | yes, unless `--backend rules_only` |
| `serve_models.py` | start or stop the two model servers | — |
| `search_corpus.py` | search, or show the prompt for one question | embedder for meaning search |
| `build_corpus.py` | check scheme files and rebuild the sections | no |
| `build_eval_questions.py` | write the frozen test questions (`--usecases`: the use-case questions) | no |
| `build_scale_corpus.py` | add ~4,600 background schemes for the scale test | no |
| `evaluate_pipeline.py` | search and answer measurements against baselines | yes |
| `search_round.py` | one SynAlign round for a search change: before/after on practice questions, kept only if nothing gets worse | embedder |
| `compare_runs.py` | keep or undo a change, from two answer runs | no |
| `make_scheme_training_data.py` | fine-tuning examples from the dev schemes' rules | no |
| `train_qwen_lora_sft.py`, `merge_qwen_lora.py` | fine-tune and merge | yes (full-precision weights) |
| `import_scheme_drafts.py` | turn researched and checked drafts into scheme files | no |
| `run_audit.py`, `run_diagnosis.py`, `regrade_audits.py`, `make_training_data.py`, `smoke_test.py` | the older audit | optional |
| `build_scheme_corpus.py`, `evaluate_context.py`, `calibrate_retrieval.py`, `evaluate_retrieval.py`, `build_retrieval_stress.py` | the older invented 5,000-scheme tests | no |
