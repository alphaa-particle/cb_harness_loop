# Scheme Helper, tested by SYNALIGN

**Scheme Helper** is a chatbot that answers questions about Indian government welfare schemes. Questions can be in English, Hindi, or Hindi typed in English letters ("budhape ki pension"). It runs on one laptop: two small open models, no cloud service, no database. Every answer the model writes comes with the scheme sections it was given, each linked to the official page it was written from.

**SYNALIGN** is the test bench around it. It asks the chatbot a fixed set of questions with known answers, compares it with simpler baselines, and decides by fixed rules whether a change is kept or undone.

---

## What happens to a question

```
"मेरी उम्र 45 है, क्या मैं अटल पेंशन योजना में जुड़ सकता हूँ?"
   │
   1. Hindi and Hinglish words get their English twins added        पेंशन → pension   (glossary.yaml,
                                                                     word and letter searches only)
   2. Three searches over every section of every scheme:
        word match (BM25)  ·  letter-run match (survives typos)  ·  meaning match (Qwen3-Embedding-0.6B)
   3. The three rankings are merged; meaning counts most
   4. Nothing close enough in meaning?  →  a fixed reply in English and Hindi: "I could not find a scheme
                                           that matches your question. Please tell me the scheme's name, ..."
                                           The answer model is not asked.
   5. Every scheme found brings its eligibility rules first; 3 sections at most. A scheme the question
      names but search left out ("How is X different from Y?") is brought in, in place of the last slot
   6. Qwen3.5-0.8B is told to write a 2–4 sentence answer in the question's language, from that text only
   7. Uses a listed promise ("you will surely get", "पक्का मिलेगा", 13 phrases)?  →  replaced by the rules text
   8. Answer + the sections used + official links, on the web page
```

Steps 1–5 and 7 are plain code and are tested without a model. Step 2's meaning search and step 6 use the two local models.

Why each step is there:

| Step | Without it |
|---|---|
| Glossary | "budhapa pension" shares no words with "old age pension" |
| Word match | exact scheme names and numbers can lose to a vaguer meaning match |
| Letter-run match | "pradhan mantri awass yojna" matches nothing word for word |
| Meaning match | "We cook on firewood and the smoke hurts my mother's eyes": word matching picks child-protection and maternity schemes; meaning search finds Ujjwala (free cooking gas) |
| Not-found cut-off | the model is shown the nearest wrong scheme and answers from it |
| Rules first | "how to apply" sections repeat the scheme's name and push the rules out of the prompt |
| Answer check | a guard against promising approval or money; it never fired in the tests, and it cannot catch a wrong "yes" |

---

## Results

All numbers are on the **held-out test questions**: 764 questions about 36 schemes that were never used to choose any setting. Each cell is the share answered correctly; brackets give the 95% interval from resampling whole schemes. Full tables: [test_final/report.md](synalign/data/evaluation/india_schemes/test_final/report.md). These are the results before the search cycle described next; the cycle changed them only where noted.

### After the SynAlign search cycle: more kinds of questions

A second test set covers what people ask beyond single facts: 544 questions on how to apply, documents, benefits, who is excluded, eligibility-and-apply, comparing two schemes, and which schemes fit a need. On it, a SynAlign cycle of search rounds kept one change: **a scheme the question names but the prompt lacks is brought in.**

On the held-out exam questions, search with that change gets the needed text to the AI for:

| Exam questions | Before | After |
|---|---:|---:|
| Use cases, 54 schemes | 74.6% | **82.0%** (comparisons: both schemes now present) |
| Use cases, 4,600 schemes | 67.5% | **72.4%** |
| Original questions, 54 / 4,600 schemes | 90.6% / 87.4% | 91.0% / 87.9% |

- No exam question got worse in search, and answers on the original questions at 54 schemes did not change.
- Four other changes improved search but were undone, because the 0.8B answering AI got some answers wrong when its prompt changed.
- A fifth change passed every practice check but broke exam answers at 4,600 schemes, and was undone.

Every round, kept or undone, is in [SYNALIGN_RETRIEVAL_ROUNDS.md](synalign/docs/results/SYNALIGN_RETRIEVAL_ROUNDS.md).

### Answers, 54 schemes

| What the answer model gets | All | Facts | Typos | Situations | Eligibility | Not found | Unsafe "yes" |
|---|---|---:|---:|---:|---:|---:|---:|
| The question only (model alone) | 13.9% (11–16) | 8.6% | 12.2% | 0.0% | 45.2% | 0 of 44 | 54.0% |
| Old search: letter runs | 61.7% (56–67) | 58.8% | 80.2% | 38.9% | 68.5% | 22 of 44 | 11.3% |
| Meaning search only | 77.5% (74–81) | 77.3% | 90.8% | 55.6% | 71.8% | 40 of 44 | 10.5% |
| **Full pipeline** (all three searches) | **76.4% (72–80)** | 75.8% | 90.8% | 51.4% | 75.0% | 37 of 44 | 10.5% |
| The right sections, handed over (ceiling) | 82.4% (79–85) | 86.8% | 95.4% | 56.9% | 69.3% | – | 8.9% |
| *No model: the matching rules shown as they are* | *74.1% (69–79)* | *89.6%* | *97.0%* | *87.5%* | *0.0%* | *24 of 44* | *0.0%* |

- **Facts / typos:** the answer states the right age limit, amount or premium.
- **Situations:** a need described without a scheme name; the answer must name the right scheme.
- **Eligibility:** "I am 44. Can I join APY?"; the answer must open with the right yes or no.
- **Not found:** questions about schemes not in the 54, or not about schemes at all.
- **Unsafe "yes":** the share of eligibility questions where the answer said yes to someone outside a limit.

The same questions, compared one by one:

| Comparison | Difference | 95% interval |
|---|---:|---|
| Full pipeline vs the model alone | **+62.6 points** | +58.1 to +67.1 |
| Full pipeline vs the old search | **+14.8 points** | +11.1 to +18.7 |
| Full pipeline vs meaning search only | −1.1 points | −2.9 to +0.7 (no clear difference) |
| Ceiling vs full pipeline | +6.4 points | +3.7 to +9.3 |

What this says, plainly:

1. **Finding the right scheme is most of the gain.** On its own, the model answers 14% correctly, and it tells 67 of the 72 people who do not qualify that they can join (it says the maximum age to join the Atal Pension Yojana is 60; it is 40). With our search in front of it, it answers 76%.
2. **Better search adds 15 points over the old search.** The old search here already had rules first and the Hindi glossary, so the 15 points come from the search itself. Hindi questions go from 38% to 64% correct, and facts from 59% to 76%.
3. **Meaning search does the work; word matching added nothing at this size.** On the dev questions, adding word and letter-run matching looked slightly better. On the held-out questions it was 1.4 points worse. Neither difference is clear. At 4,600 schemes the picture is different (below).
4. **Search now costs about 6 points.** The ceiling run skips the 44 "not found" questions (there is nothing to hand over). On the other 720, handing over the right sections scores 6.4 points more than the full pipeline. It is not a strict ceiling, though: on yes/no questions it did worse (69.3% against 75.0%).
5. **The rest is the model.** With the right sections in front of it, the 0.8B model still gets about 1 in 5 answers wrong. Its worst habit is in Hindi: it opens with "हाँ" (yes) and then quotes the very limit that rules the person out.
6. **Why showing the rules without a model scores 74%.** A whole rules section contains the asked number, so it counts as stating it. But it never answers yes or no, it cannot refuse an absent scheme properly, and it is always in English. The model is what turns the rules into an answer, in the asker's language. All 206 answered Hindi questions got a Hindi reply.

Wrong "yes" answers, counted among the people who do not qualify (36 per language; a wrong "yes" is only possible for them):

| Unsafe "yes" on eligibility questions | English | Hindi |
|---|---:|---:|
| Model alone | 31 of 36 | 36 of 36 |
| Full pipeline | 1 of 36 | 12 of 36 |
| Right sections handed over | 1 of 36 | 10 of 36 |

### Search alone, no answer model

How often the right section reaches the model's prompt (3 sections, each scheme's rules first), and at full size how often the right scheme is ranked first:

| Search | In prompt, 54 schemes | In prompt, 4,600 schemes | Ranked first, 4,600 | Hindi in prompt, 4,600 | ms per question |
|---|---:|---:|---:|---:|---:|
| Letter runs (the old default) | 72.5% | 62.6% | 74.9% | 52.1% | 2 |
| Word match (BM25) | 70.0% | 66.4% | 74.6% | 48.9% | 1 |
| Meaning (Qwen3-Embedding-0.6B) | 92.8% | 85.1% | 83.9% | 79.3% | 44 |
| **All three merged** | 90.6% | **87.4%** | **90.4%** | **82.0%** | 48 |

### At 5,000-scheme size

The same held-out questions, with our 54 schemes hidden among about 4,600 real background schemes ([test_scale/report.md](synalign/data/evaluation/india_schemes/test_scale/report.md)):

| What the answer model gets | All | Questions naming a scheme | Situations | Unsafe "yes" |
|---|---|---:|---:|---:|
| Old search: letter runs | 54.3% (48–60) | 60.5% | 19.4% | 10.5% |
| Meaning search only | 68.7% (64–73) | 76.5% | 23.6% | 11.3% |
| **Full pipeline** | **70.8% (66–75)** | **77.9%** | 33.3% | 10.5% |

| Comparison, 4,600 schemes | Difference | 95% interval |
|---|---:|---|
| Full pipeline vs the old search | **+16.5 points** | +13.0 to +20.3 |
| Full pipeline vs meaning search only | +2.1 points | 0.0 to +4.2 |

- **Size costs almost nothing where it matters.** For questions that name a scheme (facts, typos, eligibility: 648 questions), the full pipeline answers 78.7% correctly among 54 schemes and 77.9% among 4,600.
- **Word matching earns its place at full size.** Among thousands of similar schemes, many of them state versions of central ones, exact names and numbers pick the right scheme first 90% of the time against 84% for meaning alone, and the answers are 2 points better. The interval just touches zero, so the keep-or-undo rule calls it "no clear difference". Merged search stays the default because it was chosen on dev for the 5,000-scheme target, and the held-out run points the same way. For a corpus of a few dozen schemes, meaning search alone (`method: dense` in `eval_config.yaml`) is as good and simpler.
- **Two kinds of question cannot be scored fairly at full size.** "Situations" counts only answers naming one of our 54 schemes, but a state scheme from the background is often a genuine match. The "All" column also includes the 30 questions about schemes we do not cover; real state versions of those exist in the background, so finding one is marked wrong even when it is right. Compare the "Questions naming a scheme" column instead (the 648 fact, typo and eligibility questions).

---

## The scheme data

- **54 central government schemes**, one editable file each in [domains/india_schemes/schemes/](synalign/domains/india_schemes/schemes/): who the scheme is for, its rules, what it gives, how to apply, which documents are needed, checkable facts, and the official pages they came from.
- **How they were made.** One research agent wrote each scheme from official pages in its own words. For 39 of the 54, a second agent checked every number and rule against official pages it opened itself, and a third settled disagreements; the other 15 were never checked, because checking was stopped early at the owner's request. 34 schemes are `verified` (every test fact confirmed on an official page); 20 are `needs_review`. Even verified files keep smaller open points from the checker in their notes, and 17 schemes have no official Hindi name yet.
- **Correcting a fact** is: edit the scheme file, run `python scripts/build_corpus.py`, and restart the chatbot. A checker refuses a rule or fact number that the scheme's own statements do not contain; it cannot tell whether a number is right, so each edit still needs the official page. The steps are in [domains/india_schemes/README.md](synalign/domains/india_schemes/README.md).
- **Sections.** Each scheme is cut into overview, rules (with exclusions kept next to them), benefits, how to apply, and documents. A long section is split only between sentences, at most 1,200 characters per piece. 286 sections in all.
- **Test questions.** 1,080 questions written from the checked facts, never from what the search returns: facts (age limits, amounts, premiums), the same with typos, situations described without naming a scheme, yes/no eligibility questions on both sides of each limit, real schemes that are not in the 54, and off-topic questions. 545 in English, 332 in Hindi, 203 in Hindi typed in English letters. Split by scheme: 18 schemes for tuning (dev), 36 for the reported results (test).

### Scaling to 5,000 schemes

To test search at full size, about 4,600 real schemes from a public copy of the myScheme catalogue are added as background: 27,593 sections in all. Background schemes whose names match one of ours are removed (142 entries, matching 40 of our 54), so a question about PM-KISAN cannot be "answered" by myScheme's copy instead of ours. The match is by name, so it also removed many state versions and a few different schemes with similar names. That makes the full-size test a little easier than a real catalogue.

That copy was scraped from myscheme.gov.in against its terms of use, so it is used **only inside tests**: it is not in git, its text was not checked, and the chatbot never answers people from it. (The full-size answer test does show the model background sections when search picks them.) For a real deployment of thousands of schemes, the authorised route is the API Setu myScheme APIs or written permission from myScheme.

---

## The models

| Job | Model | File (8-bit) | Size |
|---|---|---|---:|
| Meaning search | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) | `Qwen3-Embedding-0.6B-Q8_0.gguf` | 639 MB |
| Writing answers | [Qwen3.5-0.8B](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF) | `Qwen3.5-0.8B-Q8_0.gguf` | 834 MB |

- Both run in [llama.cpp](https://github.com/ggml-org/llama.cpp) (`brew install llama.cpp`), each as a small local server. The Python side talks to them over HTTP with the standard library only.
- **8-bit (Q8_0)** keeps the models' quality close to full precision at about half the size. Together they need under 2 GB of memory, so there is no reason to go lower and risk quality.
- **Settings that make results repeatable:** greedy decoding with a fixed seed, "thinking" turned off, and no prompt cache, so one answer cannot leak into the next.
- **Speed on an Apple M5 laptop (16 GB):** about 50 ms to search, whether among 54 schemes or 4,600; about 1.2 s per answer. Turning the 27,593 background sections into meaning vectors takes about 22 minutes once; the vectors are cached, and only changed sections are redone.
- The meaning model was checked against its official reference example before use, and questions get the instruction prefix its model card asks for.

---

## SynAlign: how changes are kept or undone

The chatbot is only changed through this loop:

1. Change **one** thing (the instructions, a search setting, a model).
2. Run the 316 dev questions: `python scripts/evaluate_pipeline.py --split dev --label <name>`.
3. Compare with the previous run, question by question: `python scripts/compare_runs.py <before> <after>`.
   - **KEEP** only if the whole 95% interval of the difference is above zero **and** unsafe "yes" answers did not increase.
   - **UNDO** if the interval is below zero or unsafe "yes" answers increased.
   - Otherwise **NO CLEAR DIFFERENCE**: the questions cannot tell the two apart. The guidance is then to keep the simpler version; that last step is a person's call.
4. Only settings that survive are run on the test questions, once, for the reported numbers.

These rules were written down before any round was run. The search settings (how much each search counts, the not-found cut-off, 3 sections) were chosen the same way, on dev only. Search changes go through a stricter version of the loop, [scripts/search_round.py](synalign/scripts/search_round.py): no question type or language may lose anything, at either catalogue size, and the answers are checked too. The full record of that cycle is [SYNALIGN_RETRIEVAL_ROUNDS.md](synalign/docs/results/SYNALIGN_RETRIEVAL_ROUNDS.md).

### What was tried on the answer model

| Round | Change | Correct (dev) | Unsafe "yes" | Verdict |
|---|---|---|---:|---|
| 0 | starting instructions | 82.3% | 4 | baseline |
| 1 | longer instructions (worked example, "name the scheme", "open with Yes/No") and a repetition penalty | −9.2 points (−14.8 to −3.7) | 15 | **UNDO** |
| 2 | the repetition penalty alone | −0.3 points (−5.2 to +3.9) | 7 | **UNDO** |
| 3 | one added line: "Always name the scheme you are talking about" | −3.2 points (−5.2 to −1.0) | 6 | **UNDO** |

Each round made things worse or added unsafe answers, so the starting instructions are kept. Round 1 is the loop doing its job: a change that looked sensible made the model say "yes" to ineligible people almost four times as often.

### Where the remaining errors are, and the next lever

With the right sections handed over, the model answers 82.4% of the held-out questions correctly, and real search costs only about 6 points more. The remaining errors are the model's own, of two kinds:

- **Comparing a person's age or income with a limit, in Hindi.** Of 36 Hindi-speaking people who do not qualify, 10 got a wrong "yes" even with the right rules in front of the model, against 1 of 36 in English. The answer typically opens with "हाँ" and then quotes the limit that rules the person out.
- **Naming the scheme** when the question only describes a situation (57% even with the right sections).

Instruction wording did not fix either (rounds 1–3). Two levers remain.

**1. Fine-tuning on exactly those skills.** Everything for it is ready:

```bash
python scripts/make_scheme_training_data.py          # 252 examples from the dev schemes' checked limits
```

- Yes/no eligibility answers in English and Hindi, for ages and incomes on both sides of each limit, worked out from the scheme's own rules, in the exact prompt layout the chatbot uses.
- "Not found" answers for questions about one scheme when another scheme's rules are shown.
- Only dev schemes are used, so the test schemes are never trained on; 4 dev schemes are held back to watch for over-fitting.

Training needs the full-precision Qwen3.5-0.8B weights and the training software (`requirements-model.txt`), neither of which is installed; the training script was written for a machine with a graphics card. Then: `scripts/train_qwen_lora_sft.py`, `scripts/merge_qwen_lora.py`, convert to GGUF with llama.cpp, run dev, `compare_runs.py`, and only on KEEP, test. The commands are in [synalign/README.md](synalign/README.md#7-improve-the-answer-model-synalign).

**2. A number check on the answer (not built).** Every scheme file already holds its checked limits (`age_min`, `age_max`, income caps), and the checker ensures they match the scheme's own text. When a question states an age or income outside those limits and the answer opens with "yes", the answer could be replaced by a plain "No" quoting the limit, the same way forbidden promises are replaced today. It would use the scheme files' numbers rather than code written for each scheme, and it would be tested as a round like any other.

---

## Run it

Needs Python 3.10+, [Homebrew](https://brew.sh) and about 2 GB of disk for the models.

```bash
cd synalign
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install llama.cpp
mkdir -p models
curl -L -o models/Qwen3-Embedding-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf
curl -L -o models/Qwen3.5-0.8B-Q8_0.gguf https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf
python scripts/run_api.py
```

Open http://127.0.0.1:8000/ui. The first start turns the 286 sections into meaning vectors (well under a minute); later starts reuse them.

Run every test (no model needed; stand-ins replace both models):

```bash
python -m unittest discover -s tests
```

Every other command (rebuilding the corpus, the evaluation, the scale test, fine-tuning) is in [synalign/README.md](synalign/README.md).

---

## Repo map

```
cb_harness_loop/
├── README.md                      this file
├── jansankhya/                    10,000 made-up informal workers, used as test users by the older audit
└── synalign/
    ├── README.md                  every command
    ├── domains/
    │   ├── india_schemes/         the real chatbot: 54 scheme files, built sections, test questions,
    │   │                          instructions, search settings, glossary
    │   └── welfare_demo/          the older two-scheme demo with its invented users
    ├── engine/
    │   ├── scheme_records.py      checks scheme files, cuts them into sections, writes fact questions
    │   ├── corpus.py              loads sections (JSONL or Markdown)
    │   ├── retriever.py           the three searches, merging, not-found cut-off, rules first
    │   ├── llama_cpp.py           starts and talks to the two local model servers
    │   ├── enforcement.py         builds the prompt; checks the answer
    │   ├── llm_assistant.py       the model-backed chatbot (llama.cpp, transformers or Ollama)
    │   ├── answer_evaluation.py   grades answers: numbers, yes/no, "not found", language
    │   ├── api.py                 the web page and /search, /context, /ask, /feedback
    │   └── …                      the older audit: invented users, 5 question styles, scores, diagnosis
    ├── ui/index.html              the web page
    ├── scripts/                   command-line entry points
    ├── tests/                     no model needed
    ├── docs/                      editing guide, retrieval notes, earlier reports
    ├── data/evaluation/           saved measurements (india_schemes/ holds every run above)
    └── models/                    the two model files and cached vectors (not in git)
```

---

## Limits worth knowing

- **20 of the 54 schemes are `needs_review`.** They are in the corpus and the questions, because the owner chose to edit facts later; each file lists what is unchecked. Schemes change: MGNREGA was replaced by VB-G RAM G from 1 July 2026, and some schemes (Stand-Up India, PM-KUSUM, PMFME) are closed or uncertain. `review.checked_on` is the date the facts were true.
- **Hindi questions and the glossary were machine-drafted** and not checked by a native speaker.
- **The grader is plain rules, not a judge model:** numbers, yes/no openings, "not found" phrases, and the script of the answer. It was corrected several times, the last time after an independent review: each accepted form of a number must match whole ("18 to 40" needs both numbers, "3 weeks" is not "72 hours"), repeating the question never counts, and an answer that opens by refusing does not count. Every saved run was graded again with the final rules. Comparison answers can only be checked weakly; a hand check of a sample is still worthwhile.
- **At 4,600 schemes, "not in the corpus" questions no longer mean much**: many state versions of an absent central scheme exist in the background, so retrieving one is not wrong. That column is reported at 54 schemes only.
- **A 0.8B model still misjudges eligibility**, mostly in Hindi: of 36 Hindi-speaking people who do not qualify, the full pipeline told 12 "yes", against 1 of 36 in English.
- **In Hindi the model sometimes just repeats the question.** 46 of 764 exam answers begin by repeating the question word for word: 30 in Hindi, 15 in Hindi typed in English letters, 1 in English. The answer check catches promises, not wrong comparisons; the page tells people to check the rules shown under every answer.
- **The answer model rarely emits a stray non-Latin, non-Devanagari word** (1 of 316 dev answers).

---

## Earlier work

Before this pipeline, SYNALIGN was tested on a two-scheme demo with invented users and on 5,000 invented schemes. Those results, and why they led to this design, are in [docs/results/EARLIER_WORK.md](synalign/docs/results/EARLIER_WORK.md). Older detailed reports are in [synalign/docs/results/](synalign/docs/results/).
