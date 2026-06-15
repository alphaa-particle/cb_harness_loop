# SYNALIGN

A domain-agnostic quality-improvement loop for RAG assistants, with a working
**Qwen3-0.6B** RAG assistant out of the box.

SYNALIGN stress-tests a RAG chatbot with synthetic users and messy question
styles (clean / vague / typo-heavy / missing-info / misleading), measures
exactly where and why it fails, and proves whether a fix worked — on held-out
cases, with confidence intervals and safety gates.

The full design rationale lives in `build_v2.md`. This README is the quick
start.

---

## What you get

```
synalign/
├── engine/            # domain-agnostic core (never changes per domain)
│   ├── qwen_assistant.py   # Qwen3-0.6B RAG assistant (transformers + Ollama)
│   ├── assistant.py        # NaiveBaselineAssistant (instant, no deps)
│   ├── retriever.py        # TF-IDF retriever with stable chunk IDs
│   ├── evaluator.py        # 3-layer evaluator with safety gates
│   ├── audit.py / diagnosis.py / training_data.py / ...
├── domains/
│   └── welfare_demo/  # the swappable "domain pack" (documents + rules + config)
├── scripts/
│   ├── run_audit.py        # audit the naive baseline (sanity check the loop)
│   ├── smoke_test_qwen.py  # load Qwen, answer 3 questions
│   ├── run_audit_qwen.py   # audit Qwen3-0.6B end to end
│   ├── run_diagnosis.py / make_training_data.py / run_api.py
└── data/outputs/      # all generated audit/diagnosis/training files
```

---

## 1. Setup

```bash
cd synalign
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify the framework runs with the zero-dependency baseline first:

```bash
python scripts/run_audit.py
```

You should see per-split, per-condition score tables with confidence intervals.
`vague` and `missing_info` will score low (the baseline never asks follow-up
questions) — that is the loop correctly finding real failures.

---

## 2. Add the Qwen3-0.6B assistant

Pick **one** backend.

### Option A — transformers (no server, recommended)

```bash
pip install -r requirements-qwen.txt   # torch + transformers (~a few hundred MB)
python scripts/smoke_test_qwen.py      # downloads Qwen/Qwen3-0.6B on first run
```

The first run downloads the model (~1.2 GB) from Hugging Face. CPU works but is
slow; a GPU is used automatically if available.

### Option B — Ollama (lighter Python deps, needs Ollama installed)

```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen3:0.6b
pip install requests
python scripts/smoke_test_qwen.py ollama
```

---

## 3. Audit Qwen end to end

```bash
python scripts/run_audit_qwen.py                # transformers, 20 users (100 cases)
python scripts/run_audit_qwen.py --users 8      # smaller / faster
python scripts/run_audit_qwen.py --backend ollama
python scripts/run_audit_qwen.py --sample       # nucleus sampling instead of greedy
```

This prints, per question-condition: pass rate, overall score with a 95%
bootstrap confidence interval, coverage / follow-up / groundedness, retrieval
recall@k and MRR, and gate-violation counts. It then prints failure types and a
**retrieval-vs-generation attribution** (it re-answers each failure with the
gold documents forced into context — if it then passes, retrieval was the
bottleneck; if it still fails, the model is).

> A 0.6B model is small. Expect weak follow-up behavior and some groundedness
> misses — exactly the failures the improvement loop is built to fix. Keep
> `--users` low on CPU; raise it (and use a GPU) for a more stable estimate.

---

## 4. The improvement loop

```bash
python scripts/run_audit_qwen.py            # 1. baseline numbers
                                            # 2. read the attribution table
# 3. fix the dominant failure class:
#      retrieval failures  -> better chunking / embeddings / top_k (engine/retriever.py)
#      generation failures -> tune the prompt (engine/qwen_assistant.py: SYSTEM_RULES)
#                             or build SFT data (below)
python scripts/make_training_data.py        # SFT + preference data from TRAIN split only
# 4. apply fix, then re-audit with a new label and compare on the TEST split:
python scripts/run_audit_qwen.py --label qwen_v2
```

Ship only when the approval gates in `build_v2.md` §28 all hold on the **test**
split (overall up with non-overlapping CIs, worst condition up, zero new gate
violations, retrieval not regressed, human review passed).

---

## 5. Serve the assistant

```bash
# choose the backend in engine/config.py via ASSISTANT_BACKEND = "qwen" | "ollama" | "naive"
python scripts/run_api.py
# open http://localhost:8000/docs  and POST {"question": "..."} to /ask
```

---

## 6. Switching domains

Everything domain-specific is in `domains/welfare_demo/`. To target a new domain
(college admissions, healthcare, finance, internal knowledge base): copy that
folder, replace the six files (documents, profile_schema.json, aliases.yaml,
perturbations.yaml, eval_config.yaml, ground_truth.py), and set
`ACTIVE_DOMAIN = "your_domain"` in `engine/config.py`. The engine and all
scripts are unchanged. Full worked example in `build_v2.md` §30.

---

## Key configuration knobs (`engine/config.py`)

| Setting | Meaning |
|---|---|
| `ACTIVE_DOMAIN` | which domain pack to load |
| `ASSISTANT_BACKEND` | default assistant: `naive` / `qwen` / `ollama` |
| `N_SYNTHETIC_USERS` | users for the baseline audit |
| `N_USERS_LLM` | default users for the Qwen audit (kept small for CPU) |
| `SPLIT_FRACTIONS` | train / dev / test split sizes |

Qwen decoding (greedy + repetition penalty by default for reproducible audits)
and the grounding system prompt are in `engine/qwen_assistant.py`.
