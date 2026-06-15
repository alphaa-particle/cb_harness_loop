# CB Harness Loop

This repository contains **SYNALIGN**, a quality-testing harness for
document-grounded chatbots.

SYNALIGN stress-tests a RAG assistant with synthetic users and messy question
styles, then scores whether the assistant:

- finds the right evidence,
- mentions the right schemes,
- asks follow-up questions when information is missing,
- avoids unsafe eligibility confirmations,
- stays grounded in the source documents.

The current working project lives in:

```text
synalign/
```

The older one-file proof of concept, `synalign_poc.py`, is kept for historical
context. The full codebase is the `synalign/` package.

## What Is In This Repo

```text
.
+-- README.md                         # top-level overview
+-- synalign_poc.py                   # older single-file prototype
+-- synalign/
    +-- engine/                       # reusable audit, retrieval, evaluator, API code
    +-- domains/welfare_demo/         # welfare demo domain pack
    +-- scripts/                      # audit, diagnosis, API, training-data scripts
    +-- data/outputs/.gitkeep         # output folder placeholder
    +-- README.md                     # detailed SYNALIGN quick start
    +-- FINAL_MODEL_PERFORMANCE_COMPARISON.md
    +-- QWEN_RESULTS_REPORT.md
    +-- QWEN35_RESULTS_SIMPLIFIED_REPORT.md
    +-- GEMMA_1B_RESULTS_SIMPLIFIED_REPORT.md
```

Large local artifacts are intentionally not committed:

```text
.venv/
models/
synalign/data/outputs/*.jsonl
synalign/data/outputs/*.csv
```

## Tested Models

The saved reports compare three local model runs on the same 500 welfare-advice
cases:

| Model | Passed | Pass Rate | Main Takeaway |
|---|---:|---:|---|
| Qwen3 0.6B | 51 / 500 | 10.2% | Weakest overall; many missed schemes and false confirmations. |
| Qwen3.5 0.8B | 136 / 500 | 27.2% | Best pass rate and coverage, but many unsafe confirmations. |
| Gemma 1B | 111 / 500 | 22.2% | Fewest safety gates, but often incomplete and weakly grounded. |

Read the full comparison here:

```text
synalign/FINAL_MODEL_PERFORMANCE_COMPARISON.md
```

## Quick Start

Use Python 3.10+.

```powershell
cd synalign
python -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the deterministic baseline audit:

```powershell
..\.venv\Scripts\python.exe scripts\run_audit.py
```

For local transformer model audits, install the model dependencies:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements-qwen.txt
```

Then run a local model audit by passing a model path:

```powershell
..\.venv\Scripts\python.exe scripts\run_qwen_local_progress.py --cases 500 --label my_model_run --model-name F:\path\to\local\model --max-new-tokens 160 --device cuda --progress-every 10
```

The script name still contains `qwen`, but it can run compatible local
`transformers` causal language models when `--model-name` points to the local
model directory.

## Important Notes

- The committed repository does not include downloaded model weights.
- Generated audit records are ignored by Git to avoid committing large output
  files.
- The reports in `synalign/` summarize previous local runs.
- The current evaluator is useful but not perfect; numeric contradiction checks
  and stricter soft-confirmation gates are recommended next improvements.

## More Documentation

Start with:

```text
synalign/README.md
```

Then read:

```text
synalign/FINAL_MODEL_PERFORMANCE_COMPARISON.md
synalign/build_v2.md
synalign/CODEBASE_ANALYSIS.md
```
