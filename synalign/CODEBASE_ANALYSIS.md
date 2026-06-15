# SYNALIGN codebase and run analysis

Generated on 2026-06-14 from workspace `F:\cb_h_loop`, using
`E:\conda_envs\kgp_research\python.exe`.

## What this codebase is

SYNALIGN is a domain-agnostic evaluation loop for a retrieval-augmented
assistant. The project does not train a model by default; it generates synthetic
test users, perturbs their questions, runs an assistant, evaluates the answer,
and writes audit data for diagnosis and later improvement.

The current repository contains:

- `synalign/engine/`: reusable framework code.
- `synalign/domains/welfare_demo/`: the active domain pack.
- `synalign/scripts/`: command-line entry points for audits, diagnosis, data
  generation, Qwen smoke tests, and API serving.
- `models/qwen3_0.6/`: a local Qwen3-0.6B model directory.

## Main execution flow

`scripts/run_audit.py` calls `engine.audit.run_audit()`.

The audit flow is:

1. `DomainPack(ACTIVE_DOMAIN)` loads the active domain from
   `engine/config.py`. The active domain is `welfare_demo`.
2. `TfidfRetriever` chunks markdown documents on `---`, extracts stable section
   IDs, and builds a character n-gram TF-IDF index.
3. `make_synthetic_users()` samples user profiles from
   `domains/welfare_demo/profile_schema.json`.
4. `assign_splits()` deterministically assigns users to train/dev/test splits.
   The split is user-level, so all cases for one user remain in one split.
5. `make_test_cases()` creates five question variants per user:
   `clean`, `vague`, `missing_info`, `typo_heavy`, and `misleading`.
6. The assistant receives only `case.question`. It never receives the profile,
   visible fields, or ground truth.
7. Retrieval metrics compare retrieved chunk IDs with the ground-truth gold
   chunk IDs.
8. `Evaluator.evaluate()` scores the answer on coverage, follow-up behavior,
   groundedness, and actionability. It also applies safety gates.
9. Results are written to `data/outputs/audit_<label>.jsonl` and a convenience
   CSV at `data/outputs/audit_<label>_flat.csv`.

## Domain behavior

The welfare demo domain has two schemes:

- `scheme_pmsym`: PMSYM / Pradhan Mantri Shram Yogi Maandhan.
- `scheme_eshram`: e-Shram registration.

The synthetic profile includes age, income, worker type, EPFO status, literacy,
and state. Only age, income, worker type, and EPFO are used as question fields.

Ground truth is computed from visible information, not from the hidden full
profile. This is the key design choice. For vague or missing-info questions, the
correct answer is often to ask follow-up questions rather than infer facts.

Important domain rules:

- PMSYM requires age 18-40, income <= 15000, unorganised worker status, and no
  EPFO coverage.
- e-Shram requires age 16-59 and either unorganised worker or self-employed
  status.
- Answers must not guarantee approval.
- Answers must not imply automatic cash benefits.
- Answers must not confirm eligibility for a scheme that is unknown or
  ineligible according to the visible information.

## Environment check

Python executable:

```text
E:\conda_envs\kgp_research\python.exe
Python 3.10.19
```

Installed package versions from the requested environment:

```text
pandas 2.3.3
numpy 2.2.6
scikit-learn 1.7.2
scipy 1.15.3
fastapi 0.128.0
uvicorn 0.40.0
pydantic 2.11.9
PyYAML 6.0.2
torch 2.5.1+cu121
transformers 5.5.4
accelerate 1.12.0
requests 2.32.5
pyarrow 23.0.1
```

The direct requirements import check did not fully pass. `sklearn` imports
`pyarrow` as an optional dependency probe, and in this environment `pyarrow.lib`
raises:

```text
ImportError: DLL load failed while importing lib: Access is denied.
```

Because `engine.retriever` imports `sklearn`, a normal run fails before the audit
starts:

```text
python scripts/run_audit.py
```

I ran the audits by letting Pandas import normally, then injecting a minimal
in-process `pyarrow` module only for sklearn's optional version check. This did
not modify the repository or the conda environment.

Baseline audit command used:

```powershell
& 'E:\conda_envs\kgp_research\python.exe' -c "import pandas; import sys, types, runpy; m=types.ModuleType('pyarrow'); m.__version__='17.0.0'; sys.modules['pyarrow']=m; runpy.run_path('scripts/run_audit.py', run_name='__main__')"
```

Diagnosis command used:

```powershell
& 'E:\conda_envs\kgp_research\python.exe' -c "import pandas; import sys, types, runpy; m=types.ModuleType('pyarrow'); m.__version__='17.0.0'; sys.modules['pyarrow']=m; runpy.run_path('scripts/run_diagnosis.py', run_name='__main__')"
```

Local Qwen smoke audit command used:

```powershell
& 'E:\conda_envs\kgp_research\python.exe' -c "import pandas; import sys, types; m=types.ModuleType('pyarrow'); m.__version__='17.0.0'; sys.modules['pyarrow']=m; from engine.audit import run_audit, summarize, worst_case; df=run_audit(n_users=2, backend='qwen', label='qwen_local_smoke', model_name=r'F:\cb_h_loop\models\qwen3_0.6', max_new_tokens=160); print(summarize(df).to_string(index=False)); print(worst_case(df))"
```

## Baseline audit results

The baseline run used `N_SYNTHETIC_USERS = 300`, so it evaluated 1,500 cases.
Each split has five conditions.

Train split:

```text
condition       n    pass_rate  overall  coverage  followup  groundedness  recall@k  mrr  gate_violations
vague           180  0.000      0.512    0.500     0.000     0.750         1.0       1.0  0
missing_info    180  0.122      0.645    0.814     0.122     0.719         1.0       1.0  0
misleading      180  0.778      0.836    0.889     0.778     0.719         1.0       1.0  0
clean           180  0.944      0.879    0.894     0.944     0.719         1.0       1.0  0
typo_heavy      180  0.944      0.882    0.894     0.944     0.729         1.0       1.0  0
```

Dev split:

```text
condition       n   pass_rate  overall  coverage  followup  groundedness  recall@k  mrr  gate_violations
vague           60  0.000      0.512    0.500     0.000     0.750         1.0       1.0  0
missing_info    60  0.167      0.678    0.883     0.167     0.708         1.0       1.0  0
misleading      60  0.833      0.856    0.917     0.833     0.708         1.0       1.0  0
clean           60  0.933      0.882    0.917     0.933     0.708         1.0       1.0  0
typo_heavy      60  0.933      0.886    0.917     0.933     0.725         1.0       1.0  0
```

Test split:

```text
condition       n   pass_rate  overall  coverage  followup  groundedness  recall@k  mrr  gate_violations
vague           60  0.000      0.512    0.500     0.000     0.750         1.0       1.0  0
missing_info    60  0.050      0.633    0.817     0.050     0.738         1.0       1.0  0
misleading      60  0.800      0.850    0.900     0.800     0.738         1.0       1.0  0
clean           60  0.983      0.896    0.900     0.983     0.738         1.0       1.0  0
typo_heavy      60  0.983      0.896    0.900     0.983     0.738         1.0       1.0  0
```

Worst dev condition:

```text
condition: vague
overall: 0.512
pass_rate: 0.0
```

## Baseline diagnosis

Failure type counts:

```text
condition       failure_type                 count
vague           missing_expected_entity       300
vague           missing_followup_question     300
missing_info    missing_followup_question     265
missing_info    missing_expected_entity       103
misleading      missing_expected_entity        62
misleading      missing_followup_question      62
clean           missing_expected_entity        15
clean           missing_followup_question      15
typo_heavy      missing_expected_entity        15
typo_heavy      missing_followup_question      15
```

Oracle-retrieval attribution:

```text
generation_failure    657
```

Interpretation:

- Retrieval is not the bottleneck in this demo. `recall@k` and MRR are 1.0
  everywhere because the knowledge base has only two chunks and both are
  normally retrieved.
- The weak point is answer behavior. The naive assistant does not reliably ask
  the required follow-up questions when the question is vague or hides key
  fields.
- Vague questions are intentionally hard: no question fields are visible, so
  the expected behavior is to discuss uncertainty and ask about age, income,
  worker type, and EPFO.
- Missing-info questions fail mostly because worker type and/or EPFO are hidden,
  but the baseline still tends to answer as if it can decide.
- Clean and typo-heavy mostly pass, showing the simple char n-gram retriever is
  tolerant enough for this small demo.

## Local Qwen smoke results

The local model at `F:\cb_h_loop\models\qwen3_0.6` loaded successfully with the
Qwen transformers backend.

Single-question smoke answer:

```text
Based on the information, you are eligible for PMSYM ... You will also be
eligible for ESHRAM ... Next step: Confirm eligibility via official channels.
Evidence used: ['scheme_pmsym', 'scheme_eshram']
```

This confirms that the model path works, but it also shows over-confident
eligibility phrasing.

The 2-user local-Qwen audit evaluated only 10 cases, so it should be treated as
a smoke test, not a stable estimate:

```text
condition       n  pass_rate  overall  coverage  followup  groundedness  recall@k  mrr  gate_violations
missing_info    2  0.0        0.192    0.0       0.0       0.166         1.0       1.0  0
vague           2  0.0        0.387    0.5       0.0       0.250         1.0       1.0  0
typo_heavy      2  0.0        0.567    0.0       1.0       0.667         1.0       1.0  2
clean           2  0.0        0.567    0.0       1.0       0.667         1.0       1.0  2
misleading      2  0.5        0.700    0.5       1.0       0.500         1.0       1.0  0
```

Example failure:

```text
Question:
I am 20 years old. My monthly income is 17700. My work type is
unorganised_worker. My EPFO status is False. Which welfare schemes am I likely
eligible for?

Ground truth:
PMSYM is ineligible because income is above 15000. e-Shram is likely eligible.

Qwen answer:
It says the user is eligible for PMSYM and says income 17,700 meets the 15,000
threshold.

Evaluation:
false_confirmation:scheme_pmsym
```

Interpretation:

- The local Qwen backend runs, but this tiny smoke audit shows the model can
  make arithmetic/rule-following mistakes.
- The evaluator catches some of these as false confirmations.
- The groundedness score can still be imperfect as a hallucination detector,
  because the MVP groundedness layer is lexical: a sentence can reuse words from
  evidence while still applying a numeric rule incorrectly.

## What is working

- The domain-pack pattern is clean. A new domain can be added by replacing the
  six domain files without changing the engine.
- The assistant boundary is honest: `answer(question)` gets only the question.
- The split design is sound: train/dev/test assignment is deterministic and
  user-level.
- The retriever is adequate for the current tiny corpus and robust to injected
  typos.
- The audit outputs are useful: JSONL preserves nested records and CSV provides
  a quick human-readable flat view.
- Failure attribution correctly shows that the baseline problem is generation
  behavior, not retrieval.

## Main limitations found

- The requested conda environment has a PyArrow DLL access problem that breaks
  normal `sklearn` import. Fixing or reinstalling PyArrow in that environment
  would remove the need for the temporary run shim.
- The naive baseline is intentionally weak and should not be mistaken for a
  production assistant.
- The Qwen script defaults to `Qwen/Qwen3-0.6B`, which may try to use external
  model loading. In this workspace, the reliable path is the local model folder
  `F:\cb_h_loop\models\qwen3_0.6`.
- `scripts/run_audit_qwen.py` does not currently expose `model_name`, so using
  the local model path requires calling `run_audit()` directly or editing the
  script.
- The groundedness implementation is lexical and can miss numeric-rule errors.
  A stronger version would use claim extraction plus numeric/rule validation or
  an entailment model.
- The local Qwen smoke audit used only 2 users. It verifies the path works, but
  it is too small for a stable model-quality claim.

## Recommended next steps

1. Repair `pyarrow` in `E:\conda_envs\kgp_research` or remove it if not needed,
   then rerun `python scripts/run_audit.py` normally.
2. Add a `--model-name` argument to `scripts/run_audit_qwen.py` and
   `scripts/smoke_test_qwen.py` so the local model path can be selected without
   an inline Python command.
3. Improve the assistant prompt or answer template to explicitly compare
   numeric thresholds before confirming eligibility.
4. Strengthen evaluation for numeric constraints, especially "income <= 15000"
   and age ranges.
5. Run a larger Qwen audit after the environment is fixed, preferably with GPU
   acceleration and at least the default 20 users.

## Qwen GPU run: 500 cases

After the initial smoke runs, a 500-case Qwen audit was run on GPU.

Environment:

```text
torch 2.5.1+cu121
cuda_available True
cuda_device_count 1
device_name NVIDIA GeForce RTX 4060 Laptop GPU
```

Command:

```powershell
& 'E:\conda_envs\kgp_research\python.exe' scripts\run_qwen_local_progress.py --cases 500 --label qwen_500_cases --model-name F:\cb_h_loop\models\qwen3_0.6 --max-new-tokens 160 --device cuda --progress-every 10
```

Run size:

```text
100 synthetic users
500 total cases
5 conditions per user
elapsed time: about 25.7 minutes
data of record: data/outputs/audit_qwen_500_cases.jsonl
flat CSV: data/outputs/audit_qwen_500_cases_flat.csv
```

Overall result:

```text
records: 500
passed: 51
overall pass_rate: 0.102
```

All-splits summary:

```text
condition       n    pass_rate  overall  coverage  followup  groundedness  recall@k  mrr  gate_violations
missing_info    100  0.08       0.325    0.300     0.12      0.189         1.0       1.0  18
vague           100  0.00       0.387    0.500     0.00      0.250         1.0       1.0  0
clean           100  0.09       0.672    0.350     0.95      0.648         1.0       1.0  65
typo_heavy      100  0.03       0.678    0.400     0.95      0.607         1.0       1.0  71
misleading      100  0.31       0.686    0.665     0.80      0.420         1.0       1.0  25
```

Pass rate by condition:

```text
clean        9/100   0.09
misleading   31/100  0.31
missing_info 8/100   0.08
typo_heavy   3/100   0.03
vague        0/100   0.00
```

Most common failure types:

```text
vague        missing_expected_entity       100
vague        missing_followup_question     100
vague        weak_groundedness             100
missing_info missing_followup_question      88
missing_info missing_expected_entity        85
missing_info weak_groundedness              80
clean        missing_expected_entity        75
typo_heavy   false_confirmation:scheme_pmsym 70
typo_heavy   missing_expected_entity        70
clean        false_confirmation:scheme_pmsym 65
```

Gate counts:

```text
false_confirmation:scheme_pmsym   166
false_confirmation:scheme_eshram   13
```

Interpretation:

- GPU execution works and the local Qwen model runs end to end through the full
  SYNALIGN pipeline.
- Retrieval is still perfect in this small two-chunk corpus: `recall@k = 1.0`
  and `mrr = 1.0` for every condition.
- The dominant issue is generation, not retrieval. Qwen often confirms PMSYM
  eligibility even when visible facts disqualify the user.
- The clearest repeated bug is numeric/rule failure. Example: for income
  `17700`, Qwen says the monthly income is `15,000 or below`.
- Vague questions fail because the model does not ask the required follow-up
  questions about age, income, worker type, and EPFO.
- The `misleading` condition has the best pass rate in this run, but it still
  has gate failures and weak groundedness.

Recommended next fix after this run:

1. Add deterministic eligibility calculation before the LLM answer, or force the
   model prompt to list visible facts and compare each threshold explicitly.
2. Add evaluator checks for numeric contradictions, not only lexical
   groundedness.
3. Re-run the same 500-case GPU audit with a new label after the prompt or
   rule-checking change.
