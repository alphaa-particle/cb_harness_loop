# Final Model Performance Comparison

This report compares the three local model runs on the same 500-case SYNALIGN
welfare-advice audit:

```text
Qwen3 0.6B
Qwen3.5 0.8B
Gemma 1B
```

All numbers below come from the saved JSONL audit outputs in `data/outputs/`.

## Short Answer

```text
Best overall pass rate:      Qwen3.5 0.8B
Fewest safety gates:         Gemma 1B
Best scheme coverage:        Qwen3.5 0.8B
Best follow-up score:        Gemma 1B
Best groundedness score:     Qwen3 0.6B
Worst overall model:         Qwen3 0.6B
```

The practical conclusion is:

```text
Qwen3.5 0.8B is the strongest model by pass rate and coverage, but it makes
many unsafe eligibility confirmations.

Gemma 1B is safer by gate count, but it often gives incomplete or weakly
grounded answers.

Qwen3 0.6B is the weakest overall. It retrieves the right documents but fails
often during answer generation.
```

None of the three models is safe enough as a standalone welfare eligibility
assistant.

## Fairness Check

All three models were tested on the same:

```text
500 cases
100 synthetic users
5 question types per user
same train/dev/test split
same visible user facts
same hidden profiles
same ground truth
same retriever
same evaluator
```

The three runs all had perfect retrieval:

```text
retrieval_recall = 1.000
retrieval_mrr    = 1.000
```

So the main differences are from answer generation, not document retrieval.

## Overall Results

| Model | Passed | Failed | Pass Rate | Overall | Coverage | Follow-Up | Groundedness | Actionability | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B | 51 / 500 | 449 | 10.2% | 0.550 | 0.443 | 0.564 | 0.423 | 0.986 | 179 |
| Qwen3.5 0.8B | 136 / 500 | 364 | 27.2% | 0.735 | 0.975 | 0.571 | 0.406 | 0.998 | 305 |
| Gemma 1B | 111 / 500 | 389 | 22.2% | 0.530 | 0.417 | 0.655 | 0.284 | 1.000 | 4 |

## Ranking by Main Metric

### Pass Rate

```text
1. Qwen3.5 0.8B: 27.2%
2. Gemma 1B:     22.2%
3. Qwen3 0.6B:   10.2%
```

Qwen3.5 won by total pass rate. Gemma was second. Qwen3 0.6B was far behind.

### Safety Gates

```text
1. Gemma 1B:        4 gate violations
2. Qwen3 0.6B:    179 gate violations
3. Qwen3.5 0.8B:  305 gate violations
```

Gemma was clearly best on hard safety gates. Qwen3.5 had the highest pass rate,
but also the most unsafe confirmations.

### Coverage

```text
1. Qwen3.5 0.8B: 0.975
2. Qwen3 0.6B:   0.443
3. Gemma 1B:     0.417
```

Coverage means the answer mentioned the schemes it was supposed to discuss.
Qwen3.5 was much better here. Gemma often missed expected schemes.

### Follow-Up Behavior

```text
1. Gemma 1B:     0.655
2. Qwen3.5 0.8B: 0.571
3. Qwen3 0.6B:   0.564
```

Gemma was best at asking follow-up questions, but still not good enough. In
vague and missing-information cases, it often asked only part of what was
needed.

### Groundedness

```text
1. Qwen3 0.6B:   0.423
2. Qwen3.5 0.8B: 0.406
3. Gemma 1B:     0.284
```

Groundedness is a simple lexical support score. Qwen3 0.6B was highest here,
but that did not make it the best model overall because it still had very low
pass rate and many false confirmations.

## Results by Question Type

### Clean Questions

Clean questions give the important facts clearly.

| Model | Passed | Pass Rate | Overall | Coverage | Follow-Up | Groundedness | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B | 9 / 100 | 9.0% | 0.672 | 0.350 | 0.950 | 0.648 | 65 |
| Qwen3.5 0.8B | 28 / 100 | 28.0% | 0.833 | 0.990 | 0.960 | 0.388 | 72 |
| Gemma 1B | 37 / 100 | 37.0% | 0.637 | 0.445 | 0.970 | 0.354 | 1 |

Simple reading:

```text
Gemma passed the most clean cases and had almost no safety gates.
Qwen3.5 had much stronger coverage but many unsafe confirmations.
Qwen3 0.6B performed poorly and had many PMSYM false confirmations.
```

### Vague Questions

Vague questions hide the important facts.

| Model | Passed | Pass Rate | Overall | Coverage | Follow-Up | Groundedness | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B | 0 / 100 | 0.0% | 0.387 | 0.500 | 0.000 | 0.250 | 0 |
| Qwen3.5 0.8B | 0 / 100 | 0.0% | 0.607 | 1.000 | 0.000 | 0.429 | 100 |
| Gemma 1B | 0 / 100 | 0.0% | 0.387 | 0.500 | 0.250 | 0.000 | 0 |

Simple reading:

```text
All three models failed every vague case.
Qwen3.5 talked about the right schemes, but made many unsafe confirmations.
Gemma asked some follow-up questions, but not enough.
Qwen3 0.6B did not ask the required follow-ups.
```

### Missing-Info Questions

Missing-info questions give some facts but hide one or more required facts.

| Model | Passed | Pass Rate | Overall | Coverage | Follow-Up | Groundedness | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B | 8 / 100 | 8.0% | 0.325 | 0.300 | 0.120 | 0.189 | 18 |
| Qwen3.5 0.8B | 4 / 100 | 4.0% | 0.631 | 1.000 | 0.145 | 0.380 | 63 |
| Gemma 1B | 12 / 100 | 12.0% | 0.400 | 0.315 | 0.195 | 0.366 | 0 |

Simple reading:

```text
Gemma passed the most missing-info cases and had no gates.
Qwen3.5 covered the schemes well but made many unsafe confirmations.
All three models were weak at asking the exact missing question.
```

### Typo-Heavy Questions

Typo-heavy questions contain typing mistakes.

| Model | Passed | Pass Rate | Overall | Coverage | Follow-Up | Groundedness | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B | 3 / 100 | 3.0% | 0.678 | 0.400 | 0.950 | 0.607 | 71 |
| Qwen3.5 0.8B | 36 / 100 | 36.0% | 0.808 | 0.925 | 0.950 | 0.391 | 59 |
| Gemma 1B | 32 / 100 | 32.0% | 0.616 | 0.400 | 0.970 | 0.333 | 1 |

Simple reading:

```text
Qwen3.5 was best on typo-heavy questions by pass rate.
Gemma was close behind and had far fewer safety gates.
Qwen3 0.6B handled retrieval but failed answer generation badly.
```

### Misleading Questions

Misleading questions pressure the model to confirm a possibly wrong claim.

| Model | Passed | Pass Rate | Overall | Coverage | Follow-Up | Groundedness | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B | 31 / 100 | 31.0% | 0.686 | 0.665 | 0.800 | 0.420 | 25 |
| Qwen3.5 0.8B | 68 / 100 | 68.0% | 0.796 | 0.960 | 0.800 | 0.442 | 11 |
| Gemma 1B | 30 / 100 | 30.0% | 0.613 | 0.425 | 0.890 | 0.365 | 2 |

Simple reading:

```text
Qwen3.5 was clearly best at misleading questions.
Gemma had fewer safety gates but much lower coverage.
Qwen3 0.6B was similar to Gemma by pass rate but had more unsafe gates.
```

## Results by Split

| Model | Train Passed | Train Rate | Dev Passed | Dev Rate | Test Passed | Test Rate |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B | 31 / 300 | 10.3% | 11 / 100 | 11.0% | 9 / 100 | 9.0% |
| Qwen3.5 0.8B | 85 / 300 | 28.3% | 21 / 100 | 21.0% | 30 / 100 | 30.0% |
| Gemma 1B | 51 / 300 | 17.0% | 26 / 100 | 26.0% | 34 / 100 | 34.0% |

Interesting point:

```text
Gemma had the best test split pass rate at 34.0%, even though Qwen3.5 had the
best overall pass rate across all 500 cases.
```

Because each test split has only 100 cases, this should be treated as a useful
signal, not as final proof that Gemma generalizes better.

## Failure Types

| Failure Type | Qwen3 0.6B | Qwen3.5 0.8B | Gemma 1B |
|---|---:|---:|---:|
| weak_groundedness | 272 | 426 | 467 |
| missing_expected_entity | 369 | 16 | 378 |
| missing_followup_question | 218 | 217 | 205 |
| false_confirmation:scheme_pmsym | 166 | 129 | 4 |
| false_confirmation:scheme_eshram | 13 | 175 | 0 |
| forbidden_claim:automatic_cash_benefit | 0 | 1 | 0 |
| weak_next_action | 7 | 1 | 0 |

Simple reading:

```text
Qwen3.5 almost always mentioned the expected schemes, but often confirmed
eligibility too aggressively.

Gemma rarely triggered hard false-confirmation gates, but often missed expected
schemes and had very weak groundedness.

Qwen3 0.6B had both coverage problems and many PMSYM false confirmations.
```

## Model-by-Model Summary

### Qwen3 0.6B

Good parts:

- Retrieval was perfect.
- Groundedness score was the highest of the three models.
- It gave useful next steps in most answers.

Bad parts:

- Lowest pass rate: 10.2%.
- Missed expected schemes 369 times.
- Made 179 safety-gate violations.
- Often falsely confirmed PMSYM eligibility.
- Failed all vague questions.

Plain-English verdict:

```text
Qwen3 0.6B can read the retrieved documents, but it does not reliably apply the
rules or handle missing information. It is the weakest of the three models.
```

### Qwen3.5 0.8B

Good parts:

- Highest overall pass rate: 27.2%.
- Best overall score: 0.735.
- Best coverage score: 0.975.
- Best misleading-question performance: 68%.
- Strongest at mentioning the schemes it should discuss.

Bad parts:

- Highest safety-gate count: 305.
- Falsely confirmed e-Shram 175 times.
- Falsely confirmed PMSYM 129 times.
- Failed all vague questions.
- Still did not ask enough follow-up questions.

Plain-English verdict:

```text
Qwen3.5 0.8B is the strongest general answerer, but it is too confident. It
often talks about the right schemes, then makes unsafe eligibility claims.
```

### Gemma 1B

Good parts:

- Second-best overall pass rate: 22.2%.
- Best safety-gate performance: only 4 gates.
- Best follow-up score: 0.655.
- Best clean-question pass rate: 37%.
- Best missing-info pass rate: 12%.
- Best test split pass rate: 34%.

Bad parts:

- Weakest groundedness score: 0.284.
- Missed expected schemes 378 times.
- Coverage was low: 0.417.
- Failed all vague questions.
- Often gave incomplete answers.

Plain-English verdict:

```text
Gemma 1B is less reckless than Qwen3.5, but it is often incomplete. It avoids
many hard safety failures, but it misses too much information to be trusted.
```

## Final Recommendation

For the current SYNALIGN welfare task:

```text
Best model to improve further: Qwen3.5 0.8B
Best safety behavior to study: Gemma 1B
Weakest model: Qwen3 0.6B
```

Qwen3.5 is the best starting point if the goal is to maximize correct,
complete answers. But it needs strong guardrails because it confirms
eligibility too often.

Gemma is worth studying because it has far fewer hard safety gates. But it
needs better answer structure and scheme coverage.

The next system should not rely on any model alone. The best path is:

```text
1. Use deterministic rule checks for eligibility.
2. Force every answer to cover every relevant scheme.
3. Require exact follow-up questions when information is missing.
4. Let the model explain the checked result, not invent the result.
5. Re-run the same 500-case audit after every change.
```

That would combine the best parts of the runs:

```text
Qwen3.5's coverage
Gemma's lower safety-gate behavior
deterministic rule checks for numeric eligibility
```

Only then would the chatbot be moving toward reliable welfare advice.
