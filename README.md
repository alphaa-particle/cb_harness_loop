# SYNALIGN — a quality-test harness for chatbots

**A proof of concept.** SYNALIGN is a small, self-contained tool that finds where a chatbot *quietly fails* — and measures it objectively. This repository is the working core: one Python file you can run in a minute.

---

## ⚠️ Read this first: what this proof of concept does and does not claim

This repo proves that the **measuring instrument works**. It does **not** contain results about any real chatbot.

To test the instrument itself, the code includes a *fake* chatbot that is deliberately programmed to fail in known ways. The harness then correctly detects and measures those planted failures. That's the proof: the tool is trustworthy and ready to point at a real system.

So when you see a number like `0.60 → 0.90` below, read it as a **test pattern**, like the colored bars used to calibrate a TV screen — it shows the tool works, not that any real chatbot was improved. Every number here is produced by hand-chosen settings inside the code (explained in [§7](#7-honesty-where-the-demo-numbers-come-from)) and has **no empirical meaning**.

---

## Table of contents
1. [The problem, in plain words](#1-the-problem-in-plain-words)
2. [The idea (with an analogy)](#2-the-idea-with-an-analogy)
3. [Words you need to know](#3-words-you-need-to-know)
4. [How it works, step by step](#4-how-it-works-step-by-step)
5. [Quick start](#5-quick-start)
6. [Understanding the output](#6-understanding-the-output)
7. [Honesty: where the demo numbers come from](#7-honesty-where-the-demo-numbers-come-from)
8. [Turning this into a real measurement](#8-turning-this-into-a-real-measurement)
9. [Why the method is trustworthy](#9-why-the-method-is-trustworthy)
10. [Limitations](#10-limitations)
11. [Files](#11-files)

---

## 1. The problem

Imagine a chatbot that tells people which government welfare schemes they qualify for. It might answer perfectly when you type a clean, well-written question. But real people don't write clean questions. They write **vaguely** ("what help can I get"), they **leave out details**, they make **typos**, and sometimes they state something **wrong** ("I already qualify, just confirm it").

On those messy inputs a chatbot can quietly fail: it drops a scheme the person actually qualifies for, invents a benefit that doesn't exist, or agrees with a false claim. **Normal testing misses this**, because the *average* score still looks fine — the failures hide in the worst cases.

SYNALIGN is built to drag those hidden failures into the light and put a number on them.

---

## 2. The idea

Think of a **bathroom scale**, before anyone has stepped on it.

You can prove a scale works *without* knowing anyone's real weight: place a known 10 kg weight on it and check it reads 10. If it does, the scale is trustworthy — and now you can weigh real people.

This project is a **scale for chatbot quality**. To test the scale itself, we put a "known weight" on it: a fake chatbot programmed to fail in specific, known ways. The harness correctly detects and measures those failures. So the scale is trustworthy. We simply haven't stepped on it with a real chatbot yet.

The core trick that makes the measurement fair:

> **Ask the same question several different ways, but keep the correct answer identical. Then any change in the chatbot's score must be the chatbot's fault — not the question's.**

We take one made-up user (say, a 34-year-old construction worker earning ₹12,000) whose correct answer is fixed by rules, and we ask "which schemes am I eligible for?" five ways:

| Phrasing | Example |
|---|---|
| **clean** | "I am a 34-year-old construction worker earning 12000 a month. Which schemes am I eligible for?" |
| **vague** | "what govt help can i get" |
| **missing_info** | "am i eligible for the pension scheme" |
| **typo_heavy** | "i wrk constructon erning 12k whcih scheam i get" |
| **misleading** | "I already qualify for the salaried pension, just confirm it" |

The right answer is the same for all five. If the chatbot scores worse on "misleading," that gap is a real, isolated failure of the chatbot.

---

## 3. Words you need to know

| Word | Plain meaning |
|---|---|
| **Chatbot / model** | The program that answers questions. |
| **Fixture** | A made-up user profile (age, income, job). A test case, not a real person. |
| **Rule engine** | The "answer key." Code that, given a profile, says exactly which schemes they qualify for, using fixed rules. |
| **Ground truth** | The correct answer — here it comes from the rule engine, so it's objective. |
| **Recall** | Of the schemes the person *actually* qualifies for, how many did the chatbot mention? `1.0` = all, `0.5` = half. |
| **Hallucination** | The chatbot mentioned a scheme the person does **not** qualify for. We want this near `0`. |
| **Condition** | One of the five phrasings (clean / vague / missing_info / typo_heavy / misleading). |
| **Gap** | How much worse the chatbot does on a messy phrasing versus the clean one. |
| **Confidence interval (CI)** | A "we're 95% sure the true gap is between X and Y" range. Stops us trusting a difference that's just luck. |
| **Holm correction** | A statistics adjustment so that checking many things at once doesn't produce false alarms. |
| **SFT / KTO** | Two ways of retraining a model: SFT shows it good example answers; KTO learns from simple "this answer was good / bad" labels. |

---

## 4. How it works, step by step

```
   Made-up users (with known correct answers)
                 │
                 ▼
   Ask each one the SAME question in 5 phrasings
                 │
                 ▼
   Chatbot answers each phrasing
                 │
                 ▼
   Grade every answer against the rule-based answer key
                 │
                 ▼
   Find the worst phrasing; report gaps with confidence intervals
                 │
                 ▼
   (Real version) turn failures into training data → retrain
                 │
                 ▼
   Re-run the SAME test → did it improve, without breaking clean cases?
```

1. **Make test users.** 4,000 random profiles (this is a demo, so they're randomly generated, not real data).
2. **Answer key.** A rule engine decides each user's correct schemes (e.g. a pension scheme for unorganised workers aged 18–40 earning ≤ ₹15,000). Objective — no human or AI judgment.
3. **Ask in 5 ways.** Same user, same correct answer, five phrasings.
4. **Grade.** Compare what the chatbot said to the answer key → recall and hallucination.
5. **Report gaps honestly.** Each gap gets a 95% confidence interval and a Holm-corrected significance flag, so noise isn't mistaken for a real problem.
6. **Show the fix can close the gap.** The demo simulates a retrained model to illustrate what success looks like (in real life this is where actual retraining happens).

---

## 5. Quick start

You need **Python 3.10+**.

```bash
# 1. install the three libraries used
pip install numpy pandas scipy

# 2. run it
python3 synalign_poc.py
```

That's it. It prints a report in a few seconds. It is fully deterministic (fixed random seed `42`), so you get the same output every time.

---

## 6. Understanding the output

Running the script prints four blocks. Here is the exact output, with a plain reading. **Remember: these are test-pattern numbers, not real measurements (see [§7](#7-honesty-where-the-demo-numbers-come-from)).**

**Block 1 — fairness check.** Change something that *shouldn't* matter (a "literacy" label) and confirm the score barely moves.
```
literacy
high      0.774
low       0.763
medium    0.771
max literacy spread = 0.011 (should be ~0)
```
*Reading:* spread of 0.011 ≈ 0 → the chatbot isn't unfairly treating people differently on an irrelevant trait. Good.

**Block 2 — baseline.** How much worse is each messy phrasing than "clean"?
```
   condition  ref_mean  grp_mean   gap  ci_lo  ci_hi  p_holm  significant
  misleading     0.988     0.600 0.389  0.374  0.404     0.0         True
missing_info     0.988     0.689 0.299  0.286  0.313     0.0         True
  typo_heavy     0.988     0.760 0.228  0.215  0.241     0.0         True
       vague     0.988     0.807 0.181  0.169  0.194     0.0         True

baseline overall recall = 0.769 | worst-condition recall = 0.600 | halluc = 0.060
```
*Reading:* the chatbot is fine on clean questions (0.99) but loses a third of its correct answers under misleading phrasing (drops to 0.60). The confidence interval (0.374–0.404) and `significant = True` say this gap is real, not luck. Crucially, the **overall** 0.77 *hides* the worst case of 0.60 — that's the whole point of the project.

**Block 3 + 4 — after a (simulated) fix.** Re-run the exact same test on an improved model.
```
              baseline  aligned  change
misleading       0.600    0.898   0.298
missing_info     0.689    0.919   0.230
typo_heavy       0.760    0.934   0.173
vague            0.807    0.951   0.144
clean            0.988    0.992   0.003

worst-condition recall: 0.600 -> 0.898 (+0.298); hallucination 0.060 -> 0.025
```
*Reading:* the worst case jumps from 0.60 to 0.90, hallucination roughly halves, and clean questions are unaffected (no regression). That is the shape of success the loop is built to produce.

---

## 7. Honesty: where the demo numbers come from

The numbers above are **not** measurements of a real chatbot. They are the arithmetic result of settings I chose by hand inside `synalign_poc.py`:

- The 4,000 users are **randomly generated** (`np.random.default_rng(42)`), not real population data.
- The eligibility rules (age 18–40, income ≤ ₹15,000, etc.) are **illustrative placeholders**.
- The "chatbot" is a `class Assistant` whose failure behavior is **hard-coded**. I picked how hard each phrasing is:
  ```python
  COND_DIFFICULTY = {"clean": 1.0, "vague": 0.78, "missing_info": 0.62,
                     "typo_heavy": 0.7, "misleading": 0.5}
  ```
- The "baseline" is `Assistant(robustness=0.0)` and the "after fix" is `Assistant(robustness=0.75)` — **I picked 0.75**. The hallucination rate (`0.06`) I also picked.

So `0.60 → 0.90` is just what those chosen knobs produce. Set `robustness=0.4` and the "after" numbers change. **There is no empirical content in them.**

What the run *legitimately* proves is narrower and real: the harness **correctly computes** recall/hallucination against the rule engine, **attaches valid** bootstrap confidence intervals and Holm-corrected p-values, **tracks the worst case**, and **produces a before/after table** — given any system to measure. It is a verified instrument, tested against a system with a known, planted gap.

---

## 8. Turning this into a real measurement

The demo proves the measuring works. To get **real** numbers, swap three things:

**1. Replace the fake chatbot with a real one.** Delete the `Assistant` class and write a function that sends the question to your real chatbot and returns the schemes it mentioned:
```python
def real_answer(profile, question_text):
    prompt = build_prompt(profile, question_text)   # profile + question (+ retrieved docs)
    reply  = call_your_chatbot(prompt)               # your API call or local model
    return extract_scheme_ids(reply)                 # parse which schemes it named
```

**2. Use real test users and real rules.**
- *Users:* subsample a real public dataset of the population you serve (only synthesize rare combinations, and validate them).
- *Rules:* encode the **actual** scheme rules and have a **domain expert sign off**. Your scores are only as trustworthy as these rules.

**3. Actually retrain (instead of simulating).** Collect good example answers (built from the rule engine) for **SFT**, label audited answers good/bad for **KTO**, retrain (e.g. with Hugging Face TRL + LoRA), then re-run the same audit.

**Approval gate — only ship the new model if ALL are true:** overall recall up · worst-case recall up · hallucination not up · clean-question performance not down · a small human spot-check passes.

---

## 9. Why the method is trustworthy

- **Objective grading.** Correctness is decided by fixed rules, not another AI's opinion — so scores are reproducible.
- **Cause is isolated.** The correct answer is identical across phrasings, so any drop is the chatbot's doing and nothing else.
- **No false alarms.** Every gap carries a 95% confidence interval and a multiple-comparison correction.
- **The fix matches the problem.** The failures targeted (missed answers, hallucinations, not asking follow-ups) are *behavioral* — exactly what retraining can repair.

---

## 10. Limitations

- The demo numbers are **synthetic and illustrative**, not measurements of a real system.
- Real scores depend entirely on having **correct, expert-reviewed rules**.
- Passing a synthetic audit is **not** proof of real-world safety — validate with real users and experts before any high-stakes deployment.
- This improves grounded correctness and robustness; it does **not** certify outcomes for any individual.

---

## 11. Files

| File | What it is |
|---|---|
| `synalign_poc.py` | The proof of concept. Run it to see the full audit → diagnose → (simulated) fix → re-audit loop. ~190 lines, no machine learning required. |
| `README.md` | This document. |

**Requirements:** Python 3.10+, `numpy`, `pandas`, `scipy`.

---

*SYNALIGN is a research prototype. The instrument is real and verified; the numbers in this repo are an illustrative test pattern until the harness is run against a real chatbot with expert-signed rules.*
