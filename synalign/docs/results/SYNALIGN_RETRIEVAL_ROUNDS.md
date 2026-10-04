# How SynAlign found and fixed search weaknesses

This is the record of one improvement cycle on the search step of the Scheme Helper. It shows what SynAlign measured and found, what was tried, what was kept and what was thrown away, including every attempt that failed.

**Goal:** get the right scheme text in front of the answering AI more often, for every kind of question the chatbot is meant for, without making any answer worse.

**Outcome:** one change is kept: round 4, which brings in a scheme the question names when the prompt lacks it.

- **On the use-case exam questions,** the needed text now reaches the AI for **82.0% instead of 74.6%** (54 schemes) and **72.4% instead of 67.5%** (4,600 schemes). No exam question broke.
- **Undone because they hurt answers end to end:** four other changes improved search, but the small answering AI got some answers wrong when its prompt changed.
- **Undone at close-out:** a fifth change (round 5b) passed the practice checks but broke exam answers at 4,600 schemes.

---

## Ground rules

These were fixed before any change was made.

- **Practice questions only.** Every decision used the practice (dev) questions.
  - The *use-case* exam questions were new, and nobody looked at them until the end.
  - The *original* exam questions had already been run and reported before this cycle (the first results in the top-level README). No decision in this cycle used them.
- **Nothing tuned to individual questions.** The code has no scheme names, no question wording and no per-question exceptions. Every change is a general mechanism that would make sense for any catalogue of schemes.
- **One change per round,** checked twice: once on search, once end to end on the answers. Each check runs at two sizes: our 54 schemes alone, and the same 54 hidden among about 4,600 real background schemes.
- **Every round is logged, including the undone ones.**
  - Each round's record in [search_rounds/](../../data/evaluation/india_schemes/search_rounds/) says what happened at its answer check.
  - The design and diagnosis data are in [design/](../../data/evaluation/india_schemes/design/).

---

## 1. What SynAlign measured first

### The original question set

The original question set's practice part has 316 questions: facts, typos, situations, yes/no eligibility, and "not found". Of the 304 that have an answer, the right section reached the prompt for:

- **282 at 54 schemes**;
- **261 at 4,600 schemes**.

The 22 misses at 54 schemes, counted two ways (data: `design/diagnosis_r0_dev_54.jsonl`):

| By what the prompt held | Questions | By how search ranked schemes | Questions |
|---|---:|---|---:|
| the asked scheme was not in the prompt at all | 14 | another scheme ranked first (the right one 2nd in 14, 3rd in 3, lower in 1) | 18 |
| the asked scheme was in the prompt, but not the section holding the answer | 7 | the right scheme first, but the answering section missing | 3 |
| nothing was retrieved | 1 | nothing retrieved | 1 |

The columns differ because a scheme ranked second often still gets a slot.

### A gap in the testing itself: the uses people actually have

The original questions only test single facts, situations and yes/no eligibility. People also ask:

- how to apply;
- which documents they need;
- what a scheme gives;
- who is excluded;
- "can I join, and how do I apply";
- how two schemes differ;
- which schemes fit a need.

None of that was measured, so a second question set was built: **544 use-case questions** (178 practice, 366 exam), written by [build_eval_questions.py](../../scripts/build_eval_questions.py) `--usecases`.

- **How they are written:** a few fixed English phrasings per kind of question, applied to every scheme.
- **What counts as correct:** each question names the sections its answer needs. A comparison needs text from *both* schemes.
- **The practice/exam split:** by scheme, as before. For "which schemes" questions, every third category alphabetically is practice, because the hash split had put all 12 categories into the exam.

Limits that come from how these questions are built:

- **They name the scheme exactly as the catalogue does.** So finding the scheme is easy, and "who is excluded" and comparison questions really test only whether that scheme's rules or text got in. Questions using informal names would be harder.
- **The 8 practice "which schemes" questions count exam schemes as correct answers,** because their categories mix both splits.
- **The need descriptions were written knowing which schemes are in each category.**

On the use-case practice questions, search ranked the right scheme first **99%** of the time. Yet the needed sections reached the prompt for only **146 of 178** at 54 schemes and **137 of 178** at 4,600. So the main weakness for real uses was which sections of the right scheme get in:

| Kind of use-case question | 54 schemes | 4,600 schemes |
|---|---:|---:|
| Compare two schemes | 3 / 10 | 3 / 10 |
| Benefits | 23 / 36 | 18 / 36 |
| Documents | 27 / 34 | 30 / 34 |
| Eligibility and how to apply | 16 / 18 | 15 / 18 |
| How to apply | 33 / 36 | 33 / 36 |
| Who is excluded | 36 / 36 | 36 / 36 |
| Which schemes (8 questions) | 8 / 8 | 2 / 8 |

---

## 2. Grading had to be fixed first, twice

The answer checks depend on the grading, and the grading had to be corrected twice.

**First correction.**

- **The clue:** the first use-case answer baseline came back at 94.9% correct, though search had put the needed text in the prompt for only 82% of the questions.
- **The flaw:** the rule counted words that are rare in the catalogue. But a scheme's own words ("handloom", "weaver") are rare in the catalogue and appear in every section of that scheme. An answer drawn from the wrong section still passed.
- **The fix:** the rule now counts only words of the needed section that no other section of the same scheme uses.
- **The effect,** on the 160 single-scheme questions: answers whose needed section was *not* in the prompt went from 20 of 25 marked correct to 9 of 25.

**Second correction,** after an independent review of the code:

| Problem | Fix |
|---|---|
| A comparison passed when the answer just repeated the question (it names both schemes) | each scheme must get at least one word of its own text that the other scheme's text lacks; words from the question or the names never count |
| Section answers could pass on words from the question or the scheme's name | those words never count |
| Fact answers passed if any number from any accepted form appeared ("within 3 weeks" for "72 hours"; a wrong toll-free number sharing "1800"; "Poshan 2.0" read as 2) | an accepted form must match whole, with every number and the same time unit; numbers inside scheme names are ignored; forms the question itself contains don't count; an answer that opens by refusing doesn't count |
| Decimal values ("1.5%") could never be graded correct | now matched as text |

Every saved answer run was graded again with the final rules. Every number in this document and in the top-level README uses them. Re-running every round's answer check under the final rules changed no verdict.

**Still weak:** a comparison answer can pass while inventing facts about one scheme. The grader checks that each scheme gets something specific, not that it is true.

---

## 3. First design round: ideas tried and rejected

Four independent designers each prototyped a general fix, measuring only on practice questions. A fifth reviewer re-ran every number and checked the code (records: `design/design1_*`).

| Idea | What happened | Decision |
|---|---|---|
| Let meaning search override weak word matches when its lead is large | +6 at 54 schemes; broke a question at 4,600; the gain held only in a narrow band of settings | rejected: likely fitted to these questions |
| Rank whole schemes first, then fill the slots | +6 / +3; chosen from about 35 variants; broke 2 questions outside the practice set | rejected: too many variants tried on the same questions |
| Change the "not found" cut-off | 22 rules tried; any gain came from setting the cut-off just above one question | no change |
| Add a Latin spelling of Hindi-script words to the letter search | +5 / +2, no breaks | set aside: it is language coverage, which the owner asked not to pursue |

---

## 4. Second diagnosis: why the right section was left out

Reading the use-case misses showed three general causes:

1. **Long rules take two slots.** In 17 schemes the rules were cut into two parts, and "rules first" put both in.
2. **The overview crowds out the specific section.** The overview mentions the scheme most, so it outranks "How to apply" or "Benefits".
3. **A question that names schemes loses slots to other schemes.** A comparison would get only one of the two schemes.

The first prototype of name matching broke 13 practice questions at 4,600 schemes, in two ways:

- **Generic names.** "Disability Pension" is the name of five state schemes.
- **A short name inside a longer one.** "PM Kisan" sits inside "PM Kisan Maandhan".

Two guards fixed both, and the prototype then broke none (`design/prototype_evidence_assembly_guards.json`). The guards were designed after seeing which practice questions broke, so the exam is their independent check.

---

## 5. The rule every round had to pass

The rule is written in [search_round.py](../../scripts/search_round.py), with its two later revisions recorded there and in the rounds below.

**Search check** (both question sets, both sizes). A change is kept only if:

| Guard | Requirement |
|---|---|
| A1 gain | at least one more question gets the right text, net, at each size (both question sets together) |
| A2 breaks | at most one question that worked before stops working, at each size |
| A3 slices | no question type and no language, in either question set, does worse |
| A4 refusals | "not found" questions are refused no less often (judged on the 54 schemes only), and no more answerable questions get nothing |
| A5 scheme | no fewer questions get their scheme into the prompt |
| A6 interval | the 95% interval of the gain does not reach below zero |
| A7 budget | no worse with 2 or 5 sections in the prompt instead of 3 |
| A8 other wordings | on 206 differently worded questions (from the fine-tuning examples), no new miss and no new refusal |
| A9 speed | search no more than twice as slow (the median of 5 alternating passes) |

**Answer check** (four runs: both question sets at both sizes), using [compare_runs.py](../../scripts/compare_runs.py) `--same-model`:

- **B1:** a question whose prompt did not change must get exactly the same answer; otherwise the runs are invalid. This held in every run of the cycle.
- **B2:** no question type or language gets fewer correct answers.
- **B3:** no more wrong "yes" answers.
- **B4:** no "not found" question that was refused is now answered.

---

## 6. The rounds

All answer numbers use the final grading.

### Round 1: count a scheme's rules as one slot

**Search:**

- **Attempt 1: UNDO.** It failed only the speed guard: 0.30 → 1.40 ms at 54 schemes, measured in a single pass while the answer model was busy. Seven repeated timings read 0.4–0.6 → 0.5–0.7 ms. The timing was changed to the median of 5 alternating passes; this is revision 1 of the rule.
- **Attempt 2: KEEP.** The right text reached the prompt for 428 → 438 of 482 questions at 54 schemes, and 398 → 408 at 4,600, with nothing broken. Benefits gained 5, documents 2, and eligibility-and-apply 2.

**Answers: UNDO.**

- The original practice questions at 54 schemes went 260 → 258 correct (2 fixed, 4 broken). Facts, typos, English and Hindi-in-Latin-letters each lost an answer.
- The change altered 69 prompts. All four lost answers were wrong in substance, and had been graded right only because the expected number happened to appear ("the pension starts at age 18…" went on to mention 60).
- The rule counts them all the same, and it was not relaxed.

**Lesson:** a change that alters many prompts moves many of the small model's answers. It is only worth it where it fixes something.

### Rounds 2 and 3, first versions: superseded

Both were measured on top of round 1 and passed their search checks. When round 1 was undone, they were measured again as 2b and 3b.

### Round 2b: schemes the question names get all the slots

**Search: KEEP.** The right text reached the prompt for 428 → 440 questions at 54 schemes, and 398 → 415 at 4,600. Comparisons with both schemes in the prompt went from 3 to 10 of 10 at both sizes. Nothing broke.

**Answers: UNDO.**

| Practice set | Correct | Fixed / broken | Wrong "yes" | Verdict |
|---|---|---|---|---|
| Original, 54 schemes | 260 → 258 | 1 / 3 | 4 → 5 | UNDO |
| Use cases, 54 schemes | 151 → 155 | 4 / 0 | 0 → 0 | KEEP on its own |
| Original, 4,600 schemes | 237 → 235 | 3 / 5 | 4 → 6 | UNDO |
| Use cases, 4,600 schemes | 142 → 148 | 6 / 0 | 0 → 0 | KEEP on its own |

Where the original questions lost answers:

- **At 54 schemes,** two of the three breaks are the same lucky "starts at age 18" answers as in round 1.
- **At 4,600 schemes,** three of the five breaks are questions that *now* had the right scheme in the prompt, and the model still answered wrongly:
  - two Hindi widow-pension questions got the disability pension's sections first;
  - a Hindi scholarship question got the right rules, and the model compared the income wrongly and said "yes".

### Round 3b: for a named scheme, the overview only fills free slots

**Search: KEEP.** The right text reached the prompt for 440 → 455 questions at 54 schemes, and 415 → 432 at 4,600. Benefits questions went 23 → 33 and 21 → 32.

**Answers:** not run to the end, because round 2b, which it builds on, was undone. It changes the prompt of most questions that name a scheme, so by round 1's lesson it is unlikely to pass with this answer model.

### Round 4: add a named scheme only when it is missing (adopted)

The smallest change that keeps round 2b's main win: if the question names a scheme that the prompt lacks, that scheme's rules replace the last slot. Every other prompt stays exactly as it was.

**Search: KEEP.**

| | 54 schemes | 4,600 schemes |
|---|---|---|
| Right text in prompt | 428 → 435 (+7, 0 broken) | 398 → 410 (+12, 0 broken) |
| 95% interval | +0.19 to +3.60 | +0.75 to +5.20 |
| Comparisons with both schemes | 3 → 10 of 10 | 3 → 9 of 10 |
| Other fixes | – | 6 Hindi questions naming the widow pension or NMMSS |

Prompts changed in the four practice sets: 3, 8, 13 and 10 (round 2b changed 18, 29, 24 and 39).

**Answers: passed.**

| Practice set | Correct | Fixed / broken | Wrong "yes" |
|---|---|---|---|
| Original, 54 schemes | 260 → 260 | 0 / 0 | 4 → 4 |
| Use cases, 54 schemes | 151 → 153 | 2 / 0 (comparisons 8 → 10 of 10) | 0 → 0 |
| Original, 4,600 schemes | 237 → 237 | 1 / 1 (same slice) | 4 → 4 |
| Use cases, 4,600 schemes | 142 → 143 | 1 / 0 | 0 → 0 |

**An independent code review then found flaws in round 4** that the practice questions barely touch:

- **It always replaced the *last* slot,** which could push out the very section the question asked for.
- **Everyday phrases counted as names.** At 4,600 schemes, bracketed descriptions in background titles ("Medical Treatment") and everyday titles ("small business") counted as names. Three practice questions that name no scheme had a slot changed this way.
- **Its guards were blunt.** They ignored "Senior Citizens' Savings Scheme" because of the word "for" before it, and ignored short names that other schemes mention ("PMJJBY", "AAY").
- **Notes inside aliases** ("(old name)") made those names unmatchable.

### Round 5: corrected name rules, first attempt

**Search: UNDO,** on guard A5: questions with their scheme in the prompt went 470 → 469 and 449 → 446. Its "everyday phrase" test rejected real names that other schemes mention ("PM Kisan", "e-Shram", "PM SVANidhi").

### Round 5b: corrected name rules (adopted as a defect fix, then undone after the exam)

What changed from round 5:

- The everyday-phrase test is dropped. Instead, only a single word in brackets ("(PMJJBY)") counts as a short name; a description in brackets does not.
- The slot given up is the least useful one, in this order: another scheme's section, then an overview, then a later section. It is never the rules of a named or top-ranked scheme.
- Short names that other schemes mention still count.
- Joining words ("for", "scheme", "yojana") no longer block a name.
- Aliases are read the same way as titles.
- A Hindi full stop no longer sticks to the name before it.
- A named scheme without a rules section brings its best section instead.

Each fix has its own test (`tests/test_search_methods.py`, `NameRulesTwoTests`).

**Search: no clear difference.** 435 → 435 at 54 schemes; 410 → 411 at 4,600 (+1). Nothing broke.

**Answers: passed.**

| Practice set | Correct | Fixed / broken | Wrong "yes" |
|---|---|---|---|
| Original, 54 schemes | 260 → 262 | 2 / 0 | 4 → 4 |
| Use cases, 54 schemes | 153 → 153 | 0 / 0 | 0 → 0 |
| Original, 4,600 schemes | 237 → 239 | 2 / 0 | 4 → 4 |
| Use cases, 4,600 schemes | 143 → 143 | 0 / 0 | 0 → 0 |

**Revision 2 of the rule, by the owner's decision.** Under the rule as written, "no clear difference" meant *not adopted*. The owner decided that a change fixing defects found by code review, each fix with a test, may be adopted when it fails no guard and passes the answer check.

### State after the cycle, practice questions

The live settings are `add_named_schemes: true` and `name_rules: 1` (round 4 only).

| | Before | After |
|---|---:|---:|
| Right text in prompt, 54 schemes (482 questions) | 428 | 435 |
| Right text in prompt, 4,600 schemes | 398 | 410 |
| Correct answers, original questions, 54 / 4,600 schemes | 260 / 237 | 260 / 237 |
| Correct answers, use cases, 54 / 4,600 schemes | 151 / 142 | 153 / 143 |
| Wrong "yes" answers, 54 / 4,600 schemes | 4 / 4 | 4 / 4 |
| Sections in the prompt | 3 | 3 |

Rounds 2b and 3b reached bigger search numbers (455 and 432) but were not adopted. Their records predate round 4's longer-name guard, which the `named_schemes` setting now also uses, so re-running them today gives slightly different prompts.

---

## 7. The exam

The exam is the held-out questions: 720 answerable original questions and 366 use-case questions, about the 36 exam schemes. It was run once for round 4, and once for round 4 + 5b. The close-out stopped the remaining use-case answer runs, so answers were measured on the original questions only.

### Search: does the needed text reach the prompt?

| Exam questions | Before the cycle | Round 4 | Round 4 + 5b |
|---|---:|---:|---:|
| Original, 54 schemes (720) | 652 (90.6%) | 655 (91.0%): +3, 0 broken | 655: +3, 0 broken |
| Original, 4,600 schemes (720) | 629 (87.4%) | 633 (87.9%): +4, 0 broken | 629: +4, **4 broken** |
| Use cases, 54 schemes (366) | 273 (74.6%) | **300 (82.0%)**: +27 (26 comparisons), 0 broken | 301: +28, 0 broken |
| Use cases, 4,600 schemes (366) | 247 (67.5%) | **265 (72.4%)**: +18 (17 comparisons), 0 broken | 273: +28, **2 broken** |

"Not found" refusals were unchanged in every set.

### Answers, original exam questions

| | Correct, 54 schemes | Correct, 4,600 schemes | Wrong "yes" |
|---|---|---|---|
| Before the cycle | 584 of 764 | 541 of 764 | 13 / 13 |
| Round 4 | 584 (0 fixed, 0 broken) | not run | 13 |
| Round 4 + 5b | 585 (1 fixed, 0 broken) | **536 (1 fixed, 6 broken)** | 13 / 13 |

### What the exam showed

- **Round 4 holds up.** No exam question got worse in search, the comparison gains carried over (27 and 18 more use-case questions with the needed text), and the answers at 54 schemes were unchanged.
- **Round 5b did not hold up at 4,600 schemes.** Its 6 broken answers and 6 broken search questions are PMEGP, PM Fasal Bima and PMAY-Urban questions. Two of its corrections misfired on wordings the practice questions never had:
  - **Cutting a " - note" tail from names also cut real titles.** "Pradhan Mantri Awas Yojana - Urban 2.0" became "Pradhan Mantri Awas Yojana", so a background copy, "PMAY-U", matched instead.
  - **No longer letting "for" or "scheme" block a name** let everyday background titles match ordinary phrases ("small food processing unit", "hailstorm crop damage", "commercial or horticulture crops"). The slot given up was the very section the question needed.
- **Round 5b was undone at close-out.** This decision used the exam, so the exam is no longer an independent check of the *choice* between rounds 4 and 5b. It remains one for round 4 against the settings from before the cycle.
- **Fixing round 5b's two misfires needs fresh held-out questions,** because the exam has now been used.

---

## 8. What this cycle did not fix

- **Round 4's known edge cases, found by the code review.**
  - It replaces the last slot, which can be the asked-for section.
  - Bracketed descriptions and everyday titles of background schemes can count as names.
  - Its guards ignore some real names, such as "Senior Citizens' Savings Scheme" written with "for" before it, and "PMJJBY".
  - Round 5b's fixes for these misfired on the exam. A corrected version needs fresh held-out questions to be checked on.

- **Benefits questions.**
  - Only 17 of 36 practice benefits answers are correct at 54 schemes; the overview often takes the benefits section's slot.
  - Round 3b fixed this in search, but it changes many prompts.
  - A version that changes only the questions that visibly ask about benefits would be the next round to try.
- **The small model's own errors,** such as wrong yes/no comparisons in Hindi and answers that just repeat the question. Search cannot fix these; see fine-tuning in the top-level README.
- **Name collisions in a large catalogue.** A few background schemes really are named "SSA" or with everyday words, and a question using those words gets that scheme added in the least useful slot.
- **"Not found" questions naming a state scheme we don't cover.** 3 of 12 practice ones still get some evidence.
- **Hindi-script questions naming schemes that have no Hindi name on file.** That is language coverage, deliberately not pursued.

---

## Files

| What | Where |
|---|---|
| Round records (every number, the fixed and broken question ids, the answer-check outcome) | `data/evaluation/india_schemes/search_rounds/*.json` |
| Answer checks per round | `data/evaluation/india_schemes/e2e_r*_*/` |
| Diagnosis, design-round and prototype data | `data/evaluation/india_schemes/design/` |
| Exam runs | `data/evaluation/india_schemes/exam_*` (round 4's: `exam_r4_*`) |
| Use-case questions | `domains/india_schemes/evaluation/usecase_questions.jsonl` |
| The mechanisms | `engine/retriever.py` (`retrieve_evidence`, `_add_missing`, `_named_v2`, `_scheme_names_v2`) |
| The round tool and its rule | `scripts/search_round.py` |
