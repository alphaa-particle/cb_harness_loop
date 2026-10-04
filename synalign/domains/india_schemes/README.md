# Indian central government schemes

54 real central government welfare schemes, one editable file each, written from official pages in plain words. This folder is what the chatbot answers from.

## What is here

| Path | What it is |
|---|---|
| `schemes/<id>.yaml` | one scheme: who it is for, rules, benefits, how to apply, documents, checkable facts, official sources, review status |
| `documents/` | built from the scheme files by `scripts/build_corpus.py`; never edit by hand |
| `evaluation/questions.jsonl` | the frozen test questions, built by `scripts/build_eval_questions.py` |
| `prompts.yaml` | what the answering model is told, and the fixed replies |
| `eval_config.yaml` | search settings, each with the reason it was chosen |
| `glossary.yaml` | Hindi and Hinglish words mapped to the English words the documents use |
| `aliases.yaml` | promises no answer may make |

## How the scheme files were made

1. One research agent wrote each scheme from official pages (ministry sites, scheme portals, guidelines, PIB releases), in its own words, with the source links.
2. A second agent checked every fact, rule number and Hindi name against official pages it opened itself.
3. Where they disagreed, a third agent decided from the official page.

Checking was stopped early on 2026-10-03 at the owner's request. Across the 39 schemes that were checked, 568 items were confirmed on an opened official page, 21 were found wrong and 16 could not be found; the wrong ones were mostly missing Hindi names, plus a few figures, all corrected.

## Review status

Each file ends with a `review` block.

| Status | Count | Meaning |
|---|---:|---|
| `verified` | 34 | every fact was confirmed on an official page someone opened |
| `needs_review` | 20 | at least one fact or statement was not confirmed; the notes say which |

Of the 20, 15 were never checked because checking stopped early. The notes in each file say exactly what is open and which official page to look at.

Each fact also has its own status: `verified`, `unverified` or `disputed`. Test questions are written only from verified facts.

## How to correct a fact

1. Open `schemes/<id>.yaml` and change the statement under `sections`.
2. If the number is also a test fact or rule, change `facts[].value` and `rules` to match. The checker refuses a number that the statements do not contain.
3. Add the official page you used under `sources`, with today's date.
4. Set the fact's `status` to `verified` once you have confirmed it, and add a line to `review.notes` saying what changed and why.
5. Rebuild and check:

```bash
python scripts/build_corpus.py --check
python scripts/build_corpus.py
python scripts/build_eval_questions.py
```

Schemes marked `needs_review` are included, as the owner chose. Add `--verified-only` to both commands for a corpus and questions built from fully checked schemes alone (34 schemes, 900 questions); results from that are not comparable with the 54-scheme results.

Rebuilding re-embeds only the sections that changed. If questions change, results measured on the old question file no longer compare with new ones; the question file's SHA-256 is printed and stored with every result.

## Sources and terms

- Facts come from official government pages, which are linked in each file.
- The text is our own wording, not copied from those pages.
- Hindi questions and the Hindi words in `glossary.yaml` were machine-drafted and have not been checked by a native speaker.
- Some schemes change often: amounts, age bands and whether new applications are accepted. Treat `review.checked_on` as the date the facts were true.
