"""Measure the whole pipeline on the frozen questions: search alone, then answers, against baselines.

    python scripts/serve_models.py &                       # both local models
    python scripts/evaluate_pipeline.py --split dev        # tune on dev ...
    python scripts/evaluate_pipeline.py --split test       # ... report on test

Search table (no model): for each search method, how often the right scheme is
ranked first or in the top 3, and how often the section holding the answer
reaches the model's prompt; for questions the documents cannot answer, how often
nothing is retrieved.

Answer table (model): the same questions answered under each condition:

    closed_book  the model alone, no documents (what it "knows")
    rules_only   no model: the matching rules shown as they are
    old_search   letter-run search, the original default
    meaning      embedding search only
    fusion       the full pipeline: word + letter + meaning search, rules first
    oracle       the correct sections handed over (the best any search could do)

Gaps worth reading: fusion minus closed_book is what retrieval adds; oracle minus
fusion is what better search could still add; 100 minus oracle is what the
model itself gets wrong even with the right text in front of it.

    python scripts/evaluate_pipeline.py --questions usecases --split dev

measures the use-case questions instead (how to apply, documents, benefits, who is
excluded, eligibility-and-apply, comparing two schemes, which schemes fit a need;
see scripts/build_eval_questions.py). Their search is right when every section the
answer needs is in the prompt (any one scheme, for "which schemes" questions). Their
answers are graded by plain rules:

    apply, documents, benefits, exclusions   not a refusal, and uses at least 2 words of the right
                                             section that no other section of that scheme uses
                                             (words of the question and of the scheme's names
                                             never count)
    apply_eligibility                        the same, at least one such word of each part
    compare                                  not a refusal, and for each scheme a word of its text that
                                             the other scheme's text lacks (weak: it cannot tell
                                             whether the comparison is true)
    category                                 names at least one scheme of that kind
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.answer_evaluation import is_refusal, states_value, verdict
from engine.corpus import load_corpus
from engine.retriever import tokens
from engine.domain_pack import DomainPack
from engine.enforcement import MODEL_ERROR, NO_EVIDENCE, UNCLEAR
from engine.evaluator import _word_pattern
from engine.llama_cpp import DECODER_URL, chat, is_ready
from engine.llm_assistant import LlamaCppAssistant, RulesOnlyAssistant
from engine.retriever import Retriever

ANSWER_RUNS = ("closed_book", "rules_only", "old_search", "meaning", "fusion", "oracle")
RUN_METHOD = {"rules_only": "fusion", "old_search": "char_tfidf", "meaning": "dense", "fusion": "fusion"}
SEARCH_METHODS = ("char_tfidf", "bm25", "dense", "fusion")


QUESTION_FILES = {"questions": "questions.jsonl", "usecases": "usecase_questions.jsonl"}
SECTION_TYPES = ("apply", "documents", "benefits", "exclusions", "apply_eligibility")


# Search settings --set may change; method, top_k and min_score have their own options because a
# run measures several methods and records them per run.
SETTABLE = {"fusion_weights", "rrf_k", "fusion_depth", "add_named_schemes", "name_rules", "rules_count_once",
            "named_schemes", "overview_last"}


def parse_settings(pairs: list[str], extra: tuple = ()) -> dict:
    """KEY=VALUE pairs (values read as YAML) for retrieval settings; unknown keys are refused."""
    out = {}
    allowed = SETTABLE | set(extra)
    for pair in pairs:
        key, sign, value = pair.partition("=")
        key = key.strip()
        if not sign:
            sys.exit(f"--set takes KEY=VALUE, got {pair!r}")
        if key not in allowed:
            sys.exit(f"--set cannot change {key!r}; settable: {', '.join(sorted(allowed))}"
                     + (" (use --top-k / --min-score)" if key in ("top_k", "min_score", "method") else ""))
        out[key] = yaml.safe_load(value)
    return out


def merged(current: dict, change: dict) -> dict:
    """The retrieval settings with a change applied; a nested value (fusion_weights) is merged, not replaced."""
    out = dict(current)
    for key, value in change.items():
        out[key] = {**(current.get(key) or {}), **value} if isinstance(value, dict) else value
    return out


def answer_present(q: dict, evidence_ids: set[str]) -> bool:
    """Is the text the answer needs in the prompt? Every gold group ("all"), or one of them ("any")."""
    groups = q.get("gold_groups") or [q["gold_chunk_ids"]]
    hits = [bool(set(group) & evidence_ids) for group in groups]
    return any(hits) if q.get("need") == "any" else all(hits)


class SectionWords:
    """Which words of an answer show it used the right text.

    own(group)   words of a section that no other section of the same scheme uses: words shared by
                 all of a scheme's sections (its name, its topic) show only that the answer is about
                 the right scheme; these show that it used the right section
    specific()   words of one scheme's text that another scheme's text does not use (for comparisons)
    Words of the question, and of the scheme's own names, never count: repeating the question or
    naming the scheme is not an answer.
    """

    def __init__(self, documents, names: dict[str, list[str]] | None = None):
        words = {d.chunk_id: {t for t in tokens(d.text) if len(t) >= 4 and t.isalpha()} for d in documents}
        siblings: dict[str, list[str]] = {}
        for d in documents:
            siblings.setdefault(d.scheme_key, []).append(d.chunk_id)
        self.scheme = {d.chunk_id: d.scheme_key for d in documents}
        self.words, self.siblings = words, siblings
        self.scheme_words = {key: set().union(*(words[c] for c in ids)) for key, ids in siblings.items()}
        self.name_words = {key: set() for key in siblings}
        for d in documents:
            self.name_words[d.scheme_key] |= set(tokens(" ".join([d.title.split(" — ")[0], *d.aliases])))
        for key, forms in (names or {}).items():
            self.name_words.setdefault(key, set()).update(tokens(" ".join(forms)))

    def own(self, group: list[str], question: str = "") -> set[str]:
        mine = set().union(*(self.words.get(c, set()) for c in group))
        others = {c for g in group for c in self.siblings.get(self.scheme.get(g), []) if c not in group}
        schemes = {self.scheme.get(c) for c in group}
        excluded = set(tokens(question)).union(*(self.name_words.get(k, set()) for k in schemes))
        return mine - set().union(*(self.words[c] for c in others)) - excluded

    def used(self, answer: str, chunk_ids: list[str], question: str = "") -> int:
        return len(set(tokens(answer)) & self.own(chunk_ids, question))

    def specific(self, scheme: str, other: str, question: str = "") -> set[str]:
        excluded = set(tokens(question)) | self.name_words.get(scheme, set()) | self.name_words.get(other, set())
        return self.scheme_words.get(scheme, set()) - self.scheme_words.get(other, set()) - excluded


def load(path: Path, split: str) -> tuple[list[dict], str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if split == "all" or r["split"] == split], hashlib.sha256(path.read_bytes()).hexdigest()


CORPUS = None   # set by --corpus: search a different corpus (for example the 5,000-scheme one)
SETTINGS = {}   # set by --set: search settings to use instead of eval_config.yaml's


def make_pack(domain: str, method: str, min_score: float, top_k: int, glossary: bool = True) -> DomainPack:
    pack = DomainPack(domain, corpus_path=CORPUS)
    pack.eval_config["retrieval"] = merged(pack.eval_config["retrieval"], SETTINGS)
    pack.eval_config["retrieval"].update(method=method, min_score=min_score, top_k=top_k)
    if not glossary:
        pack.glossary = {}
    return pack


def group_ci(values: list[float], groups: list[str], repeats: int = 1000) -> list[float]:
    """95% interval for a mean, resampling whole schemes so related questions stay together."""
    if not values:
        return [0.0, 0.0]
    by = {}
    for v, g in zip(values, groups):
        by.setdefault(g, []).append(v)
    sums = np.array([sum(v) for v in by.values()])
    sizes = np.array([len(v) for v in by.values()])
    rng = np.random.default_rng(42)
    picks = rng.integers(0, len(by), (repeats, len(by)))
    means = sums[picks].sum(1) / sizes[picks].sum(1)
    return [round(float(x), 4) for x in np.percentile(means, [2.5, 97.5])]


def summarise(rows: list[dict], key: str) -> dict:
    """Mean of rows[key] overall and by question type and language, with scheme-level intervals."""
    def block(subset):
        values = [float(r[key]) for r in subset if r.get(key) is not None]
        groups = [r["group_id"] for r in subset if r.get(key) is not None]
        return {"n": len(values), "rate": round(float(np.mean(values)), 4) if values else None,
                "ci95": group_ci(values, groups)}
    out = {"all": block(rows)}
    for field in ("type", "language"):
        for value in sorted({r[field] for r in rows}):
            out[f"{field}={value}"] = block([r for r in rows if r[field] == value])
    return out


# --- search alone ---------------------------------------------------------------------

def search_row(retriever: Retriever, q: dict) -> dict:
    ranked = retriever.retrieve(q["question"], top_k=10)
    evidence = retriever.retrieve_evidence(q["question"])
    row = {k: q[k] for k in ("query_id", "group_id", "type", "language")}
    if q["expect_abstain"]:
        row["abstained"] = float(not evidence)
        return row
    if q.get("scheme_id"):
        schemes = list(dict.fromkeys(c.scheme_id for c in ranked))
        rank = schemes.index(q["scheme_id"]) + 1 if q["scheme_id"] in schemes else None
        row.update({"scheme_hit1": float(rank == 1), "scheme_hit3": float(bool(rank) and rank <= 3),
                    "scheme_mrr": 1 / rank if rank else 0.0})
    row["answer_in_prompt"] = float(answer_present(q, {c.chunk_id for c in evidence}))
    return row


def search_table(domain, questions, min_score, top_k, out_dir=None, methods=None):
    table = {}
    for method in methods or SEARCH_METHODS + ("fusion_no_glossary",):
        name, glossary = (("fusion", False) if method == "fusion_no_glossary" else (method, True))
        threshold = min_score if name in ("dense", "fusion") else 0.0
        started = time.time()
        retriever = Retriever(make_pack(domain, name, threshold, top_k, glossary))
        build = time.time() - started
        started = time.time()
        rows = [search_row(retriever, q) for q in questions]
        per_query = (time.time() - started) / len(questions) * 1000
        if out_dir is not None:       # per question, for scripts/compare_runs.py --search
            (out_dir / f"search_{method}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        table[method] = {"build_seconds": round(build, 2), "ms_per_question": round(per_query, 2),
                         **{metric: summarise([r for r in rows if metric in r], metric)
                            for metric in ("scheme_hit1", "scheme_hit3", "scheme_mrr", "answer_in_prompt", "abstained")}}
        print(f"search {method:20s} hit@1 {table[method]['scheme_hit1']['all']['rate']}  "
              f"in-prompt {table[method]['answer_in_prompt']['all']['rate']}  "
              f"abstain {table[method]['abstained']['all']['rate']}  ({per_query:.1f} ms/q)", flush=True)
    return table


# --- answers --------------------------------------------------------------------------

class ClosedBook:
    """The model alone: the same instructions about language and length, no documents."""

    def __init__(self, pack):
        self.system = pack.prompts["closed_book_system"]

    def answer(self, question):
        text = chat([{"role": "system", "content": self.system}, {"role": "user", "content": question}])
        return text, "answered", []


def opens_refusing(text: str) -> bool:
    """Does the answer's first sentence say it could not find the answer?"""
    return is_refusal(re.split(r"(?<=[.!?।])\s+|\n", text.strip(), maxsplit=1)[0])


def grade(q: dict, text: str, mode: str, scheme_patterns: dict, words: SectionWords | None = None,
          digit_names: re.Pattern | None = None) -> dict:
    refused = mode in (NO_EVIDENCE, UNCLEAR) or is_refusal(text)
    row = {k: q[k] for k in ("query_id", "group_id", "type", "language")}
    if q["expect_abstain"]:
        row["correct"] = float(refused)
        return row
    names = lambda sid: any(p.search(text) for p in scheme_patterns.get(sid, []))  # noqa: E731
    if q["type"] in SECTION_TYPES:
        # at least one word of each needed section's own (two in all), where the section has that many
        available = [len(words.own(group, q["question"])) for group in q["gold_groups"]]
        used = [words.used(text, group, q["question"]) for group in q["gold_groups"]]
        enough = (all(u >= min(1, a) for u, a in zip(used, available))
                  and sum(used) >= min(2, sum(available)))
        row["correct"] = float(not refused and enough)
    elif q["type"] == "compare":
        # says something specific to each scheme: a word of its text that the other's text lacks (a weak
        # check: it cannot tell whether the comparison is true; the search measure says whether both
        # schemes' text reached the model)
        a, b = q["gold_answer"]["schemes"]
        said = set(tokens(text))
        each = [bool(said & words.specific(x, y, q["question"])) or not words.specific(x, y, q["question"])
                for x, y in ((a, b), (b, a))]
        row["correct"] = float(not refused and all(each))
    elif q["type"] == "category":
        row["correct"] = float(not refused and any(names(sid) for sid in q["gold_answer"]["schemes"]))
    elif q["type"] in ("fact", "typo"):
        # numbers inside scheme names ("Poshan 2.0") are not values; forms the question itself
        # contains cannot show an answer; an answer that opens by refusing does not count
        for_numbers = digit_names.sub(" ", text) if digit_names else text
        accept = [a for a in q["gold_answer"]["accept"] if not _in_question(a, q["question"])]
        row["correct"] = float(not opens_refusing(text)
                               and states_value(for_numbers, q["gold_answer"]["value"], accept))
    elif q["type"] == "situation":
        row["correct"] = float(any(p.search(text) for p in scheme_patterns.get(q["scheme_id"], [])))
    else:
        said = verdict(text)
        row["verdict"] = said
        row["correct"] = float(said == "no") if q["gold_verdict"] == "no" else float(said in ("yes", "unsure"))
        row["unsafe_yes"] = float(q["gold_verdict"] == "no" and said == "yes")
    row["refused"] = float(refused)
    if q["language"] in ("hi", "en") and not refused:
        row["same_language"] = float(replied_in(text) == q["language"])
    return row


def _in_question(form: str, question: str) -> bool:
    """An accepted form the question already contains, word for word ("once" in "...or only once?")."""
    return bool(tokens(form)) and f" {' '.join(tokens(form))} " in f" {' '.join(tokens(question))} "


def answer_key(pack) -> tuple[dict, SectionWords, re.Pattern | None]:
    """Everything grade() needs from the domain: scheme name patterns, section words, names with digits."""
    aliases = pack.entity_aliases()
    patterns = {sid: [_word_pattern(a) for a in names] for sid, names in aliases.items()}
    documents = load_corpus(pack.documents_dir if hasattr(pack, "documents_dir") else pack.corpus_path)
    words = SectionWords(documents, aliases)
    with_digits = sorted({n for names in aliases.values() for n in names if re.search(r"\d", n)}, key=len, reverse=True)
    digit_names = re.compile("|".join(re.escape(n) for n in with_digits), re.IGNORECASE) if with_digits else None
    return patterns, words, digit_names


def replied_in(text: str) -> str:
    """'hi' when at least a third of the letters are Devanagari, else 'en'."""
    letters = [ch for ch in text if ch.isalpha()]
    devanagari = sum("\u0900" <= ch <= "\u097f" for ch in letters)
    return "hi" if letters and devanagari / len(letters) >= 1 / 3 else "en"


def oracle_ids(q: dict, top_k: int) -> list[str]:
    """The correct sections to hand over: one from each needed group, then the next of each, until full."""
    if q.get("gold_groups"):
        picks, depth = [], 0
        while len(picks) < top_k and any(depth < len(g) for g in q["gold_groups"]):
            picks += [g[depth] for g in q["gold_groups"] if depth < len(g)]
            depth += 1
        return picks[:top_k]
    if q["type"] == "situation":
        return [c for c in q["gold_chunk_ids"] if c.endswith("__eligibility")][:1] or q["gold_chunk_ids"][:1]
    return q["gold_chunk_ids"][:top_k]


def answer_run(run, domain, questions, min_score, top_k, max_tokens, out_dir):
    pack = DomainPack(domain)
    patterns, words, digit_names = answer_key(pack)
    if run == "closed_book":
        assistant, retriever = ClosedBook(pack), None
    else:
        method = RUN_METHOD.get(run, "fusion")
        threshold = min_score if method in ("dense", "fusion") else 0.0
        retriever = Retriever(make_pack(domain, method, threshold, top_k))
        cls = RulesOnlyAssistant if run == "rules_only" else LlamaCppAssistant
        assistant = cls(pack, retriever) if run == "rules_only" else cls(pack, retriever, max_new_tokens=max_tokens)
    records, graded = [], []
    started = time.time()
    for n, q in enumerate(questions, 1):
        if run == "oracle" and q["expect_abstain"]:
            continue                       # nothing correct to hand over
        t = time.time()
        if run == "closed_book":
            text, mode, evidence = assistant.answer(q["question"])
        elif run == "oracle":
            a = assistant.answer_with_context(q["question"], retriever.get_chunks_by_ids(oracle_ids(q, top_k)))
            text, mode, evidence = a.answer_text, a.mode, [c.chunk_id for c in a.retrieved_chunks]
        else:
            a = assistant.answer(q["question"])
            text, mode, evidence = a.answer_text, a.mode, [c.chunk_id for c in a.retrieved_chunks]
        if mode == MODEL_ERROR and not is_ready(DECODER_URL):
            # A stopped model would be scored as wrong answers; stop instead of reporting that.
            sys.exit(f"The answer model stopped responding during {run} (question {n}); restart it and run again.")
        row = grade(q, text, mode, patterns, words, digit_names)
        graded.append(row)
        records.append({**row, "question": q["question"], "answer": text, "mode": mode, "evidence": evidence,
                        "seconds": round(time.time() - t, 3)})
        if n % 50 == 0:
            print(f"  {run}: {n}/{len(questions)}", flush=True)
    (out_dir / f"answers_{run}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    summary = {"correct": summarise(graded, "correct"),
               "unsafe_yes": summarise([r for r in graded if "unsafe_yes" in r], "unsafe_yes"),
               "same_language": summarise([r for r in graded if "same_language" in r], "same_language"),
               "seconds_per_answer": round((time.time() - started) / max(len(graded), 1), 3)}
    print(f"answers {run:12s} correct {summary['correct']['all']['rate']}  "
          f"({summary['seconds_per_answer']} s/answer)", flush=True)
    return summary, {r["query_id"]: r for r in graded}


def regrade_run(run: str, questions: list[dict], out_dir: Path, domain: str) -> tuple[dict, dict]:
    """Score saved answers again with the current grader; the model is not run."""
    pack = DomainPack(domain)
    patterns, words, digit_names = answer_key(pack)
    by_id = {q["query_id"]: q for q in questions}
    path = out_dir / f"answers_{run}.jsonl"
    records, graded = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        old = json.loads(line)
        row = grade(by_id[old["query_id"]], old["answer"], old["mode"], patterns, words, digit_names)
        graded.append(row)
        records.append({**row, **{k: old[k] for k in ("question", "answer", "mode", "evidence", "seconds")}})
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    seconds = round(float(np.mean([r["seconds"] for r in records])), 3)
    summary = {"correct": summarise(graded, "correct"),
               "unsafe_yes": summarise([r for r in graded if "unsafe_yes" in r], "unsafe_yes"),
               "same_language": summarise([r for r in graded if "same_language" in r], "same_language"),
               "seconds_per_answer": seconds}
    print(f"regraded {run:12s} correct {summary['correct']['all']['rate']}", flush=True)
    return summary, {r["query_id"]: r for r in graded}


def paired(a: dict, b: dict) -> dict:
    """b minus a on the questions both runs answered, with a scheme-level interval."""
    ids = sorted(set(a) & set(b))
    diffs = [b[i]["correct"] - a[i]["correct"] for i in ids]
    return {"n": len(ids), "difference": round(float(np.mean(diffs)), 4) if ids else None,
            "ci95": group_ci(diffs, [a[i]["group_id"] for i in ids])}


def pct(block: dict | None) -> str:
    if not block or block.get("rate") is None:
        return "–"
    lo, hi = block["ci95"]
    return f"{block['rate'] * 100:.1f}% ({lo * 100:.0f}–{hi * 100:.0f})"


TYPE_NAMES = {"fact": "Fact", "typo": "Typo", "situation": "Situation", "eligibility": "Eligibility",
              "absent": "Not found (scheme absent)", "off_topic": "Not found (off topic)", "apply": "How to apply",
              "documents": "Documents", "benefits": "Benefits", "exclusions": "Who is excluded",
              "apply_eligibility": "Eligibility and how to apply", "compare": "Compare two", "category": "Which schemes"}


def markdown(report: dict) -> str:
    """The report as tables a person can read; every number is also in report.json."""
    def kinds(blocks: dict, prefix: str) -> list[str]:
        return [k[len(prefix):] for k in blocks if k.startswith(prefix)]

    first = next(iter(report["search"].values()))["answer_in_prompt"]
    types = [t for t in TYPE_NAMES if f"type={t}" in first]
    languages = kinds(first, "language=")
    lang_names = {"en": "English", "hi": "Hindi", "hi_latn": "Hindi in Latin letters"}
    lines = [f"# Pipeline evaluation: {report['split']} questions", "",
             f"Questions: `{report['questions_file']}` (sha256 `{report['questions_sha256'][:12]}`). "
             f"Corpus: `{report['corpus']}`. Sections in the prompt: {report['top_k']}. "
             f"Meaning-similarity floor: {report['min_score']}.", "",
             "Each cell: rate, then the 95% interval from resampling whole schemes.", "",
             "## Search alone (no model)", "",
             "Right text in the prompt: every section the answer needs is among those given to the model.", "",
             "| Method | All | " + " | ".join(TYPE_NAMES[t] for t in types) + " | "
             + " | ".join(lang_names.get(x, x) for x in languages) + " | Not-found questions refused | ms per question |",
             "|---" * (4 + len(types) + len(languages)) + "|"]
    for method, r in report["search"].items():
        m = r["answer_in_prompt"]
        lines.append(f"| {method} | {pct(m['all'])} | " + " | ".join(pct(m.get(f"type={t}")) for t in types) + " | "
                     + " | ".join(pct(m.get(f"language={x}")) for x in languages)
                     + f" | {pct(r['abstained']['all'])} | {r['ms_per_question']} |")
    if "answers" in report:
        answer_types = [t for t in TYPE_NAMES
                        if any(f"type={t}" in r["correct"] for r in report["answers"].values())]
        lines += ["", "## Answers", "",
                  "Correct: see the grading rules at the top of scripts/evaluate_pipeline.py.", "",
                  "| Condition | All | " + " | ".join(TYPE_NAMES[t] for t in answer_types)
                  + " | Unsafe yes | Hindi answer to Hindi question | s per answer |",
                  "|---" * (6 + len(answer_types)) + "|"]
        for run, r in report["answers"].items():
            c = r["correct"]
            lines.append(f"| {run} | {pct(c['all'])} | " + " | ".join(pct(c.get(f"type={t}")) for t in answer_types)
                         + f" | {pct(r['unsafe_yes']['all'])} | "
                         f"{pct((r.get('same_language') or {}).get('language=hi'))} | {r['seconds_per_answer']} |")
        lines += ["", "## Paired differences (same questions, percentage points)", "",
                  "| Comparison | Questions | Difference | 95% interval |", "|---|---|---|---|"]
        for name, d in report.get("paired", {}).items():
            if d["difference"] is not None:
                lines.append(f"| {name.replace('_minus_', ' minus ')} | {d['n']} | {d['difference'] * 100:+.1f} | "
                             f"{d['ci95'][0] * 100:+.1f} to {d['ci95'][1] * 100:+.1f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", default="india_schemes")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--questions", default="questions", choices=sorted(QUESTION_FILES),
                        help="questions (facts, situations, eligibility...) or usecases (apply, documents, compare...)")
    parser.add_argument("--runs", nargs="*", default=list(ANSWER_RUNS), choices=ANSWER_RUNS)
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--search-methods", nargs="*", choices=SEARCH_METHODS + ("fusion_no_glossary",),
                        help="Search methods to measure (default: all)")
    parser.add_argument("--min-score", type=float, help="Meaning-similarity floor (default: eval_config.yaml)")
    parser.add_argument("--top-k", type=int, help="Sections in the prompt (default: eval_config.yaml)")
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--label", help="Output folder name (default: the split)")
    parser.add_argument("--corpus", type=Path, help="Search this corpus instead of the domain's own documents")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Search settings to use instead of eval_config.yaml's (for trying a change)")
    parser.add_argument("--regrade", action="store_true",
                        help="Score the answers already saved under --label again with the current grader")
    args = parser.parse_args()
    global CORPUS
    CORPUS = args.corpus
    if args.top_k is None:
        args.top_k = int(DomainPack(args.domain).eval_config["retrieval"]["top_k"])
    SETTINGS.update(parse_settings(args.set))
    path = Path(__file__).resolve().parents[1] / "domains" / args.domain / "evaluation" / QUESTION_FILES[args.questions]
    questions, sha = load(path, args.split)
    min_score = args.min_score if args.min_score is not None else DomainPack(args.domain).eval_config["retrieval"]["min_score"]
    out_dir = Path(__file__).resolve().parents[1] / "data" / "evaluation" / args.domain / (args.label or args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(questions)} {args.split} questions (sha256 {sha[:12]}), min_score {min_score}, top_k {args.top_k}")
    if args.regrade:
        report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        # grade against the questions and split the answers were made for, whatever the options say
        path = Path(report["questions_file"])
        questions, sha = load(path, report["split"])
        if report["questions_sha256"] != sha:
            sys.exit("These answers were made for a different question file; they cannot be re-graded against this one.")
        graded = {}
        for run in [r for r in ANSWER_RUNS if (out_dir / f"answers_{r}.jsonl").is_file()]:
            report["answers"][run], graded[run] = regrade_run(run, questions, out_dir, args.domain)
        report["paired"] = {f"{b}_minus_{a}": paired(graded[a], graded[b])
                            for a, b in (("closed_book", "fusion"), ("old_search", "fusion"), ("fusion", "oracle"),
                                         ("closed_book", "old_search"), ("meaning", "fusion"))
                            if a in graded and b in graded}
        report["regraded"] = True
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (out_dir / "report.md").write_text(markdown(report), encoding="utf-8")
        print(f"Report: {out_dir / 'report.md'}")
        return
    report = {"questions_file": str(path), "questions_sha256": sha, "split": args.split, "min_score": min_score,
              "corpus": str(args.corpus or DomainPack(args.domain).corpus_path),
              "top_k": args.top_k, "questions": args.questions,
              "retrieval": {**merged(DomainPack(args.domain).eval_config["retrieval"], SETTINGS),
                            "top_k": args.top_k, "min_score": min_score},
              "search": search_table(args.domain, questions, min_score, args.top_k, out_dir, args.search_methods)}
    if not args.search_only:
        if not is_ready(DECODER_URL):
            sys.exit(f"The answer model is not running at {DECODER_URL}; start it with: python scripts/serve_models.py")
        report["answers"], graded = {}, {}
        for run in args.runs:
            report["answers"][run], graded[run] = answer_run(run, args.domain, questions, min_score, args.top_k,
                                                             args.max_tokens, out_dir)
        report["paired"] = {f"{b}_minus_{a}": paired(graded[a], graded[b])
                            for a, b in (("closed_book", "fusion"), ("old_search", "fusion"), ("fusion", "oracle"),
                                         ("closed_book", "old_search"), ("meaning", "fusion"))
                            if a in graded and b in graded}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(markdown(report), encoding="utf-8")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
