"""One SynAlign search round: try one change to the search settings, keep it only if nothing gets worse.

    python scripts/serve_models.py --only embedder --detach
    python scripts/search_round.py --label r1_example --set top_k=4 --note "why this change"

"Before" is the domain's eval_config.yaml as it stands (the last kept state); "after" is the same
with --set applied. Both are measured on the practice (dev) questions of both question files
(questions.jsonl: facts, situations, eligibility, not-found; usecase_questions.jsonl: how to
apply, documents, benefits, exclusions, comparisons, "which schemes"); a question counts when
every section its answer needs is in the prompt. The test questions are never loaded. Two sizes: the domain's own sections, and the same hidden among ~4,600 background
schemes (data/external/scale_corpus/corpus.jsonl, when present). A second set of questions, the
fine-tuning examples' questions (data/training/<domain>/), checks that the asked scheme still
reaches the prompt on wordings the round was not designed on.

The rule was written down before the rounds were run. Search changes are cheap to try and easy
to fit to a few hundred questions, so a change is KEPT only if every guard holds at both sizes:

    A1  gain        the section holding the answer reaches the prompt for at least 1 more question (net)
    A2  breaks      at most 1 question that worked before stops working
    A3  slices      no language and no question type does worse than before
    A4  refusals    not-found questions are refused no less often (judged on the domain's own
                    sections only: among background schemes an "absent" scheme may really exist),
                    and no more answerable questions get nothing
    A5  scheme      no fewer answerable questions get their scheme into the prompt
    A6  interval    the scheme-level 95% interval of the gain does not reach below zero
    A7  budget      no worse with 2 or 5 sections in the prompt instead of 3
    A8  wordings    on the fine-tuning questions, no new miss of the asked scheme and no new refusal
    A9  speed       search (without the question's embedding call) at most twice as slow as before
                    (the median of 5 passes, before and after alternating)

A1, A2, A5 and A6 are judged on both question files together, at each size; A3 per file, per
question type and per language; A4's not-found refusals on the domain's own sections only.
UNDO if any of A2-A5 or A7-A9 fails, or if the whole A6 interval is below zero; NO CLEAR
DIFFERENCE (not adopted) if only A1 fails or A6's interval reaches below zero.

One revision since the rule was first written: A9 was first timed in a single pass, and a busy
machine made one change read 0.30 -> 1.40 ms at 54 schemes while it read faster at 4,600; the
timing became the median of 5 alternating passes (round 1, attempt 2). The thresholds are unchanged.

A second revision, by the owner's decision (round 5b): a change that fixes defects found by
reviewing the code, each fix with its own test, may be adopted without a measurable gain, if it
fails no guard (A2-A5, A7-A9) and passes the answer check. Search changes meant to improve
accuracy still need the gain.

A KEEP is adopted only after the end-to-end answer check, on both question files at both sizes:

    S="--split dev --runs fusion --search-methods fusion --set <the change>"
    python scripts/evaluate_pipeline.py $S --label after_q_54
    python scripts/evaluate_pipeline.py $S --questions usecases --label after_u_54
    python scripts/evaluate_pipeline.py $S --corpus data/external/scale_corpus/corpus.jsonl --label after_q_scale
    python scripts/evaluate_pipeline.py $S --questions usecases --corpus data/external/scale_corpus/corpus.jsonl --label after_u_scale
    python scripts/compare_runs.py data/evaluation/<domain>/<before_q_54> data/evaluation/<domain>/after_q_54 --same-model
    (and the same comparison for the other three)

The result is written to data/evaluation/<domain>/search_rounds/<label>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import ROOT_DIR
from engine.domain_pack import DomainPack
from engine.retriever import Retriever
from scripts.evaluate_pipeline import QUESTION_FILES, answer_present, group_ci, merged, parse_settings

SCALE_CORPUS = ROOT_DIR / "data" / "external" / "scale_corpus" / "corpus.jsonl"
BUDGETS = (3, 2, 5)


def dev_questions(domain: str) -> list[dict]:
    """Dev questions of every question file there is, each tagged with its file ("set")."""
    out = []
    for name, filename in QUESTION_FILES.items():
        path = ROOT_DIR / "domains" / domain / "evaluation" / filename
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            q = json.loads(line) if line.strip() else None
            if q and q["split"] == "dev":
                out.append({**q, "set": name, "query_id": f"{name}:{q['query_id']}"})
    return out


def wording_questions(domain: str) -> list[dict]:
    """The questions of the fine-tuning examples (dev schemes, other wordings), one per distinct question."""
    seen, out = set(), []
    for name in ("sft_train.jsonl", "sft_dev.jsonl"):
        path = ROOT_DIR / "data" / "training" / domain / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            content = row["messages"][1]["content"]
            question = content.split("User question:\n", 1)[1].split("\n\nAnswer using", 1)[0].strip()
            if (question, row["scheme_id"]) not in seen:
                seen.add((question, row["scheme_id"]))
                out.append({"question": question, "scheme_id": row["scheme_id"]})
    return out


class Side:
    """One configuration on one corpus: the evidence for every question, at every budget."""

    def __init__(self, domain: str, corpus, settings: dict, questions: list[dict], wordings: list[dict]):
        pack = DomainPack(domain, corpus_path=corpus)
        pack.eval_config["retrieval"] = merged(pack.eval_config["retrieval"], settings)
        self.retriever = Retriever(pack)
        self.top_k = self.retriever.top_k
        self.rows = {k: {q["query_id"]: self.row(q, k) for q in questions} for k in {*BUDGETS, self.top_k}}
        self.wordings = [self.reach(w) for w in wordings]
        self.ms = None                          # set by time_sides()

    def row(self, q: dict, k: int) -> dict:
        evidence = self.retriever.retrieve_evidence(q["question"], top_k=k)
        if q["expect_abstain"]:
            return {"refused": not evidence}
        schemes = {c.scheme_id for c in evidence}
        wanted = (q.get("gold_answer") or {}).get("schemes") or [q["scheme_id"]]
        return {"in_prompt": answer_present(q, {c.chunk_id for c in evidence}),
                "scheme": (any if q.get("need") == "any" else all)(s in schemes for s in wanted),
                "refused": not evidence, "ids": [c.chunk_id for c in evidence]}

    def reach(self, w: dict) -> dict:
        evidence = self.retriever.retrieve_evidence(w["question"])
        return {"scheme": w["scheme_id"] in {c.scheme_id for c in evidence}, "refused": not evidence}


def time_sides(before: Side, after: Side, questions: list[dict], passes: int = 5) -> None:
    """Search time per question, without the embedding call (cached by now): the median of several
    passes, before and after alternating, so a busy machine slows both alike."""
    runs = {id(before): [], id(after): []}
    for _ in range(passes):
        for side in (before, after):
            started = time.perf_counter()
            for q in questions:
                side.retriever.retrieve_evidence(q["question"])
            runs[id(side)].append((time.perf_counter() - started) / len(questions) * 1000)
    before.ms, after.ms = (statistics.median(runs[id(side)]) for side in (before, after))


class CachedQueries:
    """Wraps an embedder so each distinct question is embedded once; timing then measures search alone."""

    def __init__(self, embedder):
        self.embedder, self.cache = embedder, {}
        self.model_id = embedder.model_id

    def embed_documents(self, texts):
        return self.embedder.embed_documents(texts)

    def embed_query(self, text):
        if text not in self.cache:
            self.cache[text] = self.embedder.embed_query(text)
        return self.cache[text]


def judge_size(name: str, questions: list[dict], before: Side, after: Side, own_corpus: bool) -> tuple[dict, list[str]]:
    by_id = {q["query_id"]: q for q in questions}
    b, a = before.rows[before.top_k], after.rows[after.top_k]       # each side at its own prompt size
    answerable = [i for i in b if "in_prompt" in b[i]]
    fixed = [i for i in answerable if a[i]["in_prompt"] and not b[i]["in_prompt"]]
    broken = [i for i in answerable if b[i]["in_prompt"] and not a[i]["in_prompt"]]
    diffs = [float(a[i]["in_prompt"]) - float(b[i]["in_prompt"]) for i in answerable]
    lo, hi = group_ci(diffs, [by_id[i]["group_id"] for i in answerable])
    slices = {}
    for i in answerable:
        for field in ("type", "language"):
            key = f"{by_id[i]['set']}:{field}={by_id[i][field]}"
            slices.setdefault(key, [0, 0])
            slices[key][0] += b[i]["in_prompt"]
            slices[key][1] += a[i]["in_prompt"]
    not_found = [i for i in b if "in_prompt" not in b[i]]
    count = lambda rows, ids, key: sum(bool(rows[i][key]) for i in ids)  # noqa: E731
    budgets = {kk: (count(before.rows[kk], answerable, "in_prompt"), count(after.rows[kk], answerable, "in_prompt"))
               for kk in BUDGETS}
    budgets[before.top_k] = (count(b, answerable, "in_prompt"), count(a, answerable, "in_prompt"))
    words_lost = sum(w0["scheme"] and not w1["scheme"] for w0, w1 in zip(before.wordings, after.wordings))
    words_refused = sum(w1["refused"] and not w0["refused"] for w0, w1 in zip(before.wordings, after.wordings))
    report = {
        "answerable": len(answerable), "in_prompt": [count(b, answerable, "in_prompt"), count(a, answerable, "in_prompt")],
        "fixed": fixed, "broken": broken, "gain_points": round(sum(diffs) / len(diffs) * 100, 2),
        "ci95_points": [round(lo * 100, 2), round(hi * 100, 2)],
        "slices": {key: [int(x), int(y)] for key, (x, y) in sorted(slices.items())},
        "scheme_in_prompt": [count(b, answerable, "scheme"), count(a, answerable, "scheme")],
        "answerable_refused": [count(b, answerable, "refused"), count(a, answerable, "refused")],
        "not_found_refused": [count(b, not_found, "refused"), count(a, not_found, "refused"), len(not_found)],
        "budgets": {str(kk): list(v) for kk, v in budgets.items()},
        "wordings": {"questions": len(before.wordings), "scheme_lost": words_lost, "newly_refused": words_refused,
                     "scheme_in_prompt": [sum(w["scheme"] for w in before.wordings),
                                          sum(w["scheme"] for w in after.wordings)]},
        "ms_per_question": [round(before.ms, 2), round(after.ms, 2)],
    }
    failed = []
    if len(broken) > 1:
        failed.append(f"A2 {name}: {len(broken)} questions broken")
    for key, (x, y) in report["slices"].items():
        if y < x:
            failed.append(f"A3 {name}: {key} {x} -> {y}")
    nf_b, nf_a, _ = report["not_found_refused"]
    if own_corpus and nf_a < nf_b:
        failed.append(f"A4 {name}: not-found refused {nf_b} -> {nf_a}")
    if report["answerable_refused"][1] > report["answerable_refused"][0]:
        failed.append(f"A4 {name}: answerable questions given nothing {report['answerable_refused']}")
    if report["scheme_in_prompt"][1] < report["scheme_in_prompt"][0]:
        failed.append(f"A5 {name}: scheme in prompt {report['scheme_in_prompt']}")
    for kk in (2, 5):
        x, y = budgets[kk]
        if y < x:
            failed.append(f"A7 {name}: {kk} sections {x} -> {y}")
    if words_lost or words_refused:
        failed.append(f"A8 {name}: fine-tuning questions lost their scheme {words_lost}, newly refused {words_refused}")
    if after.ms > 2 * max(before.ms, 0.05):
        failed.append(f"A9 {name}: search {before.ms:.2f} -> {after.ms:.2f} ms per question")
    weak = []
    if len(fixed) - len(broken) < 1:
        weak.append(f"A1 {name}: net gain {len(fixed) - len(broken)}")
    if lo < 0:
        weak.append(f"A6 {name}: interval reaches {lo * 100:+.2f} points")
    if hi < 0:
        failed.append(f"A6 {name}: the whole interval is below zero")
    report["guards_failed"], report["gain_not_shown"] = failed, weak
    return report, failed + [f"(weak) {w}" for w in weak]


def print_size(name: str, r: dict) -> None:
    b, a = r["in_prompt"]
    print(f"\n{name}: right section in prompt {b} -> {a} of {r['answerable']} "
          f"(fixed {len(r['fixed'])}, broken {len(r['broken'])}; {r['gain_points']:+.2f} points, "
          f"95% interval {r['ci95_points'][0]:+.2f} to {r['ci95_points'][1]:+.2f})")
    for key, (x, y) in r["slices"].items():
        print(f"    {key:34s} {x:4d} -> {y:4d}")
    nf_b, nf_a, n = r["not_found_refused"]
    print(f"  scheme in prompt {r['scheme_in_prompt'][0]} -> {r['scheme_in_prompt'][1]};  "
          f"not-found refused {nf_b} -> {nf_a} of {n};  answerable given nothing "
          f"{r['answerable_refused'][0]} -> {r['answerable_refused'][1]}")
    print("  sections 2 / 3 / 5: " + "  ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in r["budgets"].items()))
    w = r["wordings"]
    print(f"  fine-tuning wordings ({w['questions']}): scheme in prompt {w['scheme_in_prompt'][0]} -> "
          f"{w['scheme_in_prompt'][1]}, lost {w['scheme_lost']}, newly refused {w['newly_refused']}")
    print(f"  search time {r['ms_per_question'][0]} -> {r['ms_per_question'][1]} ms per question (embedding excluded)")
    if r["fixed"]:
        print(f"  fixed: {', '.join(r['fixed'])}")
    if r["broken"]:
        print(f"  broken: {', '.join(r['broken'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", default="india_schemes")
    parser.add_argument("--label", required=True, help="Name of this round (the result file's name)")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Search settings to change (retrieval.* in eval_config.yaml)")
    parser.add_argument("--base", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Settings already kept but not yet written to eval_config.yaml (applied to both sides)")
    parser.add_argument("--note", default="", help="What the change is and why, for the round's record")
    args = parser.parse_args()
    extra = ("top_k", "min_score")              # each side is judged at its own prompt size and floor
    change, base = parse_settings(args.set, extra), parse_settings(args.base, extra)
    questions, wordings = dev_questions(args.domain), wording_questions(args.domain)
    current = {**DomainPack(args.domain).eval_config["retrieval"], **base}
    print(f"{len(questions)} dev questions, {len(wordings)} fine-tuning wordings. Base: {base or 'eval_config.yaml'}. "
          f"Change: {change}")
    from engine.llama_cpp import LlamaEmbedder
    embedder = CachedQueries(LlamaEmbedder())
    import engine.llama_cpp as llama
    sizes = [("54 schemes" if args.domain == "india_schemes" else "own sections", None, True)]
    if SCALE_CORPUS.is_file():
        sizes.append(("background scale", SCALE_CORPUS, False))
    else:
        print(f"note: {SCALE_CORPUS} is missing; only the domain's own sections are measured")
    original = llama.LlamaEmbedder
    llama.LlamaEmbedder = lambda *a, **k: embedder           # one shared, cached embedder for every Retriever
    try:
        results, problems = {}, []
        for name, corpus, own in sizes:
            before = Side(args.domain, corpus, base, questions, wordings)
            after = Side(args.domain, corpus, {**base, **change}, questions, wordings)
            time_sides(before, after, questions)
            results[name], issues = judge_size(name, questions, before, after, own)
            problems += issues
            print_size(name, results[name])
    finally:
        llama.LlamaEmbedder = original
    failed = [p for p in problems if not p.startswith("(weak)")]
    weak = [p for p in problems if p.startswith("(weak)")]
    verdict = "UNDO" if failed else ("NO CLEAR DIFFERENCE" if weak or len(sizes) < 2 else "KEEP")
    print()
    for line in problems:
        print(f"  {line}")
    print(f"VERDICT: {verdict}")
    out = ROOT_DIR / "data" / "evaluation" / args.domain / "search_rounds" / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": args.label, "note": args.note, "change": change, "before": current,
                               "verdict": verdict, "problems": problems, "sizes": results},
                              indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
