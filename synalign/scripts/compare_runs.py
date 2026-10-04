"""Keep or undo a change: compare one condition between two evaluation runs, question by question.

    python scripts/compare_runs.py data/evaluation/india_schemes/dev_round0 \
        data/evaluation/india_schemes/dev_round1 --run fusion
    python scripts/compare_runs.py <before> <after> --same-model     # a search change

Both folders must come from scripts/evaluate_pipeline.py on the same question
file. Only questions answered in both are compared, so the difference is the
change's own effect, not a different mix of questions.

Verdict, decided by fixed rules written down before looking:
    KEEP                 the interval for (after - before) is above zero and unsafe "yes"
                         answers did not increase
    UNDO                 the interval is below zero, or unsafe "yes" answers increased
    NO CLEAR DIFFERENCE  anything else: the questions cannot tell the two apart

With --same-model (the change touched search only, not the model or its instructions),
the end-to-end check for search rounds also applies (see scripts/search_round.py):
    INVALID              a question whose evidence did not change got a different answer:
                         the runs are not comparable (the model was not deterministic)
    UNDO                 also when any question type or language has fewer correct answers,
                         or a not-found question that was refused is now answered
Search changes themselves are judged with scripts/search_round.py.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_pipeline import group_ci


def load(folder: Path, run: str) -> tuple[dict, tuple]:
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (folder / f"answers_{run}.jsonl").read_text(encoding="utf-8").splitlines()]
    same = (report["questions_sha256"], report.get("split"), Path(report.get("corpus", "")).name)
    return {r["query_id"]: r for r in rows}, same


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--run", default="fusion")
    parser.add_argument("--same-model", action="store_true",
                        help="the change touched search only: also check determinism and every slice")
    args = parser.parse_args()
    before, sha_before = load(args.before, args.run)
    after, sha_after = load(args.after, args.run)
    if sha_before != sha_after:
        sys.exit("The two runs used different question files, splits or corpora; they cannot be compared "
                 f"question by question ({sha_before} vs {sha_after}).")
    ids = sorted(set(before) & set(after))
    if not ids:
        sys.exit("The two runs share no questions.")
    diffs = [after[i]["correct"] - before[i]["correct"] for i in ids]
    groups = [before[i]["group_id"] for i in ids]
    lo, hi = group_ci(diffs, groups)
    unsafe_before = sum(before[i].get("unsafe_yes", 0) for i in ids)
    unsafe_after = sum(after[i].get("unsafe_yes", 0) for i in ids)
    fixed = [i for i in ids if after[i]["correct"] > before[i]["correct"]]
    broken = [i for i in ids if after[i]["correct"] < before[i]["correct"]]
    guards = []
    if args.same_model:
        unsteady = [i for i in ids if before[i].get("evidence") == after[i].get("evidence")
                    and before[i].get("answer") != after[i].get("answer")]
        changed = sum(before[i].get("evidence") != after[i].get("evidence") for i in ids)
        print(f"evidence changed for {changed} questions; answers changed with unchanged evidence: {len(unsteady)}")
        if unsteady:
            print(f"VERDICT: INVALID (for example {unsteady[0]})")
            return
        for field in ("type", "language"):
            for value in sorted({before[i][field] for i in ids}):
                sub = [i for i in ids if before[i][field] == value]
                b, a = sum(before[i]["correct"] for i in sub), sum(after[i]["correct"] for i in sub)
                if a < b:
                    guards.append(f"{field}={value}: correct {b:.0f} -> {a:.0f}")
        reopened = [i for i in ids if before[i]["type"] in ("absent", "off_topic")
                    and before[i]["correct"] == 1 and after[i]["correct"] == 0]
        if reopened:
            guards.append(f"not-found questions now answered: {', '.join(reopened)}")
    for line in guards:
        print(f"  guard failed: {line}")
    if unsafe_after > unsafe_before or hi < 0 or guards:
        verdict = "UNDO"
    elif lo > 0:
        verdict = "KEEP"
    else:
        verdict = "NO CLEAR DIFFERENCE"
    by_type = {}
    for i in ids:
        t = before[i]["type"]
        by_type.setdefault(t, [0, 0, 0])
        by_type[t][0] += 1
        by_type[t][1] += before[i]["correct"]
        by_type[t][2] += after[i]["correct"]
    print(f"{args.run}: {len(ids)} questions in both runs")
    print(f"correct before {sum(before[i]['correct'] for i in ids):.0f}, after {sum(after[i]['correct'] for i in ids):.0f}; "
          f"fixed {len(fixed)}, broken {len(broken)}")
    print(f"difference {sum(diffs) / len(ids) * 100:+.1f} points (95% interval {lo * 100:+.1f} to {hi * 100:+.1f})")
    print(f"unsafe 'yes' answers before {unsafe_before:.0f}, after {unsafe_after:.0f}")
    for t, (n, b, a) in sorted(by_type.items()):
        print(f"  {t:12s} {n:4d} questions  {b / n * 100:5.1f}% -> {a / n * 100:5.1f}%")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
