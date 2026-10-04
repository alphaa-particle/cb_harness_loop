"""Re-score saved audits with the two evaluator bugs fixed.

Background (see docs/results/FINAL_REPORT.md section 7):

The follow-up scorer in engine/evaluator.py under-counts correct behaviour in
two ways, and both cost real systems large amounts of credit:

  Artifact A — it only credits sentences ending in "?", so a correct
    "Please share your age, monthly income, worker_type, epfo."
    scores 0.0 despite asking for exactly the right fields.

  Artifact B — domains/welfare_demo/aliases.yaml lists "worker type" (space)
    but assistants write "worker_type" (underscore). _word_pattern turns the
    space into \\s+, so the alias never matches.

This script re-scores the SAVED answers with those two fixes. It does not
re-run any model, and it does not modify the audits. Weights, gates and the
pass threshold are taken from the domain pack unchanged, so the only thing
that differs from the original run is follow-up detection.

Usage (from the synalign/ directory):

    python scripts/rescore_followup_fix.py
    python scripts/rescore_followup_fix.py --runs audit_qwen_rule_trigger_fix_500

This is a diagnostic, not a replacement for fixing engine/evaluator.py and
re-running the audits properly.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import OUTPUT_DIR
from engine.domain_pack import DomainPack
from engine.evaluator import Evaluator, _sentences, _word_pattern

# Imperative ways of asking for a missing fact. The original scorer credits
# none of these, because none of them end in "?".
_REQUEST = re.compile(
    r"\b(please\s+(share|provide|tell|confirm|specify)"
    r"|could you|can you|let me know|i need to know"
    r"|kindly\s+(share|provide))\b",
    re.IGNORECASE,
)

DEFAULT_RUNS = [
    "audit_qwen",
    "audit_qwen_500_cases",
    "audit_qwen_router_rule_fix_100",
    "audit_qwen_router_rule_fix_500",
    "audit_qwen_rule_trigger_fix_500",
    "audit_qwen_lora_sft_500",
]

CONDITIONS = ["clean", "typo_heavy", "missing_info", "vague", "misleading"]


def build_field_patterns(pack, fix_alias: bool):
    """Field alias patterns, optionally including snake_case spellings."""
    evaluator_patterns = {}
    for field_id, aliases in pack.field_aliases().items():
        forms = set(aliases)
        if fix_alias:
            # the bare field id ("worker_type"), its spaced form, and the
            # underscored form of every multi-word alias
            forms.add(field_id)
            forms.add(field_id.replace("_", " "))
            forms.update(a.replace(" ", "_") for a in aliases if " " in a)
        evaluator_patterns[field_id] = [_word_pattern(a) for a in forms]
    return evaluator_patterns


def followup(text, must_ask, patterns, fix_phrasing: bool):
    if not must_ask:
        return 1.0
    candidates = [
        s for s in _sentences(text)
        if s.endswith("?") or (fix_phrasing and _REQUEST.search(s))
    ]
    if not candidates:
        return 0.0
    asked = sum(
        1 for field_id in must_ask
        if any(p.search(s) for s in candidates for p in patterns.get(field_id, []))
    )
    return asked / len(must_ask)


def rescore(records, weights, pass_score, patterns, fix_phrasing, fix_alias):
    """Return (total_passed, {condition: [n, passed]}) under the fixed scorer."""
    passed = 0
    by_condition = {}
    for rec in records:
        ev = rec["evaluation"]
        fol = followup(rec["answer"], rec["ground_truth"]["must_ask_about"],
                       patterns, fix_phrasing)
        overall = (
            weights["coverage"] * ev["coverage_score"]
            + weights["followup"] * fol
            + weights["groundedness"] * ev["groundedness_score"]
            + weights["actionability"] * ev["actionability_score"]
        )
        ok = (overall >= pass_score) and not ev["gate_violations"]
        passed += ok
        slot = by_condition.setdefault(ev["condition"], [0, 0])
        slot[0] += 1
        slot[1] += ok
    return passed, by_condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS,
                    help="audit labels to re-score (without the .jsonl suffix)")
    args = ap.parse_args()

    pack = DomainPack("welfare_demo")
    evaluator = Evaluator(pack)
    weights, pass_score = evaluator.weights, evaluator.pass_score

    modes = [
        ("as-measured", False, False),
        ("+phrasing", True, False),
        ("+phrasing+alias", True, True),
    ]
    patterns = {
        False: build_field_patterns(pack, fix_alias=False),
        True: build_field_patterns(pack, fix_alias=True),
    }

    print(f"{'run':<34}" + "".join(f"{name:>18}" for name, _, _ in modes))
    detail = {}

    for label in args.runs:
        path = OUTPUT_DIR / f"audit_{label}.jsonl" if not label.startswith("audit_") \
            else OUTPUT_DIR / f"{label}.jsonl"
        if not path.exists():
            print(f"{label:<34}  (missing: {path})")
            continue

        records = [json.loads(line) for line in open(path, encoding="utf-8")]
        n = len(records)
        cells, detail[label] = [], {}

        for name, fix_phrasing, fix_alias in modes:
            if name == "as-measured":
                count = sum(1 for r in records if r["evaluation"]["passed"])
                by_cond = {}
                for r in records:
                    ev = r["evaluation"]
                    slot = by_cond.setdefault(ev["condition"], [0, 0])
                    slot[0] += 1
                    slot[1] += ev["passed"]
            else:
                count, by_cond = rescore(records, weights, pass_score,
                                         patterns[fix_alias], fix_phrasing, fix_alias)
            cells.append(f"{count}/{n} ({count / n * 100:.1f}%)")
            detail[label][name] = by_cond

        print(f"{label:<34}" + "".join(f"{c:>18}" for c in cells))

    print()
    for label, by_mode in detail.items():
        print(f"-- {label} by condition")
        print(f"   {'condition':<14}" + "".join(f"{name:>17}" for name, _, _ in modes))
        for cond in CONDITIONS:
            if cond not in by_mode[modes[0][0]]:
                continue
            row = []
            for name, _, _ in modes:
                n, p = by_mode[name][cond]
                row.append(f"{p}/{n}")
            print(f"   {cond:<14}" + "".join(f"{v:>17}" for v in row))
        print()


if __name__ == "__main__":
    main()
