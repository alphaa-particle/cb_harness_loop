"""Turn checked scheme drafts into the editable scheme files.

    python scripts/import_scheme_drafts.py drafts.json        # writes domains/india_schemes/schemes/*.yaml

The input is the result of the corpus-building research: for each scheme a draft
record written from official pages, an independent check of every fact, rule
number and Hindi name, and (where the two disagreed) a third agent's decision.

A fact becomes "verified" only when an official page that someone actually opened
confirms it: the checker's, or the third agent's when they disagreed. A value the
third agent corrected from an official page is applied, and the change is noted.
Anything else stays "unverified" or "disputed", and the scheme is marked
needs_review, which keeps it out of the corpus until a person checks it.
Every decision is written into the scheme file's review notes, so it can be audited
and corrected by hand later.
"""

import argparse
import json
from pathlib import Path
import re
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import DOMAINS_DIR
from engine.scheme_records import RULE_KEYS, _plain, record_text, validate_record

CHECKED_ON = "2026-10-03"
METHOD = ("Written from official pages by one research agent; every fact, rule number and Hindi name "
          "re-checked against official pages by a second agent; disagreements decided by a third.")


class _Dumper(yaml.SafeDumper):
    """Readable YAML: long statements stay on one line, Hindi stays Hindi."""


def _str(dumper, value):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str)


def classify(item: str, record: dict):
    item = item.strip()
    facts = {f["key"] for f in record.get("facts", [])}
    low = item.lower()
    for prefix, kind in (("rule:", "rule"), ("rules.", "rule"), ("fact:", "fact"), ("facts.", "fact"),
                         ("statement:", "statement")):
        if low.startswith(prefix):
            return kind, item[len(prefix):].strip()
    if low.startswith("names_hi") or low.startswith("hindi"):
        return "names_hi", None
    if item in facts:
        return "fact", item
    if item in RULE_KEYS or item == "gender":
        return "rule", item
    return None, item


def replace_statement(record: dict, old_hint: str, new: str) -> bool:
    """Swap the section statement that starts with (or contains) old_hint for `new`."""
    hint = _plain(old_hint).strip(" .…")[:60]
    for key, statements in (record.get("sections") or {}).items():
        for i, statement in enumerate(statements or []):
            if hint and (_plain(statement).startswith(hint) or hint in _plain(statement)):
                statements[i] = new
                return True
    return False


def apply(record: dict, check: dict | None, decisions: dict) -> dict:
    notes, problems = [], False
    facts = {f["key"]: f for f in record.get("facts", [])}
    for fact in facts.values():
        fact["status"] = "unverified"
    if check is None:
        notes.append("No independent check was returned for this scheme.")
        problems = True
        items = []
    else:
        items = check.get("items", [])
        if check.get("still_open") == "no":
            notes.append("The checker found this scheme closed to new applicants.")
            problems = True
        elif check.get("still_open") == "unclear":
            notes.append("Whether the scheme is still open to new applicants was not clear.")
        notes += [f"Checker: {p}" for p in check.get("other_problems", [])]

    for it in items:
        kind, target = classify(it["item"], record)
        confirmed = it["verdict"] == "confirmed" and it.get("page_opened")
        decision = decisions.get(it["item"])
        corrected = None
        if not confirmed and decision and decision.get("page_opened"):
            if decision["decided"] == "writer_right":
                confirmed = True
            elif decision["decided"] in ("checker_right", "both_wrong"):
                corrected = decision
        # The checker opened an official page and read a different value there (no third look yet).
        checker_found = (not confirmed and not corrected and it["verdict"] == "wrong" and it.get("page_opened")
                         and it.get("correct_value", "").strip())
        if kind == "fact" and target in facts:
            fact = facts[target]
            if confirmed:
                fact["status"] = "verified"
            elif corrected:
                old = fact["value"]
                if corrected.get("corrected_statement"):
                    replace_statement(record, old, corrected["corrected_statement"])
                fact["value"] = corrected["final_value"]
                fact["status"] = "verified"
                notes.append(f"Fact {target} corrected from {old!r} to {corrected['final_value']!r} "
                             f"per {corrected['official_url']}.")
            else:
                fact["status"] = "disputed" if it["verdict"] == "wrong" else "unverified"
                notes.append(f"Fact {target} not confirmed on an opened official page ({it['verdict']}): {it.get('note', '')}".strip())
                problems = True
        elif kind == "rule" and target in (record.get("rules") or {}):
            if confirmed:
                continue
            if corrected and re.sub(r"[^\d]", "", corrected["final_value"]).isdigit() and target != "gender":
                old = record["rules"][target]
                record["rules"][target] = int(re.sub(r"[^\d]", "", corrected["final_value"]))
                if corrected.get("corrected_statement"):
                    replace_statement(record, str(old), corrected["corrected_statement"])
                notes.append(f"Rule {target} corrected from {old} to {record['rules'][target]} per {corrected['official_url']}.")
            elif checker_found:
                # Cannot be checked against one number: leave it out of the eligibility tests.
                notes.append(f"Rule {target}={record['rules'][target]} removed: the checker read "
                             f"{it['correct_value']!r} on {it['official_url']}.")
                del record["rules"][target]
                problems = True
            elif not corrected:
                notes.append(f"Rule {target} not confirmed on an opened official page: {it.get('note', '')}".strip())
                problems = True
        elif kind == "names_hi":
            if checker_found and re.search(r"[\u0900-\u097F]", it["correct_value"]):
                # The checker found the official Hindi name the writer could not.
                names = [n.strip(" .") for n in re.split(r"[;,]|\(", it["correct_value"])
                         if re.search(r"[\u0900-\u097F]", n)]
                record["names_hi"] = list(dict.fromkeys(names))
                notes.append(f"Official Hindi name added from {it['official_url']}.")
            elif not confirmed and record.get("names_hi"):
                # Not shown on an official page: still useful for search, but not as an official name.
                record["aliases"] = list(dict.fromkeys([*record.get("aliases", []), *record["names_hi"]]))
                record["names_hi"] = []
                notes.append("Hindi name not confirmed on an official page; kept only as a search alias.")
        elif kind == "statement" and target.lower().startswith("status"):
            if not confirmed:
                notes.append(f"Checker on the status note: {it.get('correct_value') or it.get('note', '')}")
        elif kind == "statement":
            if corrected and corrected.get("corrected_statement"):
                if replace_statement(record, target, corrected["corrected_statement"]):
                    notes.append(f"Statement corrected per {corrected['official_url']}: {corrected['corrected_statement']}")
            elif checker_found:
                notes.append(f"Statement to correct: {target!r}. The checker read {it['correct_value']!r} "
                             f"on {it['official_url']}.")
                problems = True
            elif not confirmed:
                notes.append(f"Statement not confirmed: {target} ({it.get('note', '')})")
                problems = True

    # Keep only what the written statements actually support.
    text = record_text(record)
    for key, fact in list(facts.items()):
        if _plain(str(fact["value"])) not in text:
            notes.append(f"Fact {key} dropped: its value {fact['value']!r} is not stated in the scheme's sections.")
            record["facts"] = [f for f in record["facts"] if f["key"] != key]
    eligibility = _plain("\n".join((record["sections"].get("eligibility") or []) + (record["sections"].get("exclusions") or [])))
    for key in list((record.get("rules") or {})):
        value = record["rules"][key]
        if key != "gender" and not re.search(rf"(?<!\d){value}(?!\d)", eligibility):
            notes.append(f"Rule {key}={value} dropped: the number is not in the eligibility statements.")
            del record["rules"][key]
    if any(f["status"] != "verified" for f in record.get("facts", [])):
        problems = True
    if record.get("status_note"):
        notes.insert(0, f"Writer: {record['status_note']}")
    record["review"] = {"status": "needs_review" if problems else "verified", "checked_on": CHECKED_ON,
                        "method": METHOD, "notes": notes}
    return record


def ordered(record: dict) -> dict:
    keys = ("id", "name", "short_name", "names_hi", "aliases", "level", "state", "ministry", "category", "summary",
            "sections", "rules", "facts", "situations", "sources", "review")
    return {k: record[k] for k in keys if k in record}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("drafts", type=Path, help="JSON list of {drafted, check, settle} per batch")
    parser.add_argument("--domain", default="india_schemes")
    args = parser.parse_args()
    batches = json.loads(args.drafts.read_text(encoding="utf-8"))
    folder = DOMAINS_DIR / args.domain / "schemes"
    folder.mkdir(parents=True, exist_ok=True)
    summary = {"verified": [], "needs_review": [], "invalid": []}
    for batch in batches:
        checks = {c["scheme_id"]: c for c in (batch.get("check") or {}).get("checks", [])}
        decisions = {}
        for d in (batch.get("settle") or {}).get("decisions", []):
            decisions.setdefault(d["scheme_id"], {})[d["item"]] = d
        for record in batch["drafted"]["records"]:
            record = apply(record, checks.get(record["id"]), decisions.get(record["id"], {}))
            record.pop("status_note", None)
            record["state"] = record.get("state") or ""
            errors, _ = validate_record({**record, "_file": f"{record['id']}.yaml"})
            if errors:
                record["review"]["status"] = "needs_review"
                record["review"]["notes"] += [f"Validation: {e.split(': ', 1)[-1]}" for e in errors]
                summary["invalid"].append(record["id"])
            (folder / f"{record['id']}.yaml").write_text(
                yaml.dump(ordered(record), Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=1000),
                encoding="utf-8")
            summary[record["review"]["status"]].append(record["id"])
    print(json.dumps({k: (len(v), v) for k, v in summary.items()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
