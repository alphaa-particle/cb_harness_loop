"""Scheme records: one editable YAML file per scheme, checked, then cut into search chunks.

A record holds what a scheme is, who qualifies, what it gives and how to apply,
each statement in plain words, together with the official pages it came from
and whether it has been checked. Records are the source of truth; the search
corpus (corpus.jsonl) is always rebuilt from them and never edited by hand.

    load_records(folder)      read every *.yaml
    validate_records(records) errors (must fix) and warnings (should look at)
    record_chunks(record)     the search chunks for one scheme
    build_corpus(...)         write corpus.jsonl and the question files
"""

from datetime import date
import json
from pathlib import Path
import re

import yaml

CATEGORIES = ("pension", "insurance", "health", "housing", "agriculture", "women_child", "education",
              "skills_employment", "credit_enterprise", "food", "energy", "social_assistance",
              "financial_inclusion", "other")
REVIEW_STATUSES = ("verified", "needs_review")
FACT_STATUSES = ("verified", "unverified", "disputed")
# section key in the record -> (heading shown to the model, label stored on the chunk)
SECTIONS = (("eligibility", "Eligibility rules", "rules"),
            ("benefits", "Benefits", "benefits"),
            ("how_to_apply", "How to apply", "application"),
            ("documents", "Documents needed", "documents"))
RULE_KEYS = ("age_min", "age_max", "annual_income_max", "monthly_income_max")
GENDERS = ("any", "female", "male")
MAX_CHUNK_CHARS = 1200   # about 300 tokens: three chunks fit a small model's prompt comfortably
_ID = re.compile(r"^[a-z][a-z0-9_]{1,60}$")


def load_records(folder: str | Path) -> list[dict]:
    """Every scheme file in the folder, each tagged with the file it came from."""
    records = []
    for path in sorted(Path(folder).glob("*.yaml")):
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"{path.name}: expected a mapping at the top level")
        record["_file"] = path.name
        records.append(record)
    if not records:
        raise ValueError(f"No scheme files (*.yaml) found in {folder}")
    return records


def _plain(text: str) -> str:
    """Lower-case with digit-group commas removed, so 15,000 and 15000 compare equal."""
    return re.sub(r"(?<=\d),(?=\d)", "", str(text)).lower()


def _strings(value) -> list[str]:
    return value if isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value) else []


def record_text(record: dict) -> str:
    """All statements of a record as one string, for checking that facts are really in it."""
    sections = record.get("sections") or {}
    parts = [record.get("summary") or ""]
    for key in ("eligibility", "exclusions", "benefits", "how_to_apply", "documents"):
        parts += _strings(sections.get(key))
    return _plain("\n".join(parts))


def validate_record(record: dict) -> tuple[list[str], list[str]]:
    """Problems with one record: (errors, warnings), each a plain sentence."""
    errors, warnings = [], []
    name = record.get("_file", "?")

    def need(condition, message):
        if not condition:
            errors.append(f"{name}: {message}")

    rid = record.get("id")
    need(isinstance(rid, str) and _ID.match(rid or ""), "id must be lower-case letters, digits and underscores")
    need(name == "?" or name == f"{rid}.yaml", f"the file must be named {rid}.yaml")
    for key in ("name", "ministry", "summary"):
        need(isinstance(record.get(key), str) and record[key].strip(), f"{key} is required")
    need(record.get("category") in CATEGORIES, f"category must be one of {', '.join(CATEGORIES)}")
    need(record.get("level") in ("central", "state"), "level must be central or state")
    need(record.get("level") != "state" or record.get("state"), "a state scheme needs its state")
    for key in ("names_hi", "aliases"):
        need(key not in record or record[key] == [] or _strings(record[key]), f"{key} must be a list of names")

    sections = record.get("sections") or {}
    need(_strings(sections.get("eligibility")), "sections.eligibility needs at least one statement")
    for key in ("exclusions", "benefits", "how_to_apply", "documents"):
        need(key not in sections or sections[key] in (None, []) or _strings(sections[key]),
             f"sections.{key} must be a list of statements")
    if not _strings(sections.get("benefits")):
        warnings.append(f"{name}: no benefits listed")

    sources = record.get("sources") or []
    need(isinstance(sources, list) and sources, "at least one source is required")
    for i, source in enumerate(sources if isinstance(sources, list) else [], 1):
        ok = isinstance(source, dict) and str(source.get("url", "")).startswith("https://")
        need(ok, f"source {i} needs an https url")
        need(ok and source.get("title"), f"source {i} needs a title")
        try:
            date.fromisoformat(str(source.get("accessed")) if isinstance(source, dict) else "")
        except ValueError:
            errors.append(f"{name}: source {i} needs an accessed date like 2026-10-03")

    text = record_text(record)
    eligibility_text = _plain("\n".join(_strings(sections.get("eligibility")) + _strings(sections.get("exclusions"))))
    rules = record.get("rules") or {}
    need(isinstance(rules, dict), "rules must be a mapping")
    if isinstance(rules, dict):
        for key, value in rules.items():
            if key == "gender":
                need(value in GENDERS, f"rules.gender must be one of {', '.join(GENDERS)}")
            elif key in RULE_KEYS:
                need(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                     f"rules.{key} must be a whole number")
                # The number a test is graded against must be one the written rules actually state.
                need(not isinstance(value, int) or re.search(rf"(?<!\d){value}(?!\d)", eligibility_text),
                     f"rules.{key} is {value} but that number is not in the eligibility statements")
            else:
                errors.append(f"{name}: rules.{key} is not a known rule ({', '.join(RULE_KEYS)}, gender)")
        if isinstance(rules.get("age_min"), int) and isinstance(rules.get("age_max"), int):
            need(rules["age_min"] <= rules["age_max"], "rules.age_min is above rules.age_max")

    facts = record.get("facts") or []
    need(isinstance(facts, list), "facts must be a list")
    seen = set()
    for i, fact in enumerate(facts if isinstance(facts, list) else [], 1):
        if not isinstance(fact, dict):
            errors.append(f"{name}: fact {i} must be a mapping")
            continue
        label = f"fact {fact.get('key', i)}"
        need(isinstance(fact.get("key"), str) and fact["key"] not in seen, f"{label} needs a unique key")
        seen.add(fact.get("key"))
        for key in ("statement", "question"):
            need(isinstance(fact.get(key), str) and fact[key].strip(), f"{label} needs a {key}")
        need(fact.get("status") in FACT_STATUSES, f"{label} status must be one of {', '.join(FACT_STATUSES)}")
        need(isinstance(fact.get("source"), int) and isinstance(sources, list) and 1 <= fact["source"] <= len(sources),
             f"{label} must point at one of the listed sources (1 to {len(sources) if isinstance(sources, list) else 0})")
        value = str(fact.get("value", "")).strip()
        need(value, f"{label} needs a value")
        # A fact that is not in the scheme's own text could never be answered from the documents.
        need(not value or _plain(value) in text, f"{label} value {value!r} does not appear in the scheme's statements")

    review = record.get("review") or {}
    status = review.get("status") if isinstance(review, dict) else None
    need(status in REVIEW_STATUSES, f"review.status must be one of {', '.join(REVIEW_STATUSES)}")
    if status == "verified":
        need(review.get("checked_on"), "a verified record needs review.checked_on")
        need(all(isinstance(f, dict) and f.get("status") == "verified" for f in facts if isinstance(facts, list)),
             "a verified record cannot contain unverified or disputed facts")
    elif status == "needs_review":
        warnings.append(f"{name}: marked needs_review" + (f" ({review['notes']})" if review.get("notes") else ""))
    return errors, warnings


def validate_records(records: list[dict]) -> tuple[list[str], list[str]]:
    """Problems across the whole set, including clashes between schemes."""
    errors, warnings = [], []
    ids, names = {}, {}
    for record in records:
        e, w = validate_record(record)
        errors += e
        warnings += w
        rid = record.get("id")
        if rid in ids:
            errors.append(f"{record.get('_file')}: id {rid!r} is also used by {ids[rid]}")
        ids[rid] = record.get("_file")
        for label in {record.get("name"), record.get("short_name"), *_strings(record.get("aliases")),
                      *_strings(record.get("names_hi"))} - {None, ""}:
            key = label.strip().lower()
            if key in names and names[key] != rid:
                warnings.append(f"{record.get('_file')}: the name {label!r} is also used by {names[key]}")
            names.setdefault(key, rid)
    return errors, warnings


def _split(statements: list[str], limit: int) -> list[list[str]]:
    """Group statements into parts no longer than `limit`; a statement is never cut in two."""
    parts, current, size = [], [], 0
    for statement in statements:
        line = len(statement) + 3
        if current and size + line > limit:
            parts.append(current)
            current, size = [], 0
        current.append(statement)
        size += line
    return parts + ([current] if current else [])


def record_chunks(record: dict, max_chars: int = MAX_CHUNK_CHARS) -> list[dict]:
    """The search chunks for one scheme: an overview, then one or more per section.

    Every chunk carries the scheme's full name in its title and all its other
    names as aliases, so a chunk still says which scheme it belongs to when it
    is read on its own, by the search and by the model.
    """
    rid, sections = record["id"], record.get("sections") or {}
    short = record.get("short_name")
    full_name = f"{record['name']} ({short})" if short and short != record["name"] else record["name"]
    names = [n for n in dict.fromkeys([short, *_strings(record.get("names_hi")), *_strings(record.get("aliases"))])
             if n and n != record["name"]]
    source = ((record.get("sources") or [{}])[0]).get("url", "")
    where = "Central government scheme" if record["level"] == "central" else f"State scheme of {record['state']}"
    chunks = [{"chunk_id": f"{rid}__overview", "scheme_id": rid, "section": "overview",
               "title": f"{full_name} — Overview", "aliases": names, "source": source,
               "text": f"{record['summary'].strip()}\n{where}. {record['ministry']}."}]
    for key, heading, label in SECTIONS:
        statements = _strings(sections.get(key))
        blocks = [f"- {s.strip()}" for s in statements]
        if key == "eligibility" and _strings(sections.get("exclusions")):
            # A rule and its exceptions stay together, so the model never sees one without the other.
            blocks += ["Not eligible:"] + [f"- {s.strip()}" for s in _strings(sections["exclusions"])]
        parts = _split(blocks, max_chars)
        for n, part in enumerate(parts, 1):
            suffix = "" if len(parts) == 1 else f" (part {n} of {len(parts)})"
            chunks.append({"chunk_id": f"{rid}__{key}" + ("" if n == 1 else f"_{n}"), "scheme_id": rid,
                           "section": label, "title": f"{full_name} — {heading}{suffix}",
                           "aliases": names, "source": source, "text": "\n".join(part)})
    return chunks


def record_questions(record: dict) -> list[dict]:
    """Checkable questions from a record's verified facts, labelled with where the answer is."""
    chunk_ids = {c["chunk_id"]: _plain(c["text"]) for c in record_chunks(record)}
    questions = []
    for fact in record.get("facts") or []:
        if fact.get("status") != "verified":
            continue
        value = _plain(fact["value"])
        gold = [cid for cid, text in chunk_ids.items() if value in text]
        base = {"scheme_id": record["id"], "fact": fact["key"], "answer": str(fact["value"]),
                "accept": [str(a) for a in fact.get("accept", [])], "gold_chunk_ids": gold}
        questions.append({**base, "query_id": f"{record['id']}__{fact['key']}__en", "language": "en",
                          "question": fact["question"]})
        if fact.get("question_hi"):
            questions.append({**base, "query_id": f"{record['id']}__{fact['key']}__hi", "language": "hi",
                              "question": fact["question_hi"]})
    return questions


def build_corpus(folder: str | Path, output: str | Path, include_unverified: bool = False) -> dict:
    """Validate the records, then write corpus.jsonl and fact_questions.jsonl into `output`."""
    records = load_records(folder)
    errors, warnings = validate_records(records)
    if errors:
        raise ValueError("Scheme records have errors:\n  " + "\n  ".join(errors))
    used = [r for r in records if include_unverified or r["review"]["status"] == "verified"]
    chunks = [chunk for record in used for chunk in record_chunks(record)]
    questions = [q for record in used for q in record_questions(record)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "corpus.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    (output / "fact_questions.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
    # The names the grader accepts for each scheme, so it can tell which schemes an answer mentions.
    entities = {r["id"]: list(dict.fromkeys(n for n in [r["name"], r.get("short_name"), *_strings(r.get("aliases")),
                                                        *_strings(r.get("names_hi"))] if n)) for r in used}
    (output / "entities.json").write_text(json.dumps(entities, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"records": len(records), "used": len(used), "left_out_needs_review": len(records) - len(used),
            "chunks": len(chunks), "fact_questions": len(questions), "warnings": warnings,
            "longest_chunk_chars": max(len(c["text"]) for c in chunks) if chunks else 0}
