"""Build the ~5,000-scheme search corpus: our checked schemes plus real background schemes.

    python scripts/build_scale_corpus.py

Background schemes come from a public copy of myScheme (data/external/myscheme_copy/
Schemes.csv, about 4,690 schemes). That copy was scraped from myscheme.gov.in
against its terms of use, so it is used ONLY as search background in tests:
it is kept out of git, never answered from, and its text is not checked. Credit:
scheme information from myScheme (myscheme.gov.in), National e-Governance Division.

Background schemes that are the same scheme as one of our checked records are
left out (listed in the report), so a question about PM-KISAN is not counted
as a miss when search finds myScheme's copy of PM-KISAN instead of ours.
Each background scheme is cut into the same sections as our records.
"""

import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import DOMAINS_DIR, ROOT_DIR
from engine.scheme_records import MAX_CHUNK_CHARS, load_records, record_chunks, _split

COPY = ROOT_DIR / "data" / "external" / "myscheme_copy" / "Schemes.csv"
OUTPUT = ROOT_DIR / "data" / "external" / "scale_corpus"
SECTIONS = (("eligibility_text", "Eligibility rules", "rules"), ("benefits", "Benefits", "benefits"),
            ("application_process", "How to apply", "application"), ("documents_required", "Documents needed", "documents"))


def clean(text: str) -> list[str]:
    """Plain statements from myScheme's markdown/HTML text, one per line."""
    text = re.sub(r"<[^>]+>", "\n", str(text))
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[*_`#>|]+", " ", line)
        line = re.sub(r"^\s*(?:[-•]|\d+[.)])\s*", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) > 2 and line.lower() not in ("null", "none", "nan"):
            lines.append(line)
    return lines


def name_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def same_scheme(row, ours: dict[str, set[str]]) -> str | None:
    """The id of our record this background scheme duplicates, by full name or short name."""
    name = name_key(row["name"])
    for rid, names in ours.items():
        if name in names or any(len(n) > 12 and n in name for n in names):
            return rid
    return None


def background_chunks(row) -> list[dict]:
    sid = f"ms__{row['slug']}"
    full = re.sub(r"\s+", " ", str(row["name"]).replace('"', "")).strip()
    where = (f"State scheme of {row['state']}" if row["state"] not in ("", "null", "Central") else "Central government scheme")
    owner = row["department"] if row["department"] not in ("", "null") else row["ministry"]
    chunks = [{"chunk_id": f"{sid}__overview", "scheme_id": sid, "section": "overview", "title": f"{full} — Overview",
               "source": row["official_url"], "text": " ".join(clean(row["description"]))[:MAX_CHUNK_CHARS] +
               f"\n{where}. {owner}."}]
    for column, heading, label in SECTIONS:
        parts = _split([f"- {s}" for s in clean(row[column])], MAX_CHUNK_CHARS)
        for n, part in enumerate(parts, 1):
            suffix = "" if len(parts) == 1 else f" (part {n} of {len(parts)})"
            chunks.append({"chunk_id": f"{sid}__{label}" + ("" if n == 1 else f"_{n}"), "scheme_id": sid,
                           "section": label, "title": f"{full} — {heading}{suffix}", "source": row["official_url"],
                           "text": "\n".join(s[:MAX_CHUNK_CHARS] for s in part)})
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", default="india_schemes")
    parser.add_argument("--verified-only", action="store_true", help="Leave out records marked needs_review")
    args = parser.parse_args()
    if not COPY.is_file():
        sys.exit(f"Missing {COPY}; see the docstring for where it comes from")
    records = [r for r in load_records(DOMAINS_DIR / args.domain / "schemes")
               if not args.verified_only or r["review"]["status"] == "verified"]
    ours = {r["id"]: {name_key(n) for n in [r["name"], r.get("short_name") or "", *(r.get("aliases") or [])] if n}
            for r in records}
    rows = pd.read_csv(COPY, keep_default_na=False)
    chunks = [c for r in records for c in record_chunks(r)]
    duplicates, kept = {}, 0
    for _, row in rows.iterrows():
        twin = same_scheme(row, ours)
        if twin:
            duplicates.setdefault(twin, []).append(str(row["name"]).strip(' "'))
            continue
        chunks += background_chunks(row)
        kept += 1
    seen, unique = set(), []
    for chunk in chunks:                     # the copy repeats a few slugs
        if chunk["chunk_id"] not in seen:
            seen.add(chunk["chunk_id"])
            unique.append(chunk)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "corpus.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in unique), encoding="utf-8")
    report = {"checked_schemes": len(records), "background_schemes": kept, "schemes": len(records) + kept,
              "sections": len(unique), "left_out_as_duplicates_of_ours": duplicates,
              "corpus": str(OUTPUT / "corpus.jsonl")}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "left_out_as_duplicates_of_ours"}, indent=2))
    print("left out as duplicates:", {k: len(v) for k, v in duplicates.items()})


if __name__ == "__main__":
    main()
