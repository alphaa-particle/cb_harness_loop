"""Check the scheme records and rebuild the search corpus from them.

    python scripts/build_corpus.py                  # the india_schemes records
    python scripts/build_corpus.py --check          # only report problems, write nothing
    python scripts/build_corpus.py --verified-only  # leave out records marked needs_review

Records live one per file in domains/<domain>/schemes/. To correct a fact, edit
that scheme's file, then run this again; corpus.jsonl is never edited by hand.
Records marked needs_review are included by default (the owner chose to correct
them later); --verified-only builds a corpus from fully checked records alone.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import DOMAINS_DIR
from engine.scheme_records import build_corpus, load_records, validate_records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", default="india_schemes")
    parser.add_argument("--check", action="store_true", help="Report problems only")
    parser.add_argument("--verified-only", action="store_true", help="Leave out records marked needs_review")
    args = parser.parse_args()
    folder = DOMAINS_DIR / args.domain / "schemes"
    if args.check:
        errors, warnings = validate_records(load_records(folder))
        for line in warnings:
            print("warning:", line)
        for line in errors:
            print("ERROR:", line)
        print(f"{len(errors)} errors, {len(warnings)} warnings")
        sys.exit(1 if errors else 0)
    report = build_corpus(folder, DOMAINS_DIR / args.domain / "documents", include_unverified=not args.verified_only)
    for line in report.pop("warnings"):
        print("warning:", line)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
