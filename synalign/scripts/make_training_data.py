"""Build fine-tuning data from an audit file.

    python scripts/make_training_data.py --input data/outputs/audit_baseline.jsonl \
        --output-dir data/training_new

Writes sft_train.jsonl, sft_dev.jsonl and sft_test_reference.jsonl, keeping the
audit's own train/dev/test split. Each example is the exact prompt the assistant
builds (system text + gold evidence + question) with the ideal answer from the
ground truth. Add --preference to also write preference_train.jsonl: for failed
train cases, the assistant's own answer paired against the ideal one.

No model is needed. Review the targets before training on them.
"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import ACTIVE_DOMAIN
from engine.domain_pack import DomainPack
from engine.retriever import Retriever
from engine.training_data import make_example, make_preference_pair, scheme_names

FILES = {"train": "sft_train.jsonl", "dev": "sft_dev.jsonl", "test": "sft_test_reference.jsonl"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Audit JSONL (questions, ground truth, answers)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", default=ACTIVE_DOMAIN)
    parser.add_argument("--corpus", type=Path, help="Corpus holding the gold evidence, if not the domain's own")
    parser.add_argument("--preference", action="store_true", help="Also write preference_train.jsonl")
    args = parser.parse_args()
    pack = DomainPack(args.domain, corpus_path=args.corpus)
    retriever = Retriever(pack)
    names = scheme_names(pack, retriever)
    examples, pairs = [], []
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] not in FILES:
                raise ValueError(f"Invalid split for {row.get('case_id')}")
            example = make_example(row, pack, retriever, names)
            examples.append(example)
            pair = make_preference_pair(row, example) if row["split"] == "train" else None
            if pair:
                pairs.append(pair)
    # Everything above is validated before any output file is opened.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(FILES, 0)
    with ExitStack() as stack:
        writers = {split: stack.enter_context((args.output_dir / filename).open("w", encoding="utf-8"))
                   for split, filename in FILES.items()}
        for example in examples:
            writers[example["split"]].write(json.dumps(example, ensure_ascii=False) + "\n")
            counts[example["split"]] += 1
    if args.preference:
        with (args.output_dir / "preference_train.jsonl").open("w", encoding="utf-8") as stream:
            stream.writelines(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs)
        counts["preference_pairs"] = len(pairs)
    print(json.dumps({"output_dir": str(args.output_dir), "counts": counts,
                      "corpus_sha256": retriever.corpus_fingerprint}, indent=2))


if __name__ == "__main__":
    main()
