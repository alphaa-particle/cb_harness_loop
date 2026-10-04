"""Search the scheme sections, or show the prompt a question would produce. No answer model is used.

    python scripts/search_corpus.py --query "Who can join Atal Pension Yojana?"
    python scripts/search_corpus.py --query "budhape ki pension" --context
    python scripts/search_corpus.py --corpus corpus.jsonl --method char_tfidf --query "..."

By default this searches the domain's own sections with the domain's own search
settings (eval_config.yaml), so it shows what the chatbot would do. Meaning-based
methods (dense, fusion) need the embedding server: python scripts/serve_models.py.

Without --context: the sections most similar to the question. With it: the
sections actually sent to the model (each scheme's rules first) and the full
prompt built from them.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.domain_pack import DomainPack
from engine.enforcement import build_messages
from engine.retriever import METHODS, Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", default=os.environ.get("SYNALIGN_DOMAIN", "india_schemes"))
    parser.add_argument("--corpus", default=os.environ.get("SYNALIGN_CORPUS_PATH"),
                        help="Search this corpus instead of the domain's own sections")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, help="Default: the domain's setting (SYNALIGN_RETRIEVAL_TOP_K)")
    parser.add_argument("--method", choices=METHODS, help="Default: the domain's setting (SYNALIGN_RETRIEVAL_METHOD)")
    parser.add_argument("--min-score", type=float, help="Default: the domain's setting (SYNALIGN_RETRIEVAL_MIN_SCORE)")
    parser.add_argument("--context", action="store_true", help="Show the evidence and prompt the model would get")
    args = parser.parse_args()
    pack = DomainPack(args.domain, corpus_path=Path(args.corpus) if args.corpus else None)
    chosen = {"top_k": args.top_k, "method": args.method, "min_score": args.min_score}
    pack.eval_config["retrieval"].update({k: v for k, v in chosen.items() if v is not None})
    retriever = Retriever(pack)
    report = {"corpus_sha256": retriever.corpus_fingerprint, "chunks": len(retriever.chunk_ids),
              "retrieval": pack.eval_config["retrieval"]}
    if args.context:
        chunks = retriever.retrieve_evidence(args.query)
        report.update(evidence=[asdict(c) for c in chunks],
                      messages=build_messages(pack, args.query, chunks) if chunks else [])
    else:
        report["results"] = [asdict(c) for c in retriever.retrieve(args.query)]
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
