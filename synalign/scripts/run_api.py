"""Serve the chatbot: the web page at /ui and the API, on this machine only.

    python scripts/run_api.py                                   # the Indian schemes chatbot with the local model
    python scripts/run_api.py --backend rules_only              # no answer model: show the matching rules
    python scripts/run_api.py --domain welfare_demo --backend naive

With the llama_cpp backend, or a search method that uses meaning vectors, the
local model servers are started first (see scripts/serve_models.py) and stopped
again when this stops. Open http://127.0.0.1:8000/ui in a browser.
"""

import argparse
import os
from pathlib import Path
import sys

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", default=os.environ.get("SYNALIGN_DOMAIN", "india_schemes"),
                        help="Domain pack (default india_schemes)")
    parser.add_argument("--backend", default=os.environ.get("SYNALIGN_ASSISTANT_BACKEND", "llama_cpp"),
                        help="naive, rules_only, llama_cpp (default), transformers or ollama")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # Settings are read when the engine is imported, so set them first.
    os.environ["SYNALIGN_DOMAIN"] = args.domain
    os.environ["SYNALIGN_ASSISTANT_BACKEND"] = args.backend
    from engine.config import ACTIVE_DOMAIN, ASSISTANT_BACKEND
    from engine.domain_pack import DomainPack
    from engine.llama_cpp import decoder_server, embedder_server
    from engine.retriever import DENSE_METHODS

    servers = []
    if DomainPack(ACTIVE_DOMAIN).eval_config["retrieval"].get("method") in DENSE_METHODS:
        servers.append(embedder_server())
    if ASSISTANT_BACKEND == "llama_cpp":
        servers.append(decoder_server())
    try:
        for server in servers:
            server.start()
            print(f"ready: {server.model_path.name} at {server.url}", flush=True)
        print(f"Open http://127.0.0.1:{args.port}/ui", flush=True)
        uvicorn.run("engine.api:app", host="127.0.0.1", port=args.port, workers=1)
    finally:
        for server in servers:
            server.stop()


if __name__ == "__main__":
    main()
