"""Start the two local model servers and keep them running until Ctrl+C.

    python scripts/serve_models.py              # embedder on :8091, answer model on :8092
    python scripts/serve_models.py --only embedder
    python scripts/serve_models.py --detach     # start them in the background and return
    python scripts/serve_models.py --stop       # stop servers started with --detach

Needs llama.cpp (brew install llama.cpp) and the model files in models/:
    Qwen3-Embedding-0.6B-Q8_0.gguf   from huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF
    Qwen3.5-0.8B-Q8_0.gguf           from huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF
Server logs are written next to the model files (models/embedder.log, models/decoder.log).
"""

import argparse
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.llama_cpp import decoder_server, embedder_server


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["embedder", "decoder"])
    parser.add_argument("--detach", action="store_true", help="Leave the servers running and return")
    parser.add_argument("--stop", action="store_true", help="Stop servers started with --detach")
    args = parser.parse_args()
    servers = [s for name, s in (("embedder", embedder_server()), ("decoder", decoder_server()))
               if args.only in (None, name)]
    if args.stop:
        for server in servers:
            server.stop_detached()
        return
    if args.detach:
        for server in servers:
            server.start(detach=True)
            print(f"ready: {server.model_path.name} at {server.url} (running in the background)")
        return
    try:
        for server in servers:
            server.start()
            print(f"ready: {server.model_path.name} at {server.url}", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.stop()


if __name__ == "__main__":
    main()
