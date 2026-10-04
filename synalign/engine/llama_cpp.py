"""Local Qwen models served by llama.cpp's llama-server, reached over HTTP.

Two servers run on this machine, each with one quantised model file:

    embedder   Qwen3-Embedding-0.6B (Q8_0)   turns text into meaning vectors
    decoder    Qwen3.5-0.8B (Q8_0)            writes the answer

llama-server comes from Homebrew (`brew install llama.cpp`). This module only
uses the Python standard library, so nothing else has to be installed.
Start both servers with scripts/serve_models.py, or let LlamaServer do it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from engine.config import ROOT_DIR

MODELS_DIR = ROOT_DIR / "models"
EMBEDDER_FILE = "Qwen3-Embedding-0.6B-Q8_0.gguf"
DECODER_FILE = "Qwen3.5-0.8B-Q8_0.gguf"
MODEL_REPOS = {EMBEDDER_FILE: "Qwen/Qwen3-Embedding-0.6B-GGUF", DECODER_FILE: "ggml-org/Qwen3.5-0.8B-GGUF"}
EMBEDDER_URL = os.environ.get("SYNALIGN_EMBEDDER_URL", "http://127.0.0.1:8091")
DECODER_URL = os.environ.get("SYNALIGN_DECODER_URL", "http://127.0.0.1:8092")
# Written in English even for Hindi questions, as the model card advises.
QUERY_INSTRUCTION = ("Given a question about Indian government welfare schemes, "
                     "retrieve the scheme passage that answers it")


def _request(url: str, payload: dict | None = None, timeout: float = 60) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise OSError(f"{url} answered {exc.code}: {detail}") from exc
    except ValueError as exc:            # the server sent something that is not JSON
        raise OSError(f"{url} sent an unreadable reply") from exc


def is_ready(url: str) -> bool:
    try:
        return _request(f"{url}/health", timeout=2).get("status") == "ok"
    except OSError:
        return False


class LlamaServer:
    """Starts llama-server for one model file, waits until it is ready, and stops it again."""

    def __init__(self, model_file: str, port: int, args: list[str], log_name: str):
        self.model_path = MODELS_DIR / model_file
        self.url = f"http://127.0.0.1:{port}"
        self.args = ["--port", str(port), "--host", "127.0.0.1", "--no-webui", *args]
        self.log_path = MODELS_DIR / f"{log_name}.log"
        self.process = None

    @property
    def pid_file(self) -> Path:
        return self.log_path.with_suffix(".pid")

    def start(self, wait: float = 180, detach: bool = False) -> "LlamaServer":
        """Start the server. With detach, it runs on its own and outlives this process (stop_detached() ends it)."""
        if is_ready(self.url):
            return self                       # already running (started earlier or by hand): left alone by stop()
        binary = shutil.which("llama-server")
        if not binary:
            raise OSError("llama-server not found; install it with: brew install llama.cpp")
        if not self.model_path.is_file():
            raise OSError(f"Model file missing: {self.model_path}. Download it from "
                          f"https://huggingface.co/{MODEL_REPOS[self.model_path.name]} (see scripts/serve_models.py)")
        log = self.log_path.open("w")
        self.process = subprocess.Popen([binary, "-m", str(self.model_path), *self.args],
                                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=detach)
        if detach:
            self.pid_file.write_text(str(self.process.pid))
        deadline = time.time() + wait
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise OSError(f"llama-server stopped while starting; see {self.log_path}")
            if is_ready(self.url):
                return self
            time.sleep(0.5)
        self.stop()
        raise OSError(f"llama-server was not ready after {wait:.0f}s; see {self.log_path}")

    def stop(self) -> None:
        """Stop the server only if this object started it; one that was already running is left alone."""
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.pid_file.unlink(missing_ok=True)
        self.process = None

    def stop_detached(self) -> None:
        """End a server started earlier with detach=True, through its saved process id."""
        if self.pid_file.is_file():
            try:
                os.kill(int(self.pid_file.read_text()), 15)
            except (ProcessLookupError, ValueError):
                pass
            self.pid_file.unlink(missing_ok=True)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def embedder_server(port: int = 8091) -> LlamaServer:
    # One slot and a 2,048-token window: enough for any chunk, and about 1.3 GB of memory
    # instead of the 5 GB the model's full 32K window would reserve.
    return LlamaServer(EMBEDDER_FILE, port, ["--embedding", "--pooling", "last", "-c", "2048", "-b", "2048",
                                             "-ub", "2048", "-np", "1", "-ngl", "all"], "embedder")


def decoder_server(port: int = 8092) -> LlamaServer:
    # Thinking off; one slot and no prompt cache, so one answer can never leak into the next
    # (open llama.cpp issues report state carrying over between requests for this model family).
    return LlamaServer(DECODER_FILE, port, ["-c", "4096", "-np", "1", "-ngl", "all", "--reasoning", "off",
                                            "--no-cache-prompt", "--cache-ram", "0"], "decoder")


class LlamaEmbedder:
    """Meaning vectors from the embedding server, unit length, as float32 numpy arrays."""

    def __init__(self, url: str = EMBEDDER_URL, instruction: str = QUERY_INSTRUCTION, batch_size: int = 16,
                 timeout: float = 120):
        self.url, self.instruction, self.batch_size, self.timeout = url, instruction, batch_size, timeout

    @property
    def model_id(self) -> str:
        """Names the exact model file, so vectors from different models are never mixed in one index."""
        props = _request(f"{self.url}/props", timeout=5)
        return Path(props.get("model_path", "unknown")).name

    def _embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            reply = _request(f"{self.url}/v1/embeddings", {"input": batch}, timeout=self.timeout)
            items = sorted(reply.get("data", []), key=lambda item: item["index"])
            if len(items) != len(batch):
                raise OSError(f"Embedding server returned {len(items)} vectors for {len(batch)} texts")
            rows += [item["embedding"] for item in items]
        vectors = np.asarray(rows, dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise OSError("Embedding server returned non-finite numbers; restart it before continuing")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise OSError("Embedding server returned an empty vector")
        return vectors / norms

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(list(texts))       # documents get no instruction

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([f"Instruct: {self.instruction}\nQuery:{text}"])[0]


def chat(messages: list[dict], url: str = DECODER_URL, max_tokens: int = 320, temperature: float = 0.0,
         seed: int = 42, timeout: float = 120) -> str:
    """One reply from the answering model. Greedy and seeded, so a repeat gives the same text."""
    reply = _request(f"{url}/v1/chat/completions", {
        "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
        # A repeat penalty of 1.1 was tried (dev round 2) and undone: it added unsafe "yes" answers.
        "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=timeout)
    try:
        return reply["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise OSError("Answer server sent a reply without a message") from exc
