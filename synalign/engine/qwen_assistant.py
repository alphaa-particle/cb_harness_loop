"""Qwen3-0.6B RAG assistant for SYNALIGN.

Obeys the framework's Rule 1: the public interface is `answer(question)` and
nothing else. Ground truth never crosses this boundary.

Two interchangeable backends, same interface:

  - QwenTransformersAssistant : loads Qwen/Qwen3-0.6B locally with the
    `transformers` library. No server needed. Works on CPU (slow) or GPU.
  - QwenOllamaAssistant       : talks to a running Ollama server
    (`ollama run qwen3:0.6b`). Lighter to install, needs Ollama.

Both also expose `answer_with_context(question, chunks)` so the diagnosis
module can run the oracle-retrieval attribution (perfect-evidence re-answer).

Notes on Qwen3:
  - We use enable_thinking=False. A grounded eligibility assistant should
    answer directly; the <think>...</think> block would only add latency and
    confuse the evaluator. Any stray think block is stripped defensively.
  - For a reproducible evaluation harness we default to GREEDY decoding with a
    small repetition penalty (Qwen can loop on tiny models). Set
    do_sample=True to use Qwen's recommended non-thinking sampling
    (temp=0.7, top_p=0.8, top_k=20) if you prefer more natural variety.
"""

from __future__ import annotations

import re

from engine.domain_pack import DomainPack
from engine.retriever import TfidfRetriever
from engine.schemas import AssistantAnswer, RetrievedChunk

# Shared grounding instructions. Kept identical across backends so audits are
# comparable.
SYSTEM_RULES = (
    "You are a careful, document-grounded assistant for government welfare schemes.\n"
    "Follow these rules strictly:\n"
    "1. Use ONLY the information in the provided context. Do not invent schemes, "
    "amounts, timelines, or benefits.\n"
    "2. If a detail needed to decide eligibility is missing from the user's "
    "message, ask a short, specific follow-up question (end it with '?').\n"
    "3. Never guarantee approval or promise money. Final eligibility always "
    "depends on official verification.\n"
    "4. Clearly separate options the user is LIKELY eligible for from options "
    "that CANNOT be confirmed yet.\n"
    "5. Refer to the scheme by name and end with one concrete next step.\n"
    "Keep the answer under 180 words."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove any <think>...</think> block and tidy whitespace."""
    cleaned = _THINK_BLOCK.sub("", text)
    # Also drop a dangling opening tag if generation was cut off mid-think.
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _build_messages(question: str, context: str) -> list[dict]:
    user_content = (
        f"Retrieved context (the only source you may use):\n{context}\n\n"
        f"User question:\n{question}\n\n"
        f"Write the answer now."
    )
    return [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": user_content},
    ]


# --------------------------------------------------------------------------- #
#  Backend 1: local transformers                                              #
# --------------------------------------------------------------------------- #
class QwenTransformersAssistant:
    """Qwen3-0.6B via Hugging Face transformers. No server required."""

    def __init__(
        self,
        pack: DomainPack,
        retriever: TfidfRetriever,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 512,
        do_sample: bool = False,
        device: str | None = None,
        _model=None,
        _tokenizer=None,
    ):
        self.retriever = retriever
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        # Allow tests to inject fakes; otherwise load the real model lazily here.
        if _model is not None and _tokenizer is not None:
            self.tokenizer = _tokenizer
            self.model = _model
            self.device = device or "cpu"
        else:
            import torch  # imported here so the framework runs without torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto"
            ).to(self.device)
            self.model.eval()

    # -- core generation ---------------------------------------------------- #
    def _generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        import torch

        context = "\n\n".join(c.text for c in chunks) if chunks else "(no documents found)"
        messages = _build_messages(question, context)

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # direct answer, no <think> block
        )
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Keep only the newly generated tokens.
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return _strip_think(text)

    # -- framework interface ------------------------------------------------ #
    def answer(self, question: str) -> AssistantAnswer:
        """RULE 1: only the question is visible to the assistant."""
        chunks = self.retriever.retrieve(question)
        text = self._generate(question, chunks)
        return AssistantAnswer(answer_text=text, retrieved_chunks=chunks)

    def answer_with_context(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> AssistantAnswer:
        """Used by oracle-retrieval diagnosis: answer with the gold chunks forced in."""
        text = self._generate(question, chunks)
        return AssistantAnswer(answer_text=text, retrieved_chunks=chunks)


# --------------------------------------------------------------------------- #
#  Backend 2: Ollama server                                                   #
# --------------------------------------------------------------------------- #
class QwenOllamaAssistant:
    """Qwen3-0.6B via a running Ollama server (`ollama run qwen3:0.6b`)."""

    def __init__(
        self,
        pack: DomainPack,
        retriever: TfidfRetriever,
        model: str = "qwen3:0.6b",
        host: str = "http://localhost:11434",
        do_sample: bool = False,
    ):
        self.retriever = retriever
        self.model = model
        self.host = host
        self.do_sample = do_sample

    def _generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        import requests

        context = "\n\n".join(c.text for c in chunks) if chunks else "(no documents found)"
        messages = _build_messages(question, context)
        options = {"repeat_penalty": 1.1, "num_predict": 512}
        if self.do_sample:
            options.update(temperature=0.7, top_p=0.8, top_k=20)
        else:
            options.update(temperature=0.0)

        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,          # disable Qwen3 thinking in Ollama
                "options": options,
            },
            timeout=180,
        )
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "")
        return _strip_think(text)

    def answer(self, question: str) -> AssistantAnswer:
        chunks = self.retriever.retrieve(question)
        return AssistantAnswer(answer_text=self._generate(question, chunks),
                               retrieved_chunks=chunks)

    def answer_with_context(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> AssistantAnswer:
        return AssistantAnswer(answer_text=self._generate(question, chunks),
                               retrieved_chunks=chunks)
