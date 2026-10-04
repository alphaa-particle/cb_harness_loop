"""Assistants backed by a language model.

Every answer follows the same path, and only the last step needs a model:

    question -> evidence (rules first) -> prompt -> model -> guard -> answer

The model is never asked a question without evidence in its prompt. The two
classes differ only in how they run the model: a local Hugging Face model, or
a running Ollama server.
"""

from __future__ import annotations

import re

from engine.domain_pack import DomainPack
from engine.enforcement import (ANSWERED, BLOCKED, MODEL_ERROR, NO_EVIDENCE, RULES_ONLY, UNCLEAR,
                                ResponseGuard, build_messages, rules_reply)
from engine.retriever import Retriever
from engine.schemas import AssistantAnswer, RetrievedChunk

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _looks_like_gibberish(question: str) -> bool:
    q = question.strip().lower()
    if not q:
        return True
    if q in {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay"}:
        return False
    # English vowel ratios do not describe Hindi or other writing systems.
    # Let retrieval handle those inputs, rather than declaring every query in
    # a non-Latin script to be gibberish.
    if any(ch.isalpha() and not ch.isascii() for ch in q):
        return False
    words = re.findall(r"[a-zA-Z]+", q)
    if not words:
        return True
    letters = "".join(words)
    if len(letters) >= 10 and sum(ch in "aeiou" for ch in letters) / len(letters) < 0.22:
        return True
    # Long words with almost no vowels look like keyboard mashing. Ordinary words
    # ("monthly") and typos ("schems") can look like that too, so it only counts
    # when such words make up at least half of the input.
    nonsense = sum(1 for word in words if len(word) >= 6 and sum(ch in "aeiouy" for ch in word) <= 1)
    return nonsense >= max(2, len(words) / 2)


class EnforcedAssistant:
    """The shared path. Subclasses implement _generate(messages) -> str."""

    def __init__(self, pack: DomainPack, retriever: Retriever, max_new_tokens: int = 180):
        self.pack = pack
        self.retriever = retriever
        self.max_new_tokens = max_new_tokens
        self.guard = ResponseGuard(pack)

    def _generate(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def answer(self, question: str) -> AssistantAnswer:
        if _looks_like_gibberish(question):
            return AssistantAnswer(self.pack.prompts["unclear"], [], UNCLEAR)
        chunks = self.retriever.retrieve_evidence(question)
        if not chunks:
            return AssistantAnswer(self.pack.prompts["no_evidence"], [], NO_EVIDENCE)
        return self.answer_with_context(question, chunks)

    def answer_with_context(self, question: str, chunks: list[RetrievedChunk]) -> AssistantAnswer:
        try:
            text = self._generate(build_messages(self.pack, question, chunks))
        except OSError:      # the model server is down, unreachable or timed out
            text = ""
        if not text.strip():
            # A silent or broken model must not look like an answer; the person still gets the rules.
            return AssistantAnswer(rules_reply(self.pack, chunks), chunks, MODEL_ERROR)
        if self.guard.violations(text):
            return AssistantAnswer(self.guard.replacement(chunks), chunks, BLOCKED)
        return AssistantAnswer(text, chunks, ANSWERED)


class RulesOnlyAssistant(EnforcedAssistant):
    """No model at all: search, then show the matching rules word for word."""

    def answer_with_context(self, question: str, chunks: list[RetrievedChunk]) -> AssistantAnswer:
        return AssistantAnswer(rules_reply(self.pack, chunks), chunks, RULES_ONLY)


class LlamaCppAssistant(EnforcedAssistant):
    """A quantised model served by llama.cpp's llama-server (see engine/llama_cpp.py)."""

    def __init__(self, pack: DomainPack, retriever: Retriever, model_name: str | None = None,
                 max_new_tokens: int = 320, url: str | None = None):
        super().__init__(pack, retriever, max_new_tokens)
        from engine.llama_cpp import DECODER_URL
        self.url = url or DECODER_URL
        self.model_name = model_name   # informational: the server decides which file it serves

    def _generate(self, messages: list[dict]) -> str:
        from engine.llama_cpp import chat
        return _strip_think(chat(messages, url=self.url, max_tokens=self.max_new_tokens))


class TransformersAssistant(EnforcedAssistant):
    """Any local Hugging Face causal language model."""

    def __init__(
        self,
        pack: DomainPack,
        retriever: Retriever,
        model_name: str = "models/qwen-0.8b",
        max_new_tokens: int = 180,
        _model=None,
        _tokenizer=None,
    ):
        super().__init__(pack, retriever, max_new_tokens)
        self.model_name = model_name
        if _model is not None and _tokenizer is not None:
            self.tokenizer, self.model = _tokenizer, _model
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", trust_remote_code=True, local_files_only=True,
            device_map="auto")
        self.model.eval()

    def _generate(self, messages: list[dict]) -> str:
        import torch

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            # Greedy decoding, so the same question always gets the same answer.
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id, do_sample=False)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return _strip_think(self.tokenizer.decode(new_ids, skip_special_tokens=True))


class OllamaAssistant(EnforcedAssistant):
    """A model served by a running Ollama server."""

    def __init__(
        self,
        pack: DomainPack,
        retriever: Retriever,
        model_name: str = "qwen3:0.6b",
        host: str = "http://localhost:11434",
        max_new_tokens: int = 180,
    ):
        super().__init__(pack, retriever, max_new_tokens)
        self.model_name = model_name
        self.host = host

    def _generate(self, messages: list[dict]) -> str:
        import requests

        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0, "repeat_penalty": 1.1,
                            "num_predict": self.max_new_tokens},
            },
            timeout=180,
        )
        resp.raise_for_status()
        return _strip_think(resp.json().get("message", {}).get("content", ""))
