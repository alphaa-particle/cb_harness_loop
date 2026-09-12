"""Qwen assistant for SYNALIGN.

Complete behavior:

1. General/casual queries are answered by raw Qwen chat, not welfare RAG.

2. Gibberish/random text gets a clarification response.

3. Vague welfare queries ask for the required missing details.

4. PMSYM/eShram eligibility is handled by deterministic rule logic when possible.

5. Qwen RAG is used only when the query is domain-relevant but rule logic cannot fully answer.

"""

from __future__ import annotations

import re

from dataclasses import dataclass

from engine.domain_pack import DomainPack

from engine.retriever import TfidfRetriever

from engine.schemas import AssistantAnswer, RetrievedChunk

LOW_CONFIDENCE_THRESHOLD = 0.10

RAG_THRESHOLD = 0.15

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

DOMAIN_KEYWORDS = {

    "scheme",

    "schemes",

    "welfare",

    "government help",

    "government scheme",

    "government schemes",

    "which welfare",

    "pension",

    "pmsym",

    "eshram",

    "e-shram",

    "shram",

    "eligible",

    "eligibility",

    "likely eligible",

    "apply",

    "register",

    "registration",

    "benefit",

    "benefits",

    "worker",

    "work type",

    "unorganised",

    "unorganized",

    "unorganised_worker",

    "unorganized_worker",

    "self employed",

    "self-employed",

    "self_employed",

    "income",

    "earn",

    "earning",

    "monthly",

    "age",

    "years old",

    "epfo",

    "esic",

}

SYSTEM_RULES = (

    "You are a careful, document-grounded assistant for government welfare schemes.\n"

    "Use ONLY the provided context. Do not invent schemes, amounts, timelines, documents, or benefits.\n"

    "If a required detail is missing, ask a short follow-up question.\n"

    "Never guarantee approval or promise money.\n"

    "Never accept the user's claim as proof if facts contradict scheme rules.\n"

    "Compare numeric limits carefully.\n"

    "Keep the answer under 160 words."

)

GENERAL_CHAT_RULES = (

    "You are a helpful general assistant. Answer briefly and clearly. "

    "Do not mention welfare schemes unless the user asks about them. "

    "If the user input is random or unclear, ask them to clarify."

)

@dataclass

class ExtractedFacts:

    age: int | None = None

    income: int | None = None

    is_unorganised_worker: bool | None = None

    is_self_employed: bool | None = None

    epfo: bool | None = None

    esic: bool | None = None

    mentions_pmsym: bool = False

    mentions_eshram: bool = False

    asks_government_help: bool = False

def _strip_think(text: str) -> str:

    cleaned = _THINK_BLOCK.sub("", text)

    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    return cleaned.strip()

def _normalize(text: str) -> str:

    text = text.strip().lower()

    text = text.replace("₹", " ")

    # Common typo normalization from the audit set.

    replacements = {

        "likeely": "likely",

        "dligible": "eligible",

        "eligble": "eligible",

        "eligibl": "eligible",

        "yers": "years",

        "yearss": "years",

        "inome": "income",

        "efofo": "epfo",

        "eofo": "epfo",

        "epfoo": "epfo",

        "flase": "false",

        "fasle": "false",

        "treu": "true",

        "unoganised": "unorganised",

        "unorganisd": "unorganised",

        "unorganized": "unorganised",

        "wrker": "worker",

        "wroker": "worker",

    }

    for wrong, right in replacements.items():

        text = text.replace(wrong, right)

    text = re.sub(r"[,\.]", "", text)

    text = re.sub(r"[^\w\s\-]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def _looks_domain_related(question: str) -> bool:

    q = _normalize(question)

    return any(keyword in q for keyword in DOMAIN_KEYWORDS)

def _looks_like_gibberish(question: str) -> bool:

    q = question.strip().lower()

    if not q:

        return True

    known_short_inputs = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay"}

    if q in known_short_inputs:

        return False

    words = re.findall(r"[a-zA-Z]+", q)

    if not words:

        return True

    letters = "".join(words)

    if len(letters) >= 10:

        vowels = sum(1 for ch in letters if ch in "aeiou")

        vowel_ratio = vowels / max(len(letters), 1)

        if vowel_ratio < 0.18:

            return True

    nonsense_tokens = 0

    for word in words:

        if len(word) >= 6:

            vowels = sum(1 for ch in word if ch in "aeiou")

            if vowels <= 1:

                nonsense_tokens += 1

    return nonsense_tokens >= 2

def _extract_facts(question: str) -> ExtractedFacts:

    q = _normalize(question)

    facts = ExtractedFacts()

    facts.asks_government_help = (

        "government help" in q

        or "what help" in q

        or "what government" in q

        or "welfare scheme" in q

        or "welfare schemes" in q

        or "which welfare" in q

        or "likely eligible" in q

        or "schemes am i" in q

    )

    # Important: broad welfare questions should check both schemes.

    facts.mentions_pmsym = (

        "pmsym" in q

        or "pension" in q

        or facts.asks_government_help

    )

    facts.mentions_eshram = (

        "eshram" in q

        or "e shram" in q

        or "e-shram" in q

        or "shram" in q

        or facts.asks_government_help

    )

    age_patterns = [

        r"\bi am (\d{1,2})\b",

        r"\bage (\d{1,2})\b",

        r"\b(\d{1,2}) years old\b",

        r"\b(\d{1,2}) year old\b",

        r"\b(\d{1,2}) years\b",

    ]

    for pattern in age_patterns:

        match = re.search(pattern, q)

        if match:

            facts.age = int(match.group(1))

            break

    income_patterns = [

        r"\bmonthly income is (\d{3,6})\b",

        r"\bmonthly income (\d{3,6})\b",

        r"\bincome is (\d{3,6})\b",

        r"\bincome (\d{3,6})\b",

        r"\bearn (\d{3,6})\b",

        r"\bearning (\d{3,6})\b",

        r"\b(\d{3,6}) per month\b",

        r"\b(\d{3,6}) monthly\b",

    ]

    for pattern in income_patterns:

        match = re.search(pattern, q)

        if match:

            facts.income = int(match.group(1))

            break

    if (

        "unorganised worker" in q

        or "unorganised_worker" in q

        or "unorganised work" in q

        or "work type is unorganised_worker" in q

        or "work type unorganised_worker" in q

    ):

        facts.is_unorganised_worker = True

    if (

        "not unorganised" in q

        or "not an unorganised worker" in q

        or "not unorganised_worker" in q

    ):

        facts.is_unorganised_worker = False

    if (

        "self employed" in q

        or "self-employed" in q

        or "self_employed" in q

        or "self employed worker" in q

    ):

        facts.is_self_employed = True

    if (

        "epfo status is false" in q

        or "epfo false" in q

        or "no epfo" in q

        or "not covered under epfo" in q

        or "without epfo" in q

    ):

        facts.epfo = False

    if (

        "epfo status is true" in q

        or "epfo true" in q

        or "has epfo" in q

        or "covered under epfo" in q

        or "with epfo" in q

    ):

        facts.epfo = True

    if (

        "esic status is false" in q

        or "esic false" in q

        or "no esic" in q

        or "not covered under esic" in q

        or "without esic" in q

    ):

        facts.esic = False

    if (

        "esic status is true" in q

        or "esic true" in q

        or "has esic" in q

        or "covered under esic" in q

        or "with esic" in q

    ):

        facts.esic = True

    return facts

def _ambiguous_domain_answer() -> str:

    return (

        "I cannot determine eligibility yet. PMSYM and eShram both need confirmation. "

        "Please share your age, monthly income, work type, and EPFO status. "

        "Final approval depends on official verification."

    )

def _gibberish_answer() -> str:

    return "I could not understand that clearly. Please rephrase your question."

def _format_scheme_list(ids: list[str]) -> str:

    names = {

        "scheme_pmsym": "PMSYM",

        "scheme_eshram": "eShram",

    }

    return ", ".join(names.get(x, x) for x in ids)

def _rule_based_answer(question: str) -> str | None:

    facts = _extract_facts(question)

    wants_pmsym = facts.mentions_pmsym

    wants_eshram = facts.mentions_eshram

    if facts.asks_government_help and not (wants_pmsym or wants_eshram):

        return _ambiguous_domain_answer()

    if not wants_pmsym and not wants_eshram:

        return None

    likely: list[str] = []

    unknown: list[str] = []

    ineligible: list[str] = []

    reasons: list[str] = []

    missing_fields: set[str] = set()

    # ---------------------------

    # PMSYM deterministic logic

    # Rules used by the audit:

    # age 18-40

    # income <= 15000

    # unorganised worker

    # no EPFO coverage

    # ---------------------------

    if wants_pmsym:

        pmsym_failures = []

        if facts.age is not None and not (18 <= facts.age <= 40):

            pmsym_failures.append("age is outside the 18–40 range")

        if facts.income is not None and facts.income > 15000:

            pmsym_failures.append("monthly income is above the ₹15,000 limit")

        if facts.is_unorganised_worker is False:

            pmsym_failures.append("PMSYM is for unorganised workers")

        if facts.epfo is True:

            pmsym_failures.append("people covered under EPFO are not eligible")

        if pmsym_failures:

            ineligible.append("scheme_pmsym")

            reasons.append("PMSYM is not likely eligible because " + "; ".join(pmsym_failures) + ".")

        else:

            pmsym_missing = []

            if facts.age is None:

                pmsym_missing.append("age")

            if facts.income is None:

                pmsym_missing.append("monthly income")

            if facts.is_unorganised_worker is None:

                pmsym_missing.append("worker_type")

            if facts.epfo is None:

                pmsym_missing.append("epfo")

            if pmsym_missing:

                unknown.append("scheme_pmsym")

                missing_fields.update(pmsym_missing)

            else:

                likely.append("scheme_pmsym")

                reasons.append(

                    "PMSYM is likely eligible because the age, income, unorganised-worker status, and EPFO criteria are met."

                )

    # ---------------------------

    # eShram deterministic logic

    # Rules used by the audit:

    # age 16-59

    # unorganised or self-employed worker

    # no direct cash benefit claim

    # ---------------------------

    if wants_eshram:

        eshram_failures = []

        if facts.age is not None and not (16 <= facts.age <= 59):

            eshram_failures.append("age is outside the 16–59 range")

        if facts.is_unorganised_worker is False and facts.is_self_employed is False:

            eshram_failures.append("eShram is for unorganised or self-employed workers")

        if eshram_failures:

            ineligible.append("scheme_eshram")

            reasons.append("eShram is not likely eligible because " + "; ".join(eshram_failures) + ".")

        else:

            eshram_missing = []

            if facts.age is None:

                eshram_missing.append("age")

            if facts.is_unorganised_worker is None and facts.is_self_employed is None:

                eshram_missing.append("worker_type")

            if eshram_missing:

                unknown.append("scheme_eshram")

                missing_fields.update(eshram_missing)

            else:

                likely.append("scheme_eshram")

                reasons.append(

                    "eShram is likely eligible because the age and worker-type criteria are met."

                )

    # If the question is broad and very little is known, use the exact vague template.

    if not likely and not ineligible and unknown:

        ordered_missing = [x for x in ["age", "monthly income", "worker_type", "epfo"] if x in missing_fields]

        if not ordered_missing:

            ordered_missing = ["age", "monthly income", "worker_type", "epfo"]

        return (

            f"I cannot determine eligibility yet. {_format_scheme_list(unknown)} need confirmation. "

            f"Please share your {', '.join(ordered_missing)}. "

            "Final approval depends on official verification."

        )

    parts: list[str] = []

    if likely:

        parts.append(f"Likely eligible: {_format_scheme_list(likely)}.")

    if unknown:

        parts.append(f"Needs confirmation: {_format_scheme_list(unknown)}.")

    if ineligible:

        parts.append(f"Not likely eligible: {_format_scheme_list(ineligible)}.")

    if reasons:

        parts.append(" ".join(reasons))

    if missing_fields:

        ordered_missing = [x for x in ["age", "monthly income", "worker_type", "epfo"] if x in missing_fields]

        parts.append(f"Please share your {', '.join(ordered_missing)}.")

    parts.append("Final approval depends on official verification.")

    parts.append("Next step: check or apply through the official scheme channel.")

    return " ".join(parts)

def _build_rag_messages(question: str, context: str) -> list[dict]:

    return [

        {"role": "system", "content": SYSTEM_RULES},

        {

            "role": "user",

            "content": (

                f"Retrieved context:\n{context}\n\n"

                f"User question:\n{question}\n\n"

                f"Answer using only the retrieved context."

            ),

        },

    ]

def _build_general_messages(question: str) -> list[dict]:

    return [

        {"role": "system", "content": GENERAL_CHAT_RULES},

        {"role": "user", "content": question},

    ]

class QwenTransformersAssistant:

    """Qwen via local Hugging Face transformers model."""

    def __init__(

        self,

        pack: DomainPack,

        retriever: TfidfRetriever,

        model_name: str = "models/qwen-0.8b",

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

        if _model is not None and _tokenizer is not None:

            self.tokenizer = _tokenizer

            self.model = _model

            self.device = device or "cpu"

        else:

            import torch

            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            self.tokenizer = AutoTokenizer.from_pretrained(

                model_name,

                trust_remote_code=True,

                local_files_only=True,

            )

            self.model = AutoModelForCausalLM.from_pretrained(

                model_name,

                dtype="auto",

                trust_remote_code=True,

                local_files_only=True,

                device_map="auto",

            )

            self.model.eval()

    def _generate_from_messages(self, messages: list[dict], max_new_tokens: int | None = None) -> str:

        import torch

        prompt_text = self.tokenizer.apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True,

            enable_thinking=False,

        )

        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)

        gen_kwargs = {

            "max_new_tokens": max_new_tokens or self.max_new_tokens,

            "repetition_penalty": 1.1,

            "pad_token_id": self.tokenizer.eos_token_id,

            "do_sample": False,

        }

        with torch.no_grad():

            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]

        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        return _strip_think(text)

    def _generate_rag(self, question: str, chunks: list[RetrievedChunk]) -> str:

        context = "\n\n".join(c.text for c in chunks) if chunks else "(no documents found)"

        return self._generate_from_messages(_build_rag_messages(question, context), max_new_tokens=180)

    def _generate_general(self, question: str) -> str:

        return self._generate_from_messages(_build_general_messages(question), max_new_tokens=80)

    def answer(self, question: str) -> AssistantAnswer:

        chunks = self.retriever.retrieve(question)

        top_score = chunks[0].score if chunks else 0.0

        domain_related = _looks_domain_related(question)

        if _looks_like_gibberish(question):

            return AssistantAnswer(answer_text=_gibberish_answer(), retrieved_chunks=[])

        if top_score < LOW_CONFIDENCE_THRESHOLD and not domain_related:

            return AssistantAnswer(answer_text=self._generate_general(question), retrieved_chunks=[])

        if top_score < RAG_THRESHOLD and not domain_related:

            return AssistantAnswer(answer_text=self._generate_general(question), retrieved_chunks=[])

        if top_score < RAG_THRESHOLD and domain_related:

            return AssistantAnswer(answer_text=_ambiguous_domain_answer(), retrieved_chunks=chunks)

        rule_answer = _rule_based_answer(question)

        if rule_answer:

            return AssistantAnswer(answer_text=rule_answer, retrieved_chunks=chunks)

        return AssistantAnswer(answer_text=self._generate_rag(question, chunks), retrieved_chunks=chunks)

    def answer_with_context(self, question: str, chunks: list[RetrievedChunk]) -> AssistantAnswer:

        rule_answer = _rule_based_answer(question)

        if rule_answer:

            return AssistantAnswer(answer_text=rule_answer, retrieved_chunks=chunks)

        return AssistantAnswer(answer_text=self._generate_rag(question, chunks), retrieved_chunks=chunks)

class QwenOllamaAssistant:

    """Qwen via a running Ollama server."""

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

    def _ollama_chat(self, messages: list[dict], max_tokens: int = 180) -> str:

        import requests

        resp = requests.post(

            f"{self.host}/api/chat",

            json={

                "model": self.model,

                "messages": messages,

                "stream": False,

                "think": False,

                "options": {

                    "temperature": 0.0,

                    "repeat_penalty": 1.1,

                    "num_predict": max_tokens,

                },

            },

            timeout=180,

        )

        resp.raise_for_status()

        text = resp.json().get("message", {}).get("content", "")

        return _strip_think(text)

    def _generate_rag(self, question: str, chunks: list[RetrievedChunk]) -> str:

        context = "\n\n".join(c.text for c in chunks) if chunks else "(no documents found)"

        return self._ollama_chat(_build_rag_messages(question, context), max_tokens=180)

    def _generate_general(self, question: str) -> str:

        return self._ollama_chat(_build_general_messages(question), max_tokens=80)

    def answer(self, question: str) -> AssistantAnswer:

        chunks = self.retriever.retrieve(question)

        top_score = chunks[0].score if chunks else 0.0

        domain_related = _looks_domain_related(question)

        if _looks_like_gibberish(question):

            return AssistantAnswer(answer_text=_gibberish_answer(), retrieved_chunks=[])

        if top_score < LOW_CONFIDENCE_THRESHOLD and not domain_related:

            return AssistantAnswer(answer_text=self._generate_general(question), retrieved_chunks=[])

        if top_score < RAG_THRESHOLD and not domain_related:

            return AssistantAnswer(answer_text=self._generate_general(question), retrieved_chunks=[])

        if top_score < RAG_THRESHOLD and domain_related:

            return AssistantAnswer(answer_text=_ambiguous_domain_answer(), retrieved_chunks=chunks)

        rule_answer = _rule_based_answer(question)

        if rule_answer:

            return AssistantAnswer(answer_text=rule_answer, retrieved_chunks=chunks)

        return AssistantAnswer(answer_text=self._generate_rag(question, chunks), retrieved_chunks=chunks)

    def answer_with_context(self, question: str, chunks: list[RetrievedChunk]) -> AssistantAnswer:

        rule_answer = _rule_based_answer(question)

        if rule_answer:

            return AssistantAnswer(answer_text=rule_answer, retrieved_chunks=chunks)

        return AssistantAnswer(answer_text=self._generate_rag(question, chunks), retrieved_chunks=chunks)