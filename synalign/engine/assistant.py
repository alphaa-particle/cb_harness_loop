import re

from engine.domain_pack import DomainPack
from engine.retriever import Retriever
from engine.schemas import AssistantAnswer


class NaiveBaselineAssistant:
    """A weak but HONEST assistant: it reads only the question text.

    It extracts what facts it can with regex, checks them against numeric
    rules found in the retrieved chunks, and writes an answer. It does not
    handle vagueness, typos, or pushy users well — by design, so the audit
    has real failures to surface.
    """

    def __init__(self, pack: DomainPack, retriever: Retriever):
        self.retriever = retriever

    # -- crude fact extraction from the question only ---------------------
    def _extract_facts(self, question: str) -> dict:
        facts: dict = {}
        q = question.lower()

        age = re.search(r"\b(\d{2})\s*(?:years?\s*old|yrs?\b|yr\b)", q)
        if age:
            facts["age"] = int(age.group(1))

        income = re.search(r"(?:income|earn|salary)\D{0,20}?(\d{4,6})", q)
        if income:
            facts["income"] = int(income.group(1))

        for wt in ("unorganised_worker", "salaried", "self_employed"):
            if wt.replace("_", " ") in q or wt in q:
                facts["worker_type"] = wt

        epfo = re.search(r"epfo\s*(?:status\s*)?(?:is\s*)?(true|false|yes|no)", q)
        if epfo:
            facts["epfo"] = epfo.group(1) in ("true", "yes")

        return facts

    def answer(self, question: str) -> AssistantAnswer:
        chunks = self.retriever.retrieve(question)
        facts = self._extract_facts(question)

        lines = ["Here is an assessment based on the documents I found."]

        for chunk in chunks:
            name_line = chunk.text.splitlines()[0].replace("#", "").strip()
            rules_ok = True

            # Naive numeric-rule checking against extracted facts.
            for low, high in re.findall(r"between (\d+) and (\d+)", chunk.text.lower()):
                if "age" in facts and not (int(low) <= facts["age"] <= int(high)):
                    rules_ok = False
            income_rule = re.search(r"income must be (\d+) or below", chunk.text.lower())
            if income_rule and "income" in facts and facts["income"] > int(income_rule.group(1)):
                rules_ok = False

            if rules_ok:
                lines.append(f"- {name_line}: this looks relevant to you and you may be eligible.")
            else:
                lines.append(f"- {name_line}: based on what you shared, this does not seem to fit.")

        if not chunks:
            lines.append("I could not find a clearly matching option.")

        lines.append("Final eligibility depends on official verification.")
        return AssistantAnswer(answer_text="\n".join(lines), retrieved_chunks=chunks)


BACKENDS = ("naive", "rules_only", "llama_cpp", "transformers", "ollama")
NO_MODEL_BACKENDS = ("naive", "rules_only")


def make_assistant(backend: str, pack: DomainPack, retriever: Retriever, **kwargs):
    """Build the assistant for a backend. Model libraries are imported only if needed."""
    if backend == "naive":
        return NaiveBaselineAssistant(pack, retriever)
    if backend == "rules_only":
        from engine.llm_assistant import RulesOnlyAssistant
        return RulesOnlyAssistant(pack, retriever)
    if backend == "llama_cpp":
        from engine.llm_assistant import LlamaCppAssistant
        return LlamaCppAssistant(pack, retriever, **kwargs)
    if backend == "transformers":
        from engine.llm_assistant import TransformersAssistant
        return TransformersAssistant(pack, retriever, **kwargs)
    if backend == "ollama":
        from engine.llm_assistant import OllamaAssistant
        return OllamaAssistant(pack, retriever, **kwargs)
    raise ValueError(f"Unknown assistant backend: {backend!r}; choose from {BACKENDS}")
