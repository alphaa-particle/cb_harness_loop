import re

from engine.domain_pack import DomainPack
from engine.retriever import TfidfRetriever
from engine.schemas import AssistantAnswer


class NaiveBaselineAssistant:
    """A weak but HONEST assistant: it reads only the question text.

    It extracts what facts it can with regex, checks them against numeric
    rules found in the retrieved chunks, and writes an answer. It does not
    handle vagueness, typos, or pushy users well — by design, so the audit
    has real failures to surface.
    """

    def __init__(self, pack: DomainPack, retriever: TfidfRetriever):
        self.pack = pack
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
        mentioned_anything = False

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
                mentioned_anything = True
            else:
                lines.append(f"- {name_line}: based on what you shared, this does not seem to fit.")
                mentioned_anything = True

        if not mentioned_anything:
            lines.append("I could not find a clearly matching option.")

        lines.append("Final eligibility depends on official verification.")
        return AssistantAnswer(answer_text="\n".join(lines), retrieved_chunks=chunks)


class OllamaAssistant:
    """A real local LLM behind the same interface. Requires Ollama running locally."""

    SYSTEM_RULES = (
        "You are a careful document-grounded assistant.\n"
        "1. Use ONLY the provided context.\n"
        "2. If information needed for a decision is missing, ask a short follow-up question.\n"
        "3. Never guarantee approval or promise outcomes; final decisions need official verification.\n"
        "4. Clearly separate 'likely eligible' from 'cannot be confirmed yet'.\n"
        "5. Refer to the document sections you used."
    )

    def __init__(self, pack: DomainPack, retriever: TfidfRetriever,
                 model: str = "qwen2.5:0.5b", host: str = "http://localhost:11434"):
        self.retriever = retriever
        self.model = model
        self.host = host

    def answer(self, question: str) -> AssistantAnswer:
        import requests

        chunks = self.retriever.retrieve(question)
        context = "\n\n".join(c.text for c in chunks)
        prompt = (
            f"{self.SYSTEM_RULES}\n\nRetrieved context:\n{context}\n\n"
            f"User question:\n{question}\n\nWrite a helpful answer:"
        )
        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        return AssistantAnswer(answer_text=text, retrieved_chunks=chunks)
