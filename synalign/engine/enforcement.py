"""What the model is given, and what it is not allowed to say.

Before the model runs, build_messages() writes its prompt: the fixed
instructions from the domain pack, then the evidence chosen by
Retriever.retrieve_evidence() word for word, then the question. The
training data is written with the same function, so a model is trained on the
same instructions and layout it will later be given.

After the model runs, ResponseGuard looks for promises no answer may make.
It recognises only the phrasings listed under forbidden_claims in the domain
pack's aliases.yaml; it cannot tell whether an eligibility verdict is right.

Nothing here loads or calls a model.
"""

from engine.domain_pack import DomainPack
from engine.evaluator import Evaluator
from engine.schemas import RetrievedChunk

# AssistantAnswer.mode values.
ANSWERED = "answered"        # the model's answer, passed by the guard
NO_EVIDENCE = "no_evidence"  # search found nothing, so the model was not asked
UNCLEAR = "unclear"          # the input was not readable as a question
BLOCKED = "blocked"          # the model's answer broke a rule and was replaced
RULES_ONLY = "rules_only"    # no model is in use: the matching rules are shown as they are
MODEL_ERROR = "model_error"  # the model failed or said nothing: the matching rules are shown instead


def format_evidence(chunks: list[RetrievedChunk]) -> str:
    """Source text, unchanged, headed by its title when the text does not carry it."""
    return "\n\n".join(c.text if not c.title or c.title in c.text else f"{c.title}\n{c.text}"
                       for c in chunks)


def build_messages(pack: DomainPack, question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {"role": "system", "content": pack.prompts["system"]},
        {
            "role": "user",
            "content": (
                f"Retrieved context:\n{format_evidence(chunks)}\n\n"
                f"User question:\n{question}\n\n"
                f"Answer using only the retrieved context."
            ),
        },
    ]


def rules_reply(pack: DomainPack, chunks: list[RetrievedChunk]) -> str:
    """The answer when no model writes one: a fixed line, then the evidence word for word."""
    return f"{pack.prompts['rules_only']}\n\n{format_evidence(chunks)}"


class ResponseGuard:
    """Replaces an answer that makes a forbidden claim with the rules themselves."""

    def __init__(self, pack: DomainPack):
        # The audit's own matcher, so the guard and the grader agree on what a
        # forbidden claim is, including "approval is not guaranteed" being fine.
        self._evaluator = Evaluator(pack)
        self._claims = list(pack.forbidden_claims())
        self._blocked = pack.prompts["blocked"]

    def violations(self, answer_text: str) -> list[str]:
        return self._evaluator.forbidden_claim_violations(answer_text, self._claims)

    def replacement(self, chunks: list[RetrievedChunk]) -> str:
        return f"{self._blocked}\n\n{format_evidence(chunks)}"
